import copy
from pathlib import Path
import queue
import shutil
import tempfile
import threading


class DummyFile:
    """A file that knows its name and doesn't remove file on release call
    """

    def __init__(self, filename):
        self.filename = filename

    @property
    def name(self):
        return self.filename

    def release(self):
        pass


class ReleasableFile:
    """A file that knows its name and remove file on release call
    """

    def __init__(self, filename):
        self.filename = filename
        self.exist = self.filename.is_file()

    @property
    def name(self):
        assert self.exist, f'File {self.filename} doesn\'t exist'
        return self.filename

    def release(self):
        assert self.exist, f'File {self.filename} doesn\'t exist'
        self.filename.unlink()


def create_file_iterator(files,
                         cache_dir=None,
                         num_files_in_cache=5):
    files = list(Path(f) for f in files)
    if cache_dir is None:
        return FileIterator(files)
    iterator = FileIteratorWithCache(files,
                                     FileLoader(cache_dir),
                                     num_files_in_cache)
    if num_files_in_cache < len(files):
        return iterator
    # if we can cache all files then cache them and use the basic FileIterator
    new_files = [iterator.next() for _ in files]
    return FileIterator(new_files)


class FileIterator:
    """Allows iteration over a list of files

    To use:
    >>> loader = FileLoader(Path('/storage').glob('*.hdf5'))
    >>> loader.next()
    /tmp/0.hdf5
    >>> loader.next()
    /tmp/1.hdf5
    >>> loader.step() # does nothing
    >>> loader.next()
    /tmp/2.hdf5
    """

    def __init__(self,
                 files):
        self.files = copy.deepcopy(list(files))
        self.index = 0

    def next(self):
        result = self.files[self.index]
        self.index = (self.index + 1) % len(self.files)
        return DummyFile(result)

    def step(self):
        pass

    def reset(self):
        self.index = 0


class FileLoader:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def __call__(self, filename):
        with tempfile.NamedTemporaryFile(dir=self.cache_dir,
                                         suffix=filename.suffix,
                                         delete=False) as f:
            cached = Path(f.name)
        shutil.copyfile(filename, cached)
        return cached


class FileIteratorWithCache:
    """Loads files from a remote storage to the cache.

    The code doesn't remove previously cached files.

    To use:
    # loads the first 2 elements: 0.hdf5, 1.hdf5
    >>> loader = FileLoader(Path('/storage').glob('*.hdf5'),
                            FileLoader(Path('/tmp')), 2)
    >>> loader.next() # returns the first element
    /tmp/0.hdf5
    >>> loader.next() # returns the second element. The first
                      # one is still stored
    /tmp/1.hdf5
    >>> loader.step() # doesn't remove anything from the cache:
                      # 0.hdf5 1.hdf5, 2.hdf5
    >>> loader.next()
    /tmp/2.hdf5
    >>> loader.next() # error. 3.hdf5 is not in the cache.

    The desired state:

    Assume:
    - Processing of a single file takes 1 period
    - Loading of a single file takes 2 periods
    - Cache size is 3 files
    - Load cache is 1 file

    The expected behaviour when process_only_once is set to False
    Timestamp|   Cache  |Load cache|Used file
    =========================================
        0    |__, __, __|    F0    |
        1    |__, __, __|    F0    |
        2    |F0, __, __|    F1    |    F0
        3    |F0, __, __|    F1    |    F0 <- we second time use the same file
        4    |F0, F1, __|    F2    |    F1
        5    |F0, F1, __|    F2    |    F0 <- new cycle
        6    |F0, F1, F2|    F3    |    F1
        7    |F0, F1, F2|    F3    |    F2
        8    |F1, F2, F3|    F4    |    F3
        9    |F1, F2, F3|    F4    |    F1 <- new cycle

    The expected behaviour when process_only_once is set to True
    Timestamp|   Cache  |Load cache|Used file
    =========================================
        0    |__, __, __|    F0    |
        1    |__, __, __|    F0    |
        2    |F0, __, __|    F1    |    F0
        3    |__, __, __|    F1    |       <- waiting
        4    |F1, __, __|    F2    |    F1
        5    |__, __, __|    F2    |       <- waiting
        6    |F2, __, __|    F3    |    F2
        7    |__, __, __|    F3    |       <- waiting
        8    |F3, __, __|    F4    |    F3
        9    |__, __, __|    F4    |       <- waiting
    """

    def __init__(self,
                 remote_files,
                 file_loader,
                 num_files_to_cache=5,
                 num_non_cached_files=2,
                 process_only_once=True):
        """Initializes the iterator

        Args:
            remote_files:
                A list of the files to load
            file_loader:
                A callable that can copy the input file to a cache
            num_files_to_cache:
                A maximum number of files in a cache
            num_non_cached_files:
                A maximum number of downloaded, but not yet cached files
                to store. It is usefull if processing is slower than loading
            process_only_once:
                Controls allowance of multiple passes through the file.
        """
        def thread_function(request_queue, response_queue, file_loader):
            while True:
                remote = request_queue.get()
                if remote is None:
                    break
                response_queue.put(file_loader(remote))

        self.process_only_once = process_only_once
        self.remote_files = copy.deepcopy(list(remote_files))
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self._init_cache(num_files_to_cache)
        self.read_thread = threading.Thread(target=thread_function,
                                            args=(self.request_queue,
                                                  self.response_queue,
                                                  file_loader),
                                            daemon=True)
        self.read_thread.start()
        # files in the current cache
        self.cached_files = []
        # files that was loaded but not cached
        self.loaded_files = []
        self.loaded_files_lock = threading.Lock()

    def _init_cache(self, num_files_to_cache):
        num_files_to_cache = min(num_files_to_cache, len(self.remote_files))
        for i in range(num_files_to_cache):
            self.request_queue.put(self.remote_files[i])
        self.num_left = num_files_to_cache
        self.num_files_to_cache = num_files_to_cache
        self.cached_end = (i + 1) % len(self.remote_files)

    def next(self):
        """ Returns path of the next cached element.
        """
        # it prevents deadlock
        assert self.num_left > 0, "No cached files left"
        self.num_left -= 1
        return ReleasableFile(self.response_queue.get())

    def step(self):
        """ Performs a step of the sliding window.

        Loads the next non-cached one.
        """
        self.request_queue.put(self.remote_files[self.cached_end])
        self.cached_end = (self.cached_end + 1) % len(self.remote_files)
        self.num_left += 1

    def reset(self):
        # remove all cached files but not read files
        for i in range(self.num_left):
            self.next().release()
        self.num_left = 0
        self._init_cache(self.num_files_to_cache)
