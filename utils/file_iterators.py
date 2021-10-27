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


class FileIteratorWithCache:
    """Loads files from a remote storage to the cache.

    The code doesn't remove previously cached files.

    To use:
    # loads the first 2 elements: 0.hdf5, 1.hdf5
    >>> loader = FileLoader(Path('/storage').glob('*.hdf5'), Path('/tmp'), 2)
    >>> loader.next() # returns the first element
    /tmp/0.hdf5
    >>> loader.next() # returns the second element. The first
                      # one is still stored
    /tmp/1.hdf5
    >>> loader.step() # doesn't remov anything from the cache:
                      # 0.hdf5 1.hdf5, 2.hdf5
    >>> loader.next()
    /tmp/2.hdf5
    >>> loader.next() # error. 3.hdf5 is not in the cache.
    """

    def __init__(self,
                 remote_files,
                 cache_dir,
                 num_files_to_cache=5):
        def thread_function(cache_dir, request_queue, response_queue):
            cache_dir.mkdir(exist_ok=True, parents=True)
            while True:
                remote = request_queue.get()
                if remote is None:
                    break
                with tempfile.NamedTemporaryFile(dir=cache_dir,
                                                 suffix=remote.suffix,
                                                 delete=False) as f:
                    cached = Path(f.name)
                shutil.copyfile(remote, cached)
                response_queue.put(cached)

        self.remote_files = copy.deepcopy(list(remote_files))
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self._init_cache(num_files_to_cache)
        self.read_thread = threading.Thread(target=thread_function,
                                            args=(cache_dir,
                                                  self.request_queue,
                                                  self.response_queue))
        self.read_thread.start()

    def _init_cache(self, num_files_to_cache):
        for i in range(min(num_files_to_cache, len(self.remote_files))):
            self.request_queue.put(self.remote_files[i])
        self.num_files_to_cache = i
        self.num_left = i
        self.cached_end = i % len(self.remote_files)

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
