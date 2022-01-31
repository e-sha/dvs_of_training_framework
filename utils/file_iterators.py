from abc import ABC, abstractmethod
import copy
from pathlib import Path
import queue
import shutil
import tempfile
import threading


class CacheIsFullError(Exception):
    pass


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
        self.exist = self.filename.is_file
        # access to this field can be performed without a lock
        # as only the same thread can set in_use to True and remove the file
        self.in_use = True

    @property
    def name(self):
        assert self.exist(), f'File {self.filename} doesn\'t exist'
        return self.filename

    def release(self):
        assert self.exist(), f'File {self.filename} doesn\'t exist'
        self.in_use = False

    def is_in_use(self):
        assert self.exist(), f'File {self.filename} doesn\'t exist'
        return self.in_use

    def start_use(self):
        assert self.exist(), f'File {self.filename} doesn\'t exist'
        self.in_use = True

    def remove(self):
        assert self.exist(), f'File {self.filename} doesn\'t exist'
        assert not self.in_use, 'Currently used file cannot be removed'
        self.filename.unlink()


def create_file_iterator(files,
                         cache_dir=None,
                         num_files_in_cache=5,
                         process_only_once=True):
    files = list(Path(f) for f in files)
    if cache_dir is None:
        return FileIterator(files)
    if not process_only_once and num_files_in_cache < len(files):
        # if we can cache all files in the fast memory we should do it
        # even if process_only_once is set to False
        iterator_class = FileIteratorNonBlocking
    else:
        iterator_class = FileIteratorWithCache
    iterator = iterator_class(files,
                              FileLoader(cache_dir),
                              num_files_in_cache)
    if num_files_in_cache < len(files):
        return iterator
    # if we can cache all files then cache them and use the basic FileIterator
    new_files = [iterator.next().name for _ in files]
    print('!!!!!!!!!!!!!!!')
    print(new_files)
    print('!!!!!!!!!!!!!!!')
    print(len(new_files))
    return FileIterator(new_files)


class FileIterator:
    """Allows iteration over a list of files

    To use:
    >>> loader = FileIterator(Path('/storage').glob('*.hdf5'))
    >>> loader.next()
    /tmp/0.hdf5
    >>> loader.next()
    /tmp/1.hdf5
    >>> loader.next()
    /tmp/2.hdf5
    """

    def __init__(self,
                 files):
        self.files = copy.deepcopy(list(files))
        self.index = 0

    def next(self, blocking=True):
        result = self.files[self.index]
        self.index = (self.index + 1) % len(self.files)
        return DummyFile(result)

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


class AbstractFileIteratorWithCache(ABC):
    def __init__(self,
                 remote_files,
                 file_loader,
                 num_files_to_cache):
        """Initializes the iterator

        Args:
            remote_files:
                A list of the files to load
            file_loader:
                A callable that can copy the input file to a cache
            num_files_to_cache:
                A maximum number of files in a cache
        """
        def thread_function(request_queue, response_queue, file_loader):
            while True:
                remote = request_queue.get()
                if remote is None:
                    break
                response_queue.put(file_loader(remote))

        self.remote_files = copy.deepcopy(list(remote_files))
        self.request_queue = queue.Queue()
        self.response_queue = queue.Queue()

        # files in the current cache
        self.cached_files = []
        # index of the next file in cache to return
        self.idx = 0
        # number of files not received from the read_thread
        self.num_waited = 0
        self.cached_end = 0

        self._init_cache(num_files_to_cache)
        self.read_thread = threading.Thread(target=thread_function,
                                            args=(self.request_queue,
                                                  self.response_queue,
                                                  file_loader),
                                            daemon=True)
        self.read_thread.start()

    def _init_cache(self, num_files_to_cache):
        num_files_to_cache = min(num_files_to_cache, len(self.remote_files))
        for i in range(num_files_to_cache):
            self._add_download_request()
        self.num_files_to_cache = num_files_to_cache

    def _add_download_request(self):
        self.request_queue.put(self.remote_files[self.cached_end])
        self.cached_end = (self.cached_end + 1) % len(self.remote_files)
        self.num_waited += 1

    def _remove_from_cache(self):
        assert len(self.cached_files) > 0
        self.cached_files[0].remove()
        self.cached_files = self.cached_files[1:]
        self.idx = max(1, self.idx) - 1

    def _get_loaded_file(self, block):
        # if block is False the next statement cat throw queue.Empty exception
        result = ReleasableFile(self.response_queue.get(block))
        self.num_waited -= 1
        self._add_download_request()
        return result

    @abstractmethod
    def next(self, block):
        pass

    def reset(self):
        # remove all cached files but not read files
        for i in range(self.num_waited):
            result = ReleasableFile(self.response_queue.get(True))
            result.release()
            result.remove()
        self.num_waited = 0
        self.cached_end = 0
        self.idx = 0
        self._init_cache(self.num_files_to_cache)


class FileIteratorWithCache(AbstractFileIteratorWithCache):
    """Iterates over the remotes files using the cache.

    Assume:
    - Processing of a single file takes 1 period
    - Loading of a single file takes 2 periods
    - Cache size is 3 files
    - Load cache is 1 file

    The expected behaviour is
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
                 num_non_cached_files=2):
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
        """
        super().__init__(remote_files, file_loader, num_files_to_cache)

    def next(self, block=True):
        """ Returns path of the next cached element.

        Args:
            block:
                If the call is not blocking, the function may return None
        """
        print(f'initial = {self.idx}')
        while len(self.cached_files) > 0 and \
                not self.cached_files[0].is_in_use():
            self._remove_from_cache()
        print(f'first = {self.idx}')
        if self.idx == self.num_files_to_cache:
            raise CacheIsFullError("List of the cached files is full. "
                                   "Please release the oldest file "
                                   f"'{self.cached_files[0].name}'")
        # try to update the cache
        while len(self.cached_files) < self.num_files_to_cache:
            try:
                is_blocking = block and len(self.cached_files) <= self.idx
                result = self._get_loaded_file(is_blocking)
                print(f'Load a file {result.name}')
                self.cached_files.append(result)
            except queue.Empty:
                print('File is not loaded')
                break
        print(f'second = {self.idx}')
        print(self.cached_files)
        # if there is no file to return
        if len(self.cached_files) <= self.idx:
            return None
        self.idx += 1
        return self.cached_files[self.idx - 1]


class FileIteratorNonBlocking(AbstractFileIteratorWithCache):
    """Iterates over the remote files using the cache.
    It can return the same file several times even if some files have not been
    processed yet.

    Assume:
    - Processing of a single file takes 1 period
    - Loading of a single file takes 2 periods
    - Cache size is 3 files
    - Load cache is 1 file

    The expected behaviour is
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
    """
    def __init__(self,
                 remote_files,
                 file_loader,
                 num_files_to_cache=5,
                 num_non_cached_files=2):
        super().__init__(remote_files, file_loader, num_files_to_cache)

    def next(self, block=True):
        """ Returns path of the next cached element.

        Args:
            block:
                If the call is not blocking, the function may return None.
                It can happen only on the first iterations when there are no
                data in the cache
        """
        print(f'initial = {self.idx}')
        print(f'first = {self.idx}')
        # try to update the cache
        while len(self.cached_files) < self.num_files_to_cache or \
                not self.cached_files[0].is_in_use():
            try:
                result = self._get_loaded_file(False)
                print(f'Load a file {result.name}')
                if len(self.cached_files) == self.num_files_to_cache and \
                        not self.cached_files[0].is_in_use():
                    self._remove_from_cache()
                self.cached_files.append(result)
                # if we have planed to start a new cycle
                if self.idx == 0 and not self.cached_files[0].is_in_use():
                    self.idx = len(self.cached_files) - 1
            except queue.Empty:
                print('File is not loaded')
                break
        print(f'second = {self.idx}')
        print(self.cached_files)
        # if there is no file to return
        if len(self.cached_files) == 0:
            if block:
                result = self._get_loaded_file(True)
                self.cached_files.append(result)
                self.idx += 1
                print([x.name for x in self.cached_files])
                return self.cached_files[-1]
            else:
                return None
        print(f'third = {self.idx}')
        # we cannot store processed files
        # if it is not allowed to process them several times
        result = self.cached_files[self.idx]
        result.start_use()
        self.idx += 1
        self.idx %= len(self.cached_files)
        return result
