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
    # choose appropriate cache size and
    # number of downloaded-but-not-cached-yet files.
    if num_files_in_cache < len(files):
        cache_size = max(num_files_in_cache - 1, 1)
        files_not_in_cache = 1
    else:
        # it is possible to store all the files in the cache
        cache_size = num_files_in_cache
        files_not_in_cache = 2

    iterator = iterator_class(files,
                              FileLoader(cache_dir),
                              cache_size,
                              files_not_in_cache)
    if num_files_in_cache < len(files):
        return iterator
    # if we can cache all files then cache them and use the basic FileIterator
    new_files = [iterator.next().name for _ in files]
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
                 num_files_to_cache,
                 num_non_cached_files):
        """Initializes the iterator

        Args:
            remote_files:
                A list of the files to load
            file_loader:
                A callable that can copy the input file to a cache
            num_files_to_cache:
                A maximum number of files in a cache
            num_non_cached_files:
                A number of files that can be downloaded but not yet added to
                the cache. The optimal value is greater than 1.
        """
        def thread_function(request_queue,
                            token_queue,
                            response_queue,
                            file_loader):
            while True:
                remote = request_queue.get()
                if remote is None:
                    break
                # block until size of downloaded but not cached files become
                # less than a limit.
                token_queue.put(None)
                response_queue.put(file_loader(remote))

        self.remote_files = copy.deepcopy(list(remote_files))
        self.request_queue = queue.Queue()
        # the token queue is controls a number of downloaded but not cached
        # files in the memory. If the thread can put a token to the queue then
        # there is enough space to download another file.
        self.token_queue = queue.Queue(num_non_cached_files)
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
                                                  self.token_queue,
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
        file = self.cached_files.pop(0)
        file.remove()
        self.idx = max(1, self.idx) - 1

    def _get_loaded_file(self, block):
        # if block is False the next statement can throw queue.Empty exception
        result = ReleasableFile(self.response_queue.get(block))
        # here the current thread read the file, so it is safe to read
        # the token. The call is never blocked
        self.token_queue.get(True)
        self.num_waited -= 1
        self._add_download_request()
        return result

    @abstractmethod
    def next(self, block):
        pass

    def reset(self):
        # remove all cached files but not read files
        while self.cached_files:
            file = self.cached_files.pop()
            file.release()
            file.remove()
        for i in range(self.num_waited):
            result = ReleasableFile(self.response_queue.get(True))
            self.token_queue.get(True)
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
        super().__init__(remote_files,
                         file_loader,
                         num_files_to_cache,
                         num_non_cached_files)

    def next(self, block=True):
        """ Returns path of the next cached element.

        Args:
            block:
                If the call is not blocking, the function may return None
        """
        # remove files that were previously used
        while len(self.cached_files) > 0 and \
                not self.cached_files[0].is_in_use():
            self._remove_from_cache()
        if self.idx == self.num_files_to_cache:
            raise CacheIsFullError("List of the cached files is full. "
                                   "Please release the oldest file "
                                   f"'{self.cached_files[0].name}'")
        # try to update the cache
        while len(self.cached_files) < self.num_files_to_cache:
            try:
                # block only if next file was not loaded
                is_blocking = block and len(self.cached_files) <= self.idx
                result = self._get_loaded_file(is_blocking)
                self.cached_files.append(result)
            except queue.Empty:
                break
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
        super().__init__(remote_files,
                         file_loader,
                         num_files_to_cache,
                         num_non_cached_files)

    def next(self, block=True):
        """ Returns path of the next cached element.

        Args:
            block:
                If the call is not blocking, the function may return None.
                It can happen only on the first iterations when there are no
                data in the cache
        """
        # try to update the cache
        while len(self.cached_files) < self.num_files_to_cache or \
                not self.cached_files[0].is_in_use():
            try:
                block = block and len(self.cached_files) == 0
                result = self._get_loaded_file(block)
                if len(self.cached_files) == self.num_files_to_cache and \
                        not self.cached_files[0].is_in_use():
                    self._remove_from_cache()
                self.cached_files.append(result)
            except queue.Empty:
                break
        assert not block or len(self.cached_files) > 0
        # if there is no file to return
        if len(self.cached_files) == 0:
            return None
        self.idx = self.idx % len(self.cached_files)
        result = self.cached_files[self.idx]
        result.start_use()
        self.idx += 1
        return result
