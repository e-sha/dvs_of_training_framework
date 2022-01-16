from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread

from utils.file_iterators import FileLoader, FileIteratorWithCache
from utils.file_iterators import create_file_iterator, FileIterator


def test_FileLoader():
    tmp_dir_holder = TemporaryDirectory(dir='/tmp')
    cache_dir_holder = TemporaryDirectory(dir='/tmp')
    tmp_dir = Path(tmp_dir_holder.name)
    cache_dir = Path(cache_dir_holder.name)
    filename = tmp_dir/'123'
    text = "456"
    filename.write_text(text)
    loader = FileLoader(cache_dir)
    out_file = loader(filename)
    assert out_file.parent == cache_dir
    assert out_file.read_text() == text


class FileLoaderWithDelay:
    def __init__(self, N, in_q, out_q):
        self.N = N
        self.cache_dir_holder = TemporaryDirectory(dir='/tmp')
        self.file_loader = FileLoader(Path(self.cache_dir_holder.name))
        self.in_q = in_q
        self.out_q = out_q

    def __call__(self, filename):
        for _ in range(self.N - 1):
            token = self.in_q.get()
            self.out_q.put(token)
        token = self.in_q.get()
        result = self.file_loader(filename)
        self.out_q.put(token)
        return result


class Processing:
    def __init__(self):
        self.last_loaded = None
        self.iterator = None

    def __call__(self,
                 files2process,
                 file_loader,
                 files2cache,
                 process_only_once,
                 in_q,
                 out_q):
        self.iterator = FileIteratorWithCache(
                files2process, file_loader, files2cache,
                process_only_once=process_only_once)
        while True:
            token = in_q.get()
            if self.last_loaded:
                self.last_loaded.release()
                self.last_loaded = None
            self.last_loaded = self.iterator.next(block=False)
            out_q.put(token)

    def get_last_content(self):
        if self.last_loaded:
            return self.last_loaded.name.read_text()
        return 'None'

    def get_cached_files(self):
        if self.iterator is None:
            return []
        return [f.name.read_text() for f in self.iterator.cached_files]


class TestFileIterator:
    def setup_class(self):
        self.files2cache = 3
        self.time2load = 2
        self.files2process = []
        self.tmp_dir_holder = TemporaryDirectory(dir='/tmp')
        for i in range(10):
            self.files2process.append(Path(self.tmp_dir_holder.name)/f'F{i}')
            self.files2process[-1].write_text(f'F{i}')

    def test_process_only_once(self):
        in_q, int_q, out_q = Queue(), Queue(), Queue()
        file_loader = FileLoaderWithDelay(self.time2load, in_q, int_q)
        processor = Processing()
        processing_thread = Thread(target=processor,
                                   args=(self.files2process,
                                         file_loader,
                                         self.files2cache,
                                         True,
                                         int_q,
                                         out_q),
                                   daemon=True)
        processing_thread.start()

        expected_results = [y
                            for x in ['None', 'F0', 'F1', 'F2', 'F3']
                            for y in [x, 'None']]
        expected_cached = [[], [],
                           ['F0'], [],
                           ['F1'], [],
                           ['F2'], [],
                           ['F3'], []]
        for expected in zip(expected_results, expected_cached):
            actual = processor.get_last_content()
            cached = processor.get_cached_files()
            assert actual == expected[0]
            assert cached == expected[1]
            in_q.put('token')
            out_q.get()

    def test_allow_multiple_passes(self):
        in_q, int_q, out_q = Queue(), Queue(), Queue()
        file_loader = FileLoaderWithDelay(self.time2load, in_q, int_q)
        processor = Processing()
        processing_thread = Thread(target=processor,
                                   args=(self.files2process,
                                         file_loader,
                                         self.files2cache,
                                         False,
                                         int_q,
                                         out_q),
                                   daemon=True)
        processing_thread.start()

        expected_results = ['None', 'None', 'F0', 'F0', 'F1',
                            'F0', 'F1', 'F2', 'F3', 'F1']
        expected_cached = [[], [],
                           ['F0'], ['F0'],
                           ['F0', 'F1'], ['F0', 'F1'],
                           ['F0', 'F1', 'F2'], ['F0', 'F1', 'F2'],
                           ['F1', 'F2', 'F3'], ['F1', 'F2', 'F3']]
        for expected in zip(expected_results, expected_cached):
            processing_thread.join(0.01)
            actual = processor.get_last_content()
            cached = processor.get_cached_files()
            assert actual == expected[0]
            assert cached == expected[1]
            in_q.put('token')
            out_q.get()

    def test_short_dataset_with_cache(self):
        self.cache_dir_holder = TemporaryDirectory(dir='/tmp')
        iterator = create_file_iterator(
            self.files2process, cache_dir=Path(self.cache_dir_holder.name),
            num_files_in_cache=len(self.files2process),
            process_only_once=False)
        assert isinstance(iterator, FileIterator)
        assert len(list(self.cache_dir_holder.glob("*.hdf5"))) == \
            len(self.files2process)
        for gt_file in self.files2process * 2:
            f = next(iterator)
            assert gt_file.read_text() == f.read_text()
