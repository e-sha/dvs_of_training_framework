from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread

from utils.file_iterators import FileLoader, FileIteratorWithCache


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
            loaded = self.iterator.next(block=False)
            self.last_loaded = loaded
            if loaded is None:
                out_q.put(token)
                continue
            self.iterator.step()
            out_q.put(token)

    def get_last_content(self):
        if self.last_loaded:
            text = self.last_loaded.name.read_text()
            # the document was checked thus change it to None
            self.last_loaded = None
            return text
        return 'None'


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
        for expected in expected_results:
            actual = processor.get_last_content()
            assert actual == expected
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
        for expected in expected_results:
            processing_thread.join(0.01)
            actual = processor.get_last_content()
            assert actual == expected
            in_q.put('token')
            out_q.get()
