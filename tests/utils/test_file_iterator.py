from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from threading import Barrier, Lock, Thread

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
    def __init__(self, barrier, N):
        self.barrier = barrier
        self.N = N
        self.cache_dir_holder = TemporaryDirectory(dir='/tmp')
        self.file_loader = FileLoader(Path(self.cache_dir_holder.name))

    def __call__(self, filename):
        for _ in range(self.N):
            self.barrier.wait()
        return self.file_loader(filename)


class Processing:
    def __init__(self, files2process, file_loader, files2cache):
        self.files2process = files2process
        self.file_loader = file_loader
        self.files2cache = files2cache
        self.last_loaded = None
        self.lock = Lock()

    def __call__(self):
        self.iterator = FileIteratorWithCache(self.files2process,
                                              self.file_loader,
                                              self.files2cache)
        while True:
            loaded = self.iterator.next()
            with self.lock:
                self.last_loaded = loaded
            self.iterator.step()

    def get_last_content(self):
        with self.lock:
            if self.last_loaded:
                return self.last_loaded.name.read_text()
        return 'None'


class TestFileIterator:
    def setup_class(self):
        pass

    def test_process_only_once(self):
        files2cache = 3
        time2load = 2
        barrier = Barrier(2)
        file_loader = FileLoaderWithDelay(barrier, time2load)
        tmp_dir_holder = TemporaryDirectory(dir='/tmp')
        files2process = []
        for i in range(10):
            files2process.append(Path(tmp_dir_holder.name)/f'F{i}')
            files2process[-1].write_text(f'F{i}')
        processor = Processing(files2process, file_loader, files2cache)
        processing_thread = Thread(target=processor, args=tuple(), daemon=True)
        processing_thread.start()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
        barrier.wait()
        sleep(0.1)
        print(processor.get_last_content())
