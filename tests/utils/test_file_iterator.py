from pathlib import Path
from tempfile import TemporaryDirectory

from utils.file_iterators import FileLoader


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
