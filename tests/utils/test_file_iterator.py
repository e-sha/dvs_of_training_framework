from pathlib import Path
from tempfile import TemporaryDirectory

from utils.file_iterators import FileLoader


def test_FileLoader():
    tmp_dir = Path(TemporaryDirectory(dir='/tmp'))
    cache_dir = Path(TemporaryDirectory(dir='/tmp'))
    filename = tmp_dir/'123'
    text = "456"
    filename.write_text(text)
    loader = FileLoader(cache_dir)
    out_file = loader(filename)
    assert out_file.parent == cache_dir
    assert out_file.read_text() == text
