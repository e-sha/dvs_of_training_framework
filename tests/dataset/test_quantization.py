import h5py
from pathlib import Path
import tempfile
import torch

from tests.utils import compare
from utils.dataset import encode_quantized_batch, decode_quantized_batch
from utils.dataset import join_encoded_quantized_batches
from utils.dataset import read_encoded_quantized_batch
from utils.dataset import write_encoded_quantized_batch


class TestQuantized:
    def setup_class(self):
        self.decoded_batch = torch.tensor([0, 1, 2, 3, 4, 5],
                                          dtype=torch.float32) \
            .view(-1, 2, 1, 1).tile(1, 1, 3, 4)
        self.encoded_batch = {
            'data': torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'channels_per_sample': torch.tensor([2, 2, 2])}
        self.decoded_batches = [
            torch.tensor([0, 1, 2, 3], dtype=torch.float32)
                 .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            torch.tensor([4, 5], dtype=torch.float32)
                 .view(1, 2, 1, 1).tile(1, 1, 3, 4)]
        self.encoded_batches = [{
            'data': torch.tensor([0, 1, 2, 3], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'channels_per_sample': torch.tensor([2, 2])}, {
            'data': torch.tensor([4, 5], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'channels_per_sample': torch.tensor([2])}]

    def test_encode(self):
        encoded = encode_quantized_batch(self.decoded_batch)
        compare(encoded, self.encoded_batch)

    def test_decode(self):
        decoded = decode_quantized_batch(self.encoded_batch)
        compare(decoded, self.decoded_batch)

    def test_join(self):
        joined = join_encoded_quantized_batches(self.encoded_batches)
        compare(joined, self.encoded_batch)

    def test_read_write(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5') as f:
            filename = Path(f.name)
        write_encoded_quantized_batch(filename, self.encoded_batch)
        assert filename.is_file()
        with h5py.File(filename, 'r') as f:
            channels_per_sample = torch.tensor(f['channels_per_sample'])
            read = read_encoded_quantized_batch(f, channels_per_sample, 0, 3)
        compare(read, self.encoded_batch)
        filename.unlink()
