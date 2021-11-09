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
        self.decoded_batch = {
            'data': torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03, 0,
                                        0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'sample_idx': torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
                                       dtype=torch.long),
            'images': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   dtype=torch.float32).view(-1, 1, 1, 1)
                                                       .tile(1, 1, 3, 4),
            'augmentation_params': {
                'idx': torch.tensor([0, 1, 2], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1, 4], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2, 3], dtype=torch.short),
                'box': torch.tensor([[0, 0, 3, 4],
                                     [0, 1, 3, 4],
                                     [1, 0, 3, 4]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True, False, True])},
            'size': 3}
        self.encoded_batch = {
            'data': torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'channels_per_sample': torch.tensor([2, 2, 2]),
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03, 0,
                                        0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'images': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   dtype=torch.uint8).view(-1, 1, 1, 1)
                                                     .tile(1, 1, 10, 10),
            'augmentation_params': {
                'idx': torch.tensor([0, 1, 2], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1, 4], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2, 3], dtype=torch.short),
                'box': torch.tensor([[0, 0, 3, 4],
                                     [0, 1, 3, 4],
                                     [1, 0, 3, 4]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True, False, True])}}
        self.decoded_batches = [{
            'data': torch.tensor([0, 1, 2, 3], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03],
                                       dtype=torch.float32),
            'sample_idx': torch.tensor([0, 0, 0, 1, 1],
                                       dtype=torch.long),
            'images': torch.tensor([0, 1, 2, 3, 4],
                                   dtype=torch.float32).view(-1, 1, 1, 1)
                                                       .tile(1, 1, 3, 4),
            'augmentation_params': {
                'idx': torch.tensor([0, 1], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2], dtype=torch.short),
                'box': torch.tensor([[0, 0, 3, 4],
                                     [0, 1, 3, 4]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2], dtype=torch.float32),
                'is_flip': torch.tensor([True, False])},
            'size': 2}, {
            'data': torch.tensor([4, 5], dtype=torch.float32)
                         .view(1, 2, 1, 1).tile(1, 1, 3, 4),
            'timestamps': torch.tensor([0, 0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'sample_idx': torch.tensor([0, 0, 0, 0, 0],
                                       dtype=torch.long),
            'images': torch.tensor([5, 6, 7, 8],
                                   dtype=torch.float32).view(-1, 1, 1, 1)
                                                       .tile(1, 1, 3, 4),
            'augmentation_params': {
                'idx': torch.tensor([2], dtype=torch.long),
                'sequence_length': torch.tensor([4], dtype=torch.short),
                'collapse_length': torch.tensor([3], dtype=torch.short),
                'box': torch.tensor([[1, 0, 3, 4]], dtype=torch.long),
                'angle': torch.tensor([0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True])},
            'size': 1}]
        self.encoded_batches = [{
            'data': torch.tensor([0, 1, 2, 3], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'channels_per_sample': torch.tensor([2, 2]),
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03],
                                       dtype=torch.float32),
            'images': torch.tensor([0, 1, 2, 3, 4], dtype=torch.uint8)
                           .view(-1, 1, 1, 1).tile(1, 1, 3, 4),
            'augmentation_params': {
                'idx': torch.tensor([0, 1], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2], dtype=torch.short),
                'box': torch.tensor([[0, 0, 3, 4],
                                     [0, 1, 3, 4]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2], dtype=torch.float32),
                'is_flip': torch.tensor([True, False])}}, {
            'data': torch.tensor([4, 5], dtype=torch.float32)
                         .view(-1, 2, 1, 1).tile(1, 1, 3, 4),
            'channels_per_sample': torch.tensor([2]),
            'timestamps': torch.tensor([0, 0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'images': torch.tensor([5, 6, 7, 8], dtype=torch.uint8)
                           .view(-1, 1, 1, 1).tile(1, 1, 3, 4),
            'augmentation_params': {
                'idx': torch.tensor([2], dtype=torch.long),
                'sequence_length': torch.tensor([4], dtype=torch.short),
                'collapse_length': torch.tensor([3], dtype=torch.short),
                'box': torch.tensor([[1, 0, 3, 4]], dtype=torch.long),
                'angle': torch.tensor([0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True])}}]

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
