import torch
from utils.dataset import encode_batch, decode_batch, join_batches


def compare(computed, groundtruth, prefix=''):
    if isinstance(computed, torch.Tensor):
        assert isinstance(groundtruth, torch.Tensor), prefix
        assert torch.equal(computed, groundtruth), prefix
        return
    assert isinstance(computed, dict) and isinstance(groundtruth, dict),\
        prefix
    assert len(computed) == len(groundtruth),\
        f'{prefix}: {computed.keys()} {groundtruth.keys()}'
    for k in computed.keys():
        assert k in groundtruth
        compare(computed[k], groundtruth[k], prefix=prefix+f'.{k}')


class TestDatasetEncoding:
    def setup_class(self):
        self.decoded = {
            'events': torch.tensor([[1, 2, 0.02, -1, 0, 0],
                                    [2, 1, 0.06, 1, 1, 0],
                                    [2, 3, 0.07, -1, 1, 0],
                                    [1, 4, 0.015, 1, 0, 1],
                                    [4, 1, 0.01, 1, 0, 2],
                                    [6, 6, 0.05, 1, 2, 2],
                                    [7, 8, 0.07, -1, 3, 2]],
                                   dtype=torch.float32),
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03, 0,
                                        0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'sample_idx': torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
                                       dtype=torch.long),
            'images': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   dtype=torch.float32).view(-1, 1, 1, 1)
                                                       .tile(1, 1, 10, 10),
            'augmentation_params': {
                'idx': torch.tensor([0, 1, 2], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1, 4], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2, 3], dtype=torch.short),
                'box': torch.tensor([[0, 0, 10, 10],
                                     [0, 1, 10, 10],
                                     [1, 0, 10, 10]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True, False, True])}}
        self.encoded = {
            'events': {'x': torch.tensor([1, 2, 2, 1, 4, 6, 7],
                                         dtype=torch.short),
                       'y': torch.tensor([2, 1, 3, 4, 1, 6, 8],
                                         dtype=torch.short),
                       'timestamp': torch.tensor([0.02, 0.06, 0.07, 0.015,
                                                  0.01, 0.05, 0.07],
                                                 dtype=torch.float32),
                       'polarity': torch.tensor([False, True, False,
                                                 True, True, True, False]),
                       'events_per_element': torch.tensor([1, 2, 1, 1,
                                                           0, 1, 1],
                                                          dtype=torch.long),
                       'elements_per_sample': torch.tensor([2, 1, 4],
                                                           dtype=torch.short)},
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
                'box': torch.tensor([[0, 0, 10, 10],
                                     [0, 1, 10, 10],
                                     [1, 0, 10, 10]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True, False, True])}}

        self.encoded_parts = [
            {'events': {'x': torch.tensor([1, 2, 2, 1],
                                          dtype=torch.short),
                        'y': torch.tensor([2, 1, 3, 4],
                                          dtype=torch.short),
                        'timestamp': torch.tensor([0.02, 0.06, 0.07, 0.015],
                                                  dtype=torch.float32),
                        'polarity': torch.tensor([False, True, False,
                                                  True]),
                        'events_per_element': torch.tensor([1, 2, 1],
                                                           dtype=torch.long),
                        'elements_per_sample':
                        torch.tensor([2, 1], dtype=torch.short)},
             'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03],
                                        dtype=torch.float32),
             'images': torch.tensor([0, 1, 2, 3, 4],
                                    dtype=torch.uint8).view(-1, 1, 1, 1)
                                                      .tile(1, 1, 10, 10),
             'augmentation_params': {
                 'idx': torch.tensor([0, 1], dtype=torch.long),
                 'sequence_length': torch.tensor([2, 1], dtype=torch.short),
                 'collapse_length': torch.tensor([1, 2], dtype=torch.short),
                 'box': torch.tensor([[0, 0, 10, 10],
                                      [0, 1, 10, 10]], dtype=torch.long),
                 'angle': torch.tensor([0.1, 0.2], dtype=torch.float32),
                 'is_flip': torch.tensor([True, False])}},
            {'events': {'x': torch.tensor([4, 6, 7],
                                          dtype=torch.short),
                        'y': torch.tensor([1, 6, 8],
                                          dtype=torch.short),
                        'timestamp': torch.tensor([0.01, 0.05, 0.07],
                                                  dtype=torch.float32),
                        'polarity': torch.tensor([True, True, False]),
                        'events_per_element': torch.tensor([1, 0, 1, 1],
                                                           dtype=torch.long),
                        'elements_per_sample':
                        torch.tensor([4], dtype=torch.short)},
             'timestamps': torch.tensor([0, 0.02, 0.04, 0.06, 0.08],
                                        dtype=torch.float32),
             'images': torch.tensor([5, 6, 7, 8],
                                    dtype=torch.uint8).view(-1, 1, 1, 1)
                                                      .tile(1, 1, 10, 10),
             'augmentation_params': {
                 'idx': torch.tensor([2], dtype=torch.long),
                 'sequence_length': torch.tensor([4], dtype=torch.short),
                 'collapse_length': torch.tensor([3], dtype=torch.short),
                 'box': torch.tensor([[1, 0, 10, 10]], dtype=torch.long),
                 'angle': torch.tensor([0.3], dtype=torch.float32),
                 'is_flip': torch.tensor([True])}}
            ]

    def test_encode(self):
        encoded = encode_batch(**self.decoded)
        encoded = {'events': encoded[0],
                   'timestamps': encoded[1],
                   'images': encoded[2],
                   'augmentation_params': encoded[3]}
        compare(encoded, self.encoded)

    def test_decoded(self):
        decoded = decode_batch(**self.encoded)
        decoded = {'events': decoded[0],
                   'timestamps': decoded[1],
                   'sample_idx': decoded[2],
                   'images': decoded[3],
                   'augmentation_params': decoded[4]}
        compare(decoded, self.decoded)

    def test_join(self):
        joined = join_batches(self.encoded_parts)
        compare(joined, self.encoded)
