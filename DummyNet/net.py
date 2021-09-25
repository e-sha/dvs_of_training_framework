import torch
from torch import nn


def get_local_idx(shard_idx: torch.Tensor):
    """Returns list of local indices and shard sizes given a list of shard
    indices

    Args:
        shard_idx:
            A vector of shard indices. All indices should be nonnegative
            long integers. For example, [0, 0, 1, 2]

    Returns:
        A vector of local indices.
        A vector of shard sizes.

        For example,
        shard_idx   [0, 0, 1, 1, 2, 1, 2, 2, 2]
        local_idx   [0, 1, 0, 1, 0, 2, 1, 2, 3]
        shard_sizes [2, 3, 4]


    Raises:
    """
    assert shard_idx.dtype == torch.long
    device = shard_idx.device
    bs = shard_idx.max() + 1
    num_elements = shard_idx.numel()
    sparse_mask = torch.zeros(bs * num_elements, dtype=torch.long,
                              device=device)
    indices = shard_idx * num_elements + torch.arange(num_elements,
                                                      dtype=torch.long,
                                                      device=device)
    sparse_mask[indices] = 1
    sparse_mask = sparse_mask.view(bs, num_elements)
    sparse_local_indices = torch.cumsum(sparse_mask, dim=1)
    local_idx = sparse_local_indices.view(-1)[indices] - 1
    return local_idx, sparse_local_indices.max(dim=1).values


class Model(nn.Module):
    def __init__(self,
                 device,
                 prefix_length=0,
                 suffix_length=0):
        super(Model, self).__init__()
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length

    def forward(self,
                events,
                timestamps,
                sample_idx,
                imsize,
                raw=True,
                intermediate=False):

        # compute extended image size
        outsize = [tuple(map(lambda x: x//2**i, imsize))
                   for i in range(4)][::-1]

        # shrink image to original size
        batch_size = sample_idx[-1] + 1
        result = tuple(torch.zeros([batch_size, 2, h, w], dtype=torch.float32)
                       for h, w in outsize)
        add_info = (tuple(), ) if intermediate else tuple()

        # choose timestamp and sample index for each result
        with torch.no_grad():
            element_idx, num_timestamps = get_local_idx(sample_idx)
            assert (num_timestamps ==
                    (2 + self.prefix_length + self.suffix_length)).all()
            mask = element_idx == self.prefix_length
            result_sample_idx = sample_idx[mask]
            mask = torch.logical_or(mask,
                                    element_idx == self.prefix_length + 1)
            result_timestamps = timestamps[mask].view(-1, 2)

        return (result, result_timestamps, result_sample_idx) + add_info
