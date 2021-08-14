import re
import torch


def get_torch_version():
    version = re.match(r'^([0-9.]*)', torch.__version__).groups()[0]
    return tuple(map(int, version.split('.')))


def cumsum_with_prefix(tensor: torch.Tensor,
                       dtype=None):
    """Returns cummulative sum of 1d tensor shifted by 1 element.
    [1, 2, 3] -> [0, 1, 3, 6].

    Args:
        tensor:
            An input tensor
        dtype:
            A data type of the resulting tensor

    Returns:
        shifted cummulative sum.
    """
    if dtype is None:
        dtype = tensor.dtype
    cs = torch.cumsum(tensor, dim=0)
    n = cs.numel() + 1
    result = torch.empty(n,
                         dtype=dtype,
                         device=cs.device,
                         requires_grad=cs.requires_grad)
    result[0] = 0
    result[1:] = cs
    return result
