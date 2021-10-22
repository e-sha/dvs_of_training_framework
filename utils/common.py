import copy
import numpy as np
import os
from pathlib import Path
import re
import subprocess
import sys
import torch
from typing import Union, Dict
import yaml


def is_inside_docker():
    return 'INSIDE_DOCKER' in os.environ and bool(os.environ['INSIDE_DOCKER'])


def get_torch_version():
    version = re.match(r'^([0-9.]*)', torch.__version__).groups()[0]
    return tuple(map(int, version.split('.')))


def mean(values):
    return sum(values) / len(values)


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


def get_commithash(cwd=None):
    """Returns a commit hash for the specified directory

    Args:
        cwd:
            Directory of the repository.

    Returns:
        A git commit hash as a string
    """
    return subprocess \
        .check_output('git rev-parse --verify HEAD',
                      shell=True, cwd=cwd) \
        .decode() \
        .strip()


def encode_args(args):
    """Converts arguments to a format that can be written to a text file.

    Converts
    * pathlib.Path -> string
    * tuple -> list
    * torch.device -> string

    Args:
        args:
            A SimpleNamespace that contains (key, value) pair representation
            of the arguments.

    Returns:
        A string representation of the input arguments.
    """
    result = copy.deepcopy(vars(args))
    for k, v in result.items():
        if isinstance(v, Path):
            result[k] = str(v)
        elif isinstance(v, tuple):
            result[k] = list(v)
        elif isinstance(v, torch.device):
            result[k] = str(v)
    return yaml.dump(result)


def write_params(out_dir, args):
    """Writes arguments to a file

    Serializes arguments to a out_dir/parameters file.

    Args:
        out_dir:
            Path to write arguments.
        args:
            A SimpleNamespace that contains (key, value) pair representation
            of the arguments.
    """
    strings = [' '.join(sys.argv),
               f'commit hash: {get_commithash()}']
    if 'flownet_path' in vars(args):
        strings.append(
                f'model commit hash: {get_commithash(args.flownet_path)}')
    strings.append(encode_args(args))
    data2write = '\n'.join(strings)
    (out_dir/'parameters').write_text(data2write)


def to_tensor(data: Union[np.ndarray, Dict]):
    """Comverts data to torch.Tensor

    If data is dict, the function convert each item of it to torch.Tensor.
    Otherwise, apply torch.Tensor to the data.

    Args:
        data:
            A tensor data or dictionary of str->tensor data

    Returns:
        A converted data
    """
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = to_tensor(v)
        return data
    if isinstance(data, np.ndarray) and data.dtype == np.int_:
        return torch.tensor(data, dtype=torch.long)
    return torch.tensor(data, dtype=torch.float32)
