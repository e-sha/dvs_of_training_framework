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


def collect_execution_info(args):
    """Collects information about the environment and args in a text format

    Args:
        args:
            A SimpleNamespace that contains (key, value) pair representation
            of the arguments.

    Return:
        A textual representation of the execution information
    """
    strings = [' '.join(sys.argv),
               '--',
               f'commit hash: {get_commithash()}']
    if 'flownet_path' in vars(args):
        strings.append(
                f'model commit hash: {get_commithash(args.flownet_path)}')
    strings.append('--')
    strings.append(encode_args(args))
    return '\n'.join(strings)


def file_for_execution_info(out_dir):
    """Returns path to a file to store information about the environment

    Args:
        out_dir:
            A directory to store data

    Return:
        pathlib.Path object of the file
    """
    return out_dir/'parameters'


def write_execution_info(out_dir, execution_info):
    """Writes information about the environment and args to a text file

    Writes data to a out_dir/parameters file.

    Args:
        out_dir:
            A directory to write data
        execution_info:
            An execution information as a text
    """
    file_for_execution_info(out_dir).write_text(execution_info)


def read_execution_info(out_dir):
    """Reads information about the environment and args

    Args:
        out_dir:
            A directory to read data. File located in out_dir/parameters.

    Return:
        A textual representation of the execution information.
        None if file does not exits.
    """
    path = file_for_execution_info(out_dir)
    if path.is_file():
        return path.read_text()
    return None


def split_execution_info_into_groups(execution_info):
    """Splits execution information into groups according to '--' separator.

    Args:
        execution_info:
            A textual representation of the execution_info.

    Returns:
        A grouped representation of the execution_info
    """
    return re.split(r'^--$|^--\n|\n--$|\n--\n', execution_info)


def execution_info2code_revisions(execution_info):
    """Extracts information about the repository revision from the execution
    info.

    Args:
        execution_info:
            A textual representation of the execution_info.

    Returns:
        A dictory representation of revisions in the {name: revision} format.
    """
    revisions_group = split_execution_info_into_groups(execution_info)[1]
    return dict(map(lambda y: y.strip(), x.split(':'))
                for x in revisions_group.split('\n'))


def execution_info2args(execution_info):
    """Extracts arguments from the execution info.

    Args:
        execution_info:
            A textual representation of the execution_info.

    Returns:
        A dictory representation of the arguments
    """
    return yaml.safe_load(split_execution_info_into_groups(execution_info)[2])


def check_execution_info(out_dir, execution_info, args):
    """Checks that information about the execution is correct

    Args:
        execution_info:
            A textual representation of the execution_info.
        out_dir:
            A directory to read data. File located in out_dir/parameters.
        args:
            A SimpleNamespace that contains (key, value) pair representation
            of the arguments.
    """
    previous_execution_info = read_execution_info(out_dir)
    if previous_execution_info is not None:
        if not args.allow_obsolete_code:
            previous_revisions = \
                    execution_info2code_revisions(previous_execution_info)
            current_revisions = execution_info2code_revisions(execution_info)
            for k in set(previous_revisions) & set(current_revisions):
                assert previous_revisions[k] == current_revisions[k], \
                    f"Stored and current revisions for repository {k} are " \
                    f"different ({previous_revisions[k]} " \
                    f"vs {current_revisions[k]})"
        if not args.allow_arguments_change:
            previous_args = execution_info2args(previous_execution_info)
            current_args = execution_info2args(execution_info)
            keys = set(current_args) & set(previous_args)
            for k in set(keys) - {'allow_arguments_change',
                                  'allow_obsolete_code',
                                  'cache-dir'}:
                assert previous_args[k] == current_args[k], \
                    f'Stored and current value for argument {k} are ' \
                    f'different ({previous_args[k]} vs {current_args[k]})'


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
