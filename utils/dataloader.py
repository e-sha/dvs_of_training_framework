from pathlib import Path
import torch
from types import SimpleNamespace

from utils.common import is_inside_docker
from utils.dataset import Dataset, IterableDataset, collate_wrapper
from utils.dataset import PreprocessedDataloader


script_dir = Path(__file__).resolve().parent.parent


def choose_data_path(args):
    """ Chooses path with the input data

    Args:
        args:
            A key-value storage of the arguments.

    Returns:
        An updated arguments with a data_path key is set
    """
    if is_inside_docker():
        data_path = Path('/data/training/mvsec')
    else:
        base_dir = (script_dir/'..').resolve()
        data_path = base_dir/'data'/'training'/'mvsec'
    args.data_path = data_path
    return args


def choose_collate_function(is_raw):
    return collate_wrapper if is_raw else None


def get_common_dataset_params(args):
    return SimpleNamespace(shape=args.shape,
                           batch_size=args.mbs,
                           pin_memory=True,
                           num_workers=args.num_workers,
                           min_seq_length=args.min_sequence_length,
                           max_seq_length=args.max_sequence_length,
                           is_static_seq_length=not args.dynamic_sample_length)


def get_trainset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day2'
    params.augmentation = True
    params.collapse_length = args.cl
    params.shuffle = True
    params.infinite = True
    params.is_raw = args.is_raw
    params.collate_fn = choose_collate_function(params.is_raw)
    params.preprocessed_dataset_path = args.preprocessed_dataset_path \
        if 'preprocessed_dataset_path' in args else None
    params.cache_dir = args.cache_dir if 'cache_dir' in args else None
    return params


def get_valset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day1'
    params.augmentation = False
    params.collapse_length = 1
    params.shuffle = False
    params.infinite = False
    params.is_raw = True
    params.collate_fn = choose_collate_function(params.is_raw)
    params.preprocessed_dataset_path = None
    return params


def get_dataset(params):
    kwargs = {'path': params.path,
              'shape': params.shape,
              'augmentation': params.augmentation,
              'collapse_length': params.collapse_length,
              'is_raw': params.is_raw,
              'min_seq_length': params.min_seq_length,
              'max_seq_length': params.max_seq_length,
              'is_static_seq_length': params.is_static_seq_length}
    if params.infinite:
        return IterableDataset(shuffle=params.shuffle, **kwargs)
    return Dataset(**kwargs)


def get_dataloader(params, sample_idx=0, process_only_once=True):
    if params.preprocessed_dataset_path is not None:
        loader = PreprocessedDataloader(
            path=params.preprocessed_dataset_path,
            batch_size=params.batch_size,
            is_raw=params.is_raw,
            cache_dir=params.cache_dir,
            process_only_once=process_only_once)
        loader.set_index(sample_idx)
        return loader
    loader_kwargs = {}
    if not params.infinite:
        loader_kwargs['shuffle'] = params.shuffle
    return torch.utils.data.DataLoader(get_dataset(params),
                                       collate_fn=params.collate_fn,
                                       batch_size=params.batch_size,
                                       num_workers=params.num_workers,
                                       pin_memory=params.pin_memory,
                                       **loader_kwargs)
