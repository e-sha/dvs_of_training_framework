from pathlib import Path
import torch
from types import SimpleNamespace

from utils.common import is_inside_docker
from utils.dataset import Dataset, IterableDataset, collate_wrapper
from utils.dataset import PreprocessedDataloader


script_dir = Path(__file__).resolve().parent.parent


def choose_data_path(args):
    args.model.mkdir(exist_ok=True, parents=True)
    if is_inside_docker():
        data_path = Path('/data/training/mvsec')
    else:
        base_dir = (script_dir/'..').resolve()
        data_path = base_dir/'data'/'training'/'mvsec'
    args.data_path = data_path
    args.log_path = args.model/'log'
    return args


def get_common_dataset_params(args):
    is_raw = args.is_raw
    collate_fn = collate_wrapper if is_raw else None
    return SimpleNamespace(is_raw=is_raw,
                           collate_fn=collate_fn,
                           shape=args.shape,
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
    params.preprocessed_dataset = args.preprocessed_dataset
    return params


def get_valset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day1'
    params.augmentation = False
    params.collapse_length = 1
    params.shuffle = False
    params.infinite = False
    params.preprocessed_dataset = False
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


def get_dataloader(params):
    if params.preprocessed_dataset:
        path = params.path.parent / (params.path.name + '_preprocessed')
        return PreprocessedDataloader(
            path=path,
            batch_size=params.batch_size)
    loader_kwargs = {}
    if not params.infinite:
        loader_kwargs['shuffle'] = params.shuffle
    return torch.utils.data.DataLoader(get_dataset(params),
                                       collate_fn=params.collate_fn,
                                       batch_size=params.batch_size,
                                       num_workers=params.num_workers,
                                       pin_memory=params.pin_memory,
                                       **loader_kwargs)