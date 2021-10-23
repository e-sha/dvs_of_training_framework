from argparse import ArgumentParser
import h5py
import sys
from tqdm import tqdm


try:
    from train_flownet import get_trainset_params
except ImportError:
    from pathlib import Path

    script_path = Path(__file__).resolve().parent
    sys.path.append(str(script_path.parent))
    from train_flownet import get_trainset_params


try:
    from train_flownet import get_dataloader
    from utils.common import write_execution_info, collect_execution_info
    from utils.common import check_execution_info
    from utils.dataset import encode_batch, write_encoded_batch, join_batches
    from utils.options import validate_dataset_args, add_dataset_arguments
    from utils.options import add_dataset_preprocessing_arguments
    from utils.options import add_dataloader_arguments, add_common_arguments
except ImportError:
    raise


def parse_args(args, is_write=True):
    parser = ArgumentParser()
    parser = add_common_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_dataloader_arguments(parser)
    parser = add_dataset_preprocessing_arguments(parser)
    args = parser.parse_args(args)
    args = validate_dataset_args(args)
    execution_info = collect_execution_info(args)
    check_execution_info(args.output, execution_info, args)
    args.output.mkdir(exist_ok=True, parents=True)
    if is_write:
        write_execution_info(args.output, execution_info)
    return args


def main(args):
    loader = get_dataloader(get_trainset_params(args))
    args.output.mkdir(exist_ok=True)
    written_files = list(args.output.glob('*.hdf5'))
    written_indices = [int(f.stem) for f in written_files]
    num_written = 0
    for filename in written_files:
        with h5py.File(filename, 'r') as f:
            num_written += len(f['events']['elements_per_sample'])
    num_batches_per_write = (args.samples_per_file - 1) // args.mbs + 1
    encoded_batches = []
    j = 0
    initial = num_written // args.mbs
    total = (args.size - num_written) // args.mbs + initial
    for i, (events, timestamps, sample_idx, images, augmentation_params) \
            in tqdm(enumerate(loader), initial=initial, total=total):
        if num_written >= args.size:
            break
        encoded_batches.append(encode_batch(events, timestamps, sample_idx,
                                            images, augmentation_params))
        num_written += \
            len(encoded_batches[-1]['events']['elements_per_sample'])
        is_last = num_written >= args.size
        if (i + 1) % num_batches_per_write == 0 or is_last:
            joined_batches = join_batches(encoded_batches)
            while j in written_indices:
                j += 1
            write_encoded_batch(args.output/f'{j}.hdf5', joined_batches)
            j += 1
            encoded_batches = []
        if is_last:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
