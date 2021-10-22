from argparse import ArgumentParser
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
    from utils.common import write_params
    from utils.dataset import encode_batch, write_encoded_batch, join_batches
    from utils.options import validate_dataset_args, add_dataset_arguments
    from utils.options import add_dataset_preprocessing_arguments
except ImportError:
    raise


def parse_args(args, is_write=True):
    parser = ArgumentParser()
    args = add_dataset_arguments(parser)
    args = add_dataset_preprocessing_arguments(parser)
    args = parse_args(args)
    args = validate_dataset_args(args)
    if is_write:
        write_params(args.model, args)
    return args


def main(args):
    loader = get_dataloader(get_trainset_params(args))
    dataset_name = f'{loader.dataset._dataset.path.name}_preprocessed'
    (args.model/'parameters').unlink()
    out_path = args.model/dataset_name
    out_path.mkdir(exist_ok=True)
    written_indices = [int(f.stem) for f in out_path.glob('*.hdf5')]
    num_steps = args.accum_step * args.training_steps
    encoded_batches = []
    num_batches_per_write = 64
    j = 0
    for i, (events, timestamps, sample_idx, images, augmentation_params) \
            in tqdm(enumerate(loader), total=num_steps):
        encoded_batches.append(encode_batch(events, timestamps, sample_idx,
                                            images, augmentation_params))
        if (i + 1) % num_batches_per_write == 0:
            joined_batches = join_batches(encoded_batches)
            while j in written_indices:
                j += 1
            write_encoded_batch(out_path/f'{j}.hdf5', joined_batches)
            j += 1
            encoded_batches = []
        if i + 1 == num_steps:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
