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
    from train_flownet import get_dataloader, parse_args
    from utils.dataset import encode_batch, write_encoded_batch, join_batches
except ImportError:
    raise


def main(args):
    loader = get_dataloader(get_trainset_params(args))
    dataset_name = f'{loader.dataset._dataset.path.name}_preprocessed'
    out_path = args.data_path/dataset_name
    out_path.mkdir(exist_ok=True)
    num_steps = args.accum_step * args.training_steps
    encoded_batches = []
    num_batches_per_write = 500
    j = 0
    for i, (events, timestamps, sample_idx, images, augmentation_params) \
            in tqdm(enumerate(loader), total=num_steps):
        encoded_batches.append(encode_batch(events, timestamps, sample_idx,
                                            images, augmentation_params))
        if (i + 1) % num_batches_per_write == 0:
            joined_batches = join_batches(encoded_batches)
            write_encoded_batch(out_path/f'{j}.hdf5', joined_batches)
            j += 1
            encoded_batches = []
        if i + 1 == num_steps:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
