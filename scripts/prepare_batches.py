import h5py
import sys
from tqdm import tqdm


try:
    from train_flownet import get_trainset_params
except ImportError:
    from pathlib import Path

    script_path = Path(__file__).resolve().parent
    sys.path.append(str(script_path.parent))


try:
    from train_flownet import get_dataloader, parse_args
except ImportError:
    raise


def main(args):
    loader = get_dataloader(get_trainset_params(args))
    dataset_name = f'{loader.dataset._dataset.path.name}_preprocessed'
    out_path = args.data_path/dataset_name
    out_path.mkdir(exist_ok=True)
    num_steps = args.accum_step * args.training_steps
    j = 0
    for i, (events, timestamps, sample_idx, images) in tqdm(enumerate(loader),
                                                            total=num_steps):
        with h5py.File(out_path/f'{j}.hdf5', 'w') as f:
            f.create_dataset('events', data=events)
            f.create_dataset('timestamps', data=timestamps)
            f.create_dataset('sample_idx', data=sample_idx)
            f.create_dataset('images', data=images)
        j += 1
        if i + 1 == num_steps:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
