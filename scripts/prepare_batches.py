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
    from train_flownet import get_dataloader, parse_args
    from utils.dataset import encode_batch
except ImportError:
    raise


def main(args):
    dataset_params = get_trainset_params(args)
    dataset_params.return_aug = True
    loader = get_dataloader(dataset_params)
    print(loader.dataset._dataset.return_aug)
    dataset_name = f'{loader.dataset._dataset.path.name}_preprocessed'
    out_path = args.data_path/dataset_name
    out_path.mkdir(exist_ok=True)
    num_steps = args.accum_step * args.training_steps
    j = 0
    for i, (events, timestamps, sample_idx, images, augmentation_params) \
            in tqdm(enumerate(loader), total=num_steps):
        events, timestamps, images, augmentation_params = encode_batch(
                events, timestamps, sample_idx, images, augmentation_params)
        with h5py.File(out_path/f'{j}.hdf5', 'w') as f:
            event_group = f.create_group('events')
            event_group.create_dataset('x', data=events['x'])
            event_group.create_dataset('y', data=events['y'])
            event_group.create_dataset('timestamp', data=events['timestamp'])
            event_group.create_dataset('polarity', data=events['polarity'])
            event_group.create_dataset('events_per_element',
                                       data=events['events_per_element'])
            event_group.create_dataset('elements_per_sample',
                                       data=events['elements_per_sample'])
            f.create_dataset('timestamps', data=timestamps)
            f.create_dataset('images', data=images)
            augmentation_group = f.create_group('augmentation')
            for k, v in augmentation_params.items():
                augmentation_group.create_dataset(k, data=v)
        j += 1
        if i + 1 == num_steps:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
