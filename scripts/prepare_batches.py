import h5py
import sys
import torch
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
except ImportError:
    raise


def get_size(tensor):
    return tensor.nelement() * tensor.element_size()


def print_info(tensor_before, tensor_after, name):
    if isinstance(tensor_before, torch.Tensor):
        tensor_before = [tensor_before]
    if isinstance(tensor_after, torch.Tensor):
        tensor_after = [tensor_after]
    size_before = sum(map(get_size, tensor_before))
    size_after = sum(map(get_size, tensor_after))
    print(f'{name}: before {size_before} after {size_after} -- '
          f'{size_after / size_before * 100}%')
    for before, after in zip(tensor_before, tensor_after):
        tmp = after.to(before.dtype)
        assert torch.equal(before, tmp)


def main(args):
    loader = get_dataloader(get_trainset_params(args))
    dataset_name = f'{loader.dataset._dataset.path.name}_preprocessed'
    out_path = args.data_path/dataset_name
    out_path.mkdir(exist_ok=True)
    num_steps = args.accum_step * args.training_steps
    j = 0
    for i, (events, timestamps, sample_idx, images) in tqdm(enumerate(loader),
                                                            total=num_steps):
        events_x = events[:, 0]
        events_y = events[:, 1]
        events_t = events[:, 2]
        events_p = (events[:, 3] + 1) / 2
        events_e = events[:, 4]
        events_s = events[:, 5]
        events_x_after = events_x.to(torch.int16)
        events_y_after = events_y.to(torch.int16)
        events_t_after = events_t
        events_p_after = events_p.to(torch.bool)
        events_e_after = events_e.to(torch.uint8)
        events_s_after = events_s.to(torch.uint8)
        timestamps_after = timestamps
        sample_idx_after = sample_idx.to(torch.uint8)
        images_after = images.to(torch.uint8)
        # print_info(events_x, events_x_after, 'x')
        # print_info(events_y, events_y_after, 'y')
        # print_info(events_t, events_t_after, 't')
        # print_info(events_p, events_p_after, 'p')
        # print_info(events_e, events_e_after, 'e')
        # print_info(events_s, events_s_after, 's')
        # print_info(timestamps, timestamps_after, 'timestamps')
        # print_info(sample_idx, sample_idx_after, 'sample_idx')
        # print_info(images, images_after, 'images')
        # print_info([events_x, events_y, events_t, events_p, events_e,
        #             events_s, timestamps, sample_idx, images],
        #            [events_x_after, events_y_after, events_t_after,
        #             events_p_after, events_e_after, events_s_after,
        #             timestamps_after, sample_idx_after, images_after], 'all')
        with h5py.File(out_path/f'{j}.hdf5', 'w') as f:
            event_group = f.create_group('events')
            event_group.create_dataset('x', data=events_x_after)
            event_group.create_dataset('y', data=events_y_after)
            event_group.create_dataset('t', data=events_t_after)
            event_group.create_dataset('p', data=events_p_after)
            event_group.create_dataset('e', data=events_e_after)
            event_group.create_dataset('s', data=events_s_after)
            f.create_dataset('timestamps', data=timestamps_after)
            f.create_dataset('sample_idx', data=sample_idx_after)
            f.create_dataset('images', data=images_after)
        j += 1
        if i + 1 == num_steps:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
