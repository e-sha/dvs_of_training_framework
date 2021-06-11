from pathlib import Path
import numpy as np
import h5py
import yaml
from tqdm import tqdm
import os
import sys


sys.path.append(os.getcwd())

try:
    from utils.dataset import read_info
except ImportError:
    exit(-1)


is_inside = 'INSIDE_DOCKER' in os.environ.keys() and \
            bool(os.environ['INSIDE_DOCKER'])


def write_samples(events, images, image_ts, img2event_map, out_dir, ts0):
    for i, (b, e, start_ts, stop_ts) in tqdm(enumerate(zip(img2event_map[:-1],
                                                           img2event_map[1:],
                                                           image_ts[:-1],
                                                           image_ts[1:])),
                                             total=img2event_map.size-1):

        # very strange fix, but it is required
        frame_events = np.array(events[b+1:e+1])
        assert frame_events[0, 2] >= start_ts, 'The first event is before ' \
                                               'the first image'
        assert b < 0 or events[b, 2] <= start_ts, 'Some events are missed'
        assert frame_events[-1, 2] <= stop_ts, \
               'The last event is after the second image'
        assert e+1 >= events.shape[0] or events[e+1, 2] >= stop_ts, \
               'Some events are missed'
        frame_events[:, 2] -= ts0
        image1 = np.array(images[i])
        image2 = np.array(images[i+1])
        with h5py.File(str(out_dir/f'{i:06d}.hdf5'), 'w') as of:
            of.create_dataset('image1', data=image1)
            of.create_dataset('image2', data=image2)
            of.create_dataset('events', data=frame_events)
            of.create_dataset('start', data=start_ts - ts0)
            of.create_dataset('stop', data=stop_ts - ts0)


script_dir = Path(__file__).resolve().parent.parent
if is_inside:
    data_dir = Path('/data')
    info_dir = data_dir/'info'
else:
    data_dir = (script_dir/'..'/'data').resolve()
    info_dir = script_dir/'data'/'info'

config_dir = script_dir/'config'
raw_data_dir = data_dir/'raw'
training_dir = data_dir/'training'

config_path = config_dir/'training_datasets.yml'
with open(config_path, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

for ds_name, ds_config in config.items():
    # dir with dataset data
    ds_raw_dir = raw_data_dir/ds_name
    # dir to write samples
    ds_training_dir = training_dir/ds_name
    # dir with dataset info: first timestamp for each sequence
    info_file = info_dir/(ds_name + '.hdf5')
    ds_info = read_info(str(info_file))
    for seq_name, seq_config in ds_config.items():
        # file with sequence data
        seq_raw_file = ds_raw_dir/seq_name[:-1]/(seq_name+'_data.hdf5')
        # dir to write sequence samples
        seq_training_dir = ds_training_dir/seq_name
        # begin and end of the training part
        start_ts = seq_config['start']
        stop_ts = seq_config['stop']
        # initial timestamp of the sequence
        t0 = ds_info[seq_name]

        if start_ts is None:
            start_ts = 0

        # sure output dir exists
        seq_training_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(seq_raw_file), 'r') as data:
            events = data['davis']['left']['events']
            image_ts = data['davis']['left']['image_raw_ts']
            images = data['davis']['left']['image_raw']
            img2event_map = np.array(data['davis']
                                         ['left']
                                         ['image_raw_event_inds'],
                                     dtype=np.int)

            # mask of images to use
            mask = image_ts >= t0 + start_ts
            if stop_ts is not None:
                mask = np.logical_and(mask, image_ts <= t0 + stop_ts)

            image_ts = image_ts[mask]
            images = images[mask, :]
            img2event_map = img2event_map[mask]

            write_samples(events, images, image_ts,
                          img2event_map, seq_training_dir, t0)
