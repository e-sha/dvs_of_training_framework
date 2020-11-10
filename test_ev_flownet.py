from pathlib import Path
import numpy as np
import h5py
import sys
import os

from dvs_optical_flow_ev_flownet import OpticalFlow

from utils.testing import evaluate, read_config, ravel_config
from utils.data import central_shift, EventCrop, ImageCrop
from utils.dataset import read_info

from argparse import ArgumentParser

def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to learned weights',
            default=None, required=False)

    return parser.parse_args(args)

args = parse_args(sys.argv[1:])
model_path = args.model

script_dir = Path(__file__).resolve().parent
data_dir = (script_dir/'..'/'data'/'raw').resolve()
info_dir = script_dir/'data'/'info'

if not model_path is None:
    model_path = str(Path(os.getcwd())/model_path)

config = read_config(script_dir/'config'/'testing.yml')

for ds_name, ds_config in config.items(): # process all datasets
    # dir with dataset data
    ds_dir = data_dir/ds_name
    # load info
    info_file = info_dir/(ds_name + '.hdf5')
    ds_info = read_info(str(info_file))
    for seq_name, seq_config in ds_config.items(): # process all sequences
        seq_file = ds_dir/seq_name[:-1]/(seq_name+'_data.hdf5')
        gt_file = ds_dir/'FlowGT'/seq_name[:-1]/(seq_name+'_gt_flow_dist.npz')

        # load events
        with h5py.File(str(seq_file), 'r') as data:
            events = np.array(data['davis']['left']['events'], dtype=np.float64).T
            image_ts = np.array(data['davis']['left']['image_raw_ts'], dtype=np.float64)

        # load gt
        gt = np.load(str(gt_file))
        gt = {k: gt[k] for k in gt.keys()}

        # first timestamp for the sequence
        first_ts = ds_info[seq_name]

        for cfg in ravel_config(seq_config):
            # get data from config
            start = cfg['start']
            stop = cfg['stop']
            step = cfg['step']
            test_shape = cfg['test_shape']
            crop_type = cfg['crop_type']
            is_car = cfg['is_car']

            if start is None:
                start = 0
            if stop is None:
                stop = min(events[2][-1], gt['timestamps'][-2]) - first_ts

            # generates frames
            start, stop = map(lambda x: x + first_ts, (start, stop))
            b, e = np.searchsorted(image_ts, [start, stop])
            frames = list(zip(image_ts[b:e-step], image_ts[b+step:e]))

            # prepare event preprocesser
            imshape = gt['x_flow_dist'].shape[1:]
            event_preproc_fun = None
            pred_postproc_fun = None
            gt_proc_fun = None

            if crop_type == 'central':
                box = list(central_shift(imshape, test_shape)) + test_shape
                event_preproc_fun = EventCrop(box)
                gt_proc_fun = ImageCrop(box)

            # generate optical flow predictor
            if model_path is None:
                of = OpticalFlow(test_shape)
            else:
                of = OpticalFlow(test_shape, model_path)

            mAEE, mpAEE = evaluate(of, events, frames, gt, is_car=is_car,
                event_preproc_fun=event_preproc_fun,
                pred_postproc_fun=pred_postproc_fun,
                gt_proc_fun=gt_proc_fun, log=False)
            print(f'[{seq_name}, {start}, {stop}, {step}, {test_shape}, {crop_type}, {is_car}]: Mean AEE: {mAEE:.6f}, mean %AEE: {mpAEE*100:.6f}')
