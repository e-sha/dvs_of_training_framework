from argparse import ArgumentParser
from copy import deepcopy
import h5py
import math
import multiprocessing
import numpy as np
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile
import torch
import torch.utils.tensorboard
from types import SimpleNamespace

from utils.data import central_shift, EventCrop, ImageCrop
from utils.dataset import read_info
from utils.model import filter_kwargs, import_module
from utils.options import add_test_arguments, validate_test_args, options2model_kwargs
from utils.serializer import Serializer
from utils.testing import evaluate, read_config, ravel_config

def parse_args():
    parser = ArgumentParser()
    args = add_test_arguments(parser).parse_args()
    args = validate_test_args(args)
    return args

def get_output_path(args):
    if args.model.suffix == '.pt':
        model_path = args.model
    else:
        serializer = Serializer(args.model)
        model_path = serializer._id2path(args.step)
    return args.output/(model_path.stem + '.pkl')

def preprocess_args(args):
    args.output = get_output_path(args)
    args.is_temporary_model = True
    f = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    Serializer(args.model).finalize(args.step, f.name, map_location=args.device)
    args.model = Path(f.name)
    f.close()
    return args

def init_model(args, test_shape):
    module = import_module(f'{args.flownet_path}.__init__', args.flownet_path/'__init__.py')
    model_kwargs = options2model_kwargs(args)
    model_kwargs = filter_kwargs(module.OpticalFlow, model_kwargs)
    model_kwargs.update({'device': args.device})
    if args.model is None:
        return module.OpticalFlow(test_shape, **model_kwargs)
    else:
        return module.OpticalFlow(test_shape, model=args.model, **model_kwargs)

def load_events(path):
    with h5py.File(str(path), 'r') as data:
        events = np.array(data['davis']['left']['events'], dtype=np.float64).T
        image_ts = np.array(data['davis']['left']['image_raw_ts'], dtype=np.float64)
    return events, image_ts

def load_gt(path):
    gt = np.load(str(path))
    return {k: gt[k] for k in gt.keys()}

def get_preprocessing_functions(imshape, test_shape, crop_type):
    pred_postproc_fun = None

    if crop_type == 'central':
        box = list(central_shift(imshape, test_shape)) + test_shape
        return EventCrop(box), ImageCrop(box)
    else:
        raise ValueError(f'Unknown crop type "{crop_type}"')

def postprocess_config(config, dataset):
    if config.start is None:
        config.start = dataset.first_ts
    else:
        config.start += dataset.first_ts

    if config.stop is None:
        config.stop = min(dataset.events[2][-1], dataset.gt['timestamps'][-2])
    else:
        config.stop += dataset.first_ts
    return config

def generate_frames(cfg, image_ts):
    b, e = np.searchsorted(image_ts, [cfg.start, cfg.stop])
    return list(zip(image_ts[b : e - cfg.step], image_ts[b + cfg.step : e]))

def seq2paths(dataset_path, seq_name):
    seq_type = re.sub(r'\d+$', '', seq_name)
    seq_file = dataset_path/seq_type/(seq_name+'_data.hdf5')
    gt_file = dataset_path/'FlowGT'/seq_type/(seq_name+'_gt_flow_dist.npz')
    return seq_file, gt_file

def perform_single_test(args, cfg, dataset):
    cfg = postprocess_config(cfg, dataset)
    dataset.is_car = cfg.is_car

    # generates frames
    dataset.frames = generate_frames(cfg, dataset.image_ts)

    # prepare event preprocesser
    event_preproc_fun, gt_proc_fun = get_preprocessing_functions(dataset.imshape,
                                                                 cfg.test_shape,
                                                                 cfg.crop_type)

    # generate optical flow predictor
    of = init_model(args, cfg.test_shape)

    return evaluate(of, dataset.events, dataset.frames, dataset.gt, is_car=dataset.is_car,
                    event_preproc_fun=event_preproc_fun,
                    pred_postproc_fun=None,
                    gt_proc_fun=gt_proc_fun, log=False)

def process_single(args):
    args = preprocess_args(args)
    if args.output.is_file():
        if args.is_temporary_model:
            args.model.unlink()
        return

    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir/'..'/'data'/'raw').resolve()
    info_dir = script_dir/'data'/'info'

    config = read_config(script_dir/'config'/'testing.yml')

    results = []

    for ds_name, ds_config in config.items(): # process all datasets
        # dir with dataset data
        ds_dir = data_dir/ds_name
        # load info
        info_file = info_dir/(ds_name + '.hdf5')
        ds_info = read_info(str(info_file))
        for seq_name, seq_config in ds_config.items(): # process all sequences
            seq_file, gt_file = seq2paths(ds_dir, seq_name)

            dataset = SimpleNamespace(name=seq_name)
            dataset.events, dataset.image_ts = load_events(seq_file)
            dataset.gt = load_gt(gt_file)
            dataset.imshape = dataset.gt['x_flow_dist'].shape[1:]

            # first timestamp for the sequence
            dataset.first_ts = ds_info[seq_name]

            for cfg in ravel_config(seq_config):
                cfg.dataset = ds_name
                cfg.sequence = seq_name
                cfg.mAEE, cfg.mpAEE = perform_single_test(args, cfg, dataset)
                results.append(cfg)
                print(f'[{cfg.sequence}, {cfg.start}, {cfg.stop}, {cfg.step}, {cfg.test_shape}, {cfg.crop_type}, {cfg.is_car}]: Mean AEE: {cfg.mAEE:.6f}, mean %AEE: {cfg.mpAEE*100:.6f}')
    with args.output.open('wb') as f:
        pickle.dump(results, f)
    if args.is_temporary_model:
        args.model.unlink()

def get_samples_passed(args):
    serializer = Serializer(args.model)
    model_path = serializer._id2path(args.step)
    data = torch.load(model_path, map_location='cpu')
    return data.get('samples_passed', data['global_step'] * args.bs)

class GPUPool:
    def __init__(self, pool, gpus, tests_per_gpu, timeout=1):
        self._pool = pool
        self._gpus = gpus
        self._tests_per_gpu = tests_per_gpu
        self._timeout = timeout

    def _wait(self, results, decrease=False):
        is_continue = True
        while is_continue:
            is_continue = decrease
            for d, device_results in results.items():
                after = []
                for r in device_results:
                    if r.ready():
                        is_continue = False
                    else:
                        after.append(r)
                results[d] = after
            if is_continue:
                time.sleep(self._timeout)
        return results

    def _best_device(self, results):
        best_device = results.keys()[0]
        for device in results:
            if len(results[device]) < len(results[best_device]):
                best_device = device
        return best_device

    def __call__(self, func, args_list):
        devices = self._gpus
        results = {device: [] for device in self._gpus}
        with self._pool:
        for args in args_list:
            decrease = False
            while True:
                results = self._wait(results, decrease=decrease)
                best_device = self._best_device(results)
                if len(results[best_device]) >= self._tests_per_gpu:
                    decrease = True
                else:
                    break
            args.device = best_device
            results[best_device].append(self._pool.apply_async(func, args))
        for _, device_results in results.items():
            for r in device_results:
                r.wait()

def process_all(args):
    args.__dict__.pop('step', None)
    serializer = Serializer(args.model)
    all_args = [SimpleNamespace(step=s, **args.__dict__) for s in serializer.list_known_steps()]
    with multiprocessing.Pool(args.tests_per_gpu) as p:
        GPUPool(p, gpus, tests_per_gpu)(process_single, all_args)
        #p.map(process_single, all_args)
    writer = torch.utils.tensorboard.SummaryWriter(args.output/'log')
    for step_args in all_args:
        samples_passed = get_samples_passed(step_args)
        with get_output_path(step_args).open('rb') as f:
            results = pickle.load(f)
        for result in results:
            tag = f'{result.dataset}/{result.sequence}/{result.step}/{result.start}/{result.stop}'
            writer.add_scalar(f'Test/mean AEE/{tag}', result.mAEE, samples_passed)
            writer.add_scalar(f'Test/mean %AEE/{tag}', result.mpAEE * 100, samples_passed)

def main():
    args = parse_args()
    if args.step is None:
        process_all(args)
    else:
        process_single(args)

if __name__=='__main__':
    main()
