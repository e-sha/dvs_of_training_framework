from pathlib import Path
import numpy as np
import h5py
import sys
import os
import yaml
import subprocess
from types import SimpleNamespace
import inspect
from functools import partial
from argparse import ArgumentParser

script_dir = Path(__file__).resolve().parent
ranger_path = script_dir/'Ranger-Deep-Learning-Optimizer'
sys.path.append(str(ranger_path))

import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn

from ranger import Ranger
from RAdam.radam import RAdam

from utils.testing import evaluate
from utils.dataset import read_info, Dataset, IterableDataset, collate_wrapper
from utils.options import add_train_arguments, validate_train_args, options2model_kwargs
from utils.training import train, validate, make_hook_periodic
from utils.loss import Losses
from utils.timer import SynchronizedWallClockTimer, FakeTimer
from utils.serializer import Serializer
from utils.model import import_module, filter_kwargs

def init_losses(shape, batch_size, model, device, timers=FakeTimer()):
    events = torch.zeros((0, 5), dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(events,
                    torch.tensor([0], dtype=torch.float32, device=device),
                    torch.tensor([0.04], dtype=torch.float32, device=device),
                    shape, raw=True)
    out_shapes = tuple(tuple(flow.shape[2:]) for flow in out)
    return Losses(out_shapes, batch_size, device, timers=timers)

def get_commithash():
    return subprocess.check_output('git rev-parse --verify HEAD', shell=True).decode().strip()

def write_params(out_dir, args):
    data2write = '\n'.join([' '.join(sys.argv),
                            f'commit_hash: {get_commithash()}',
                            yaml.dump(vars(args))])
    (out_dir/'parameters').write_text(data2write)

def is_inside_docker():
    return 'INSIDE_DOCKER' in os.environ and bool(os.environ['INSIDE_DOCKER'])

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

def parse_args():
    parser = ArgumentParser()
    args = add_train_arguments(parser).parse_args()
    args = validate_train_args(args)
    args = choose_data_path(args)
    write_params(args.model, args)
    return args

def get_resolution(args):
    return args.height, args.width

def get_common_dataset_params(args):
    is_raw = args.is_raw
    collate_fn = collate_wrapper if is_raw else None
    return SimpleNamespace(is_raw=is_raw,
                           collate_fn=collate_fn,
                           shape=get_resolution(args),
                           batch_size=args.mbs,
                           pin_memory=True,
                           num_workers=args.num_workers)

def get_trainset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day2'
    params.augmentation = True
    params.collapse_length = args.cl
    params.shuffle = True
    params.infinite = True
    return params

def get_valset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day1'
    params.augmentation = False
    params.collapse_length = 1
    params.shuffle = False
    params.infinite = False
    return params

def get_dataloader(params):
    kwargs = {'path': params.path,
              'shape': params.shape,
              'augmentation': params.augmentation,
              'collapse_length': params.augmentation,
              'is_raw': params.is_raw}
    loader_kwargs = {}
    if params.infinite:
        dataset = IterableDataset(shuffle = params.shuffle, **kwargs)
    else:
        dataset = Dataset(**kwargs)
        loader_kwargs['shuffle'] = params.shuffle
    return torch.utils.data.DataLoader(dataset,
                                       collate_fn=params.collate_fn,
                                       batch_size=params.batch_size,
                                       num_workers=params.num_workers,
                                       pin_memory=params.pin_memory,
                                       **loader_kwargs)


def init_model(args, device):
    module = import_module(f'{args.flownet_path.name}.net', args.flownet_path/'net.py')
    model_kwargs = options2model_kwargs(args)
    model_kwargs = filter_kwargs(module.Model, model_kwargs)
    model = module.Model(device, **model_kwargs)
    if not args.sp is None:
        model.load_state_dict(torch.load(args.sp, map_location=device))
    model.to(device)
    return model

def get_params2optimize(model):
    if hasattr(model, 'quantization_layer'):
        return [{'params': model.quantization_layer.parameters()},
                {'params': model.predictor.parameters()}]
    return [{'params': model.parameters()}]

def construct_optimizer(args, params):
    kwargs = {}
    if args.optimizer == 'ADAM':
        opt = optim.AdamW
        kwargs = {'amsgrad':True}
    elif args.optimizer == 'RADAM':
        opt = RAdam
    elif args.optimizer == 'RANGER':
        opt = Ranger
    else:
        assert hasattr(torch.optim, args.optimizer), 'Unknown optimizer type'
        opt = getattr(torch.optim, args.optimizer)
    return opt(params, lr=args.lr, weight_decay=args.wdw, **kwargs)

def construct_optimizer_and_scheduler(args, model, passed_steps=0):
    is_splitted = hasattr(model, 'quantization_layer')
    representation_start = args.training_steps * args.rs
    if is_splitted:
        representation_params = [{'params': model.quantization_layer.parameters(), 'weight_decay': args.wdw}]
        predictor_params = [{'params': model.predictor.parameters()}]
    else:
        representation_params = []
        predictor_params = [{'params': model.parameters(), 'weight_decay': args.wdw}]

    predictor_scheduler = lambda step: 2 ** (-step / args.half_life)
    representation_scheduler = lambda step: predictor_scheduler(step) if step > representation_start else 0

    optimizer = construct_optimizer(args, representation_params + predictor_params)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=[representation_scheduler] * len(representation_params) + \
                                                      [predictor_scheduler] * len(predictor_params))
    for _ in range(passed_steps):
        scheduler.step()
    return optimizer, scheduler

def create_hooks(args, model, optimizer, losses, logger, serializer):
    device = torch.device(args.device)
    loader = get_dataloader(get_valset_params(args))
    hooks = {'serialization': lambda steps, samples: serializer.checkpoint_model(model,
                                                                                 optimizer,
                                                                                 global_step=steps,
                                                                                 samples_passed=samples),
             'validation': lambda step, samples: validate(model,
                                                          device,
                                                          loader,
                                                          samples,
                                                          logger,
                                                          losses,
                                                          weights=args.loss_weights,
                                                          is_raw=args.is_raw)}
    periods = {'serialization': args.checkpointing_interval,
               'validation': args.vp}
    periodic_hooks = {k: make_hook_periodic(hooks[k], periods[k]) for k in periods}
    return periodic_hooks, hooks

def main():
    #torch.autograd.set_detect_anomaly(True)

    args = parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.timers:
        timers = SynchronizedWallClockTimer()
    else:
        timers = FakeTimer()

    model = init_model(args, device)

    loader = get_dataloader(get_trainset_params(args))

    serializer = Serializer(args.model,
                            args.num_checkpoints,
                            args.permanent_interval)

    args.do_not_continue = args.do_not_continue or len(serializer.list_known_steps()) == 0
    last_step = 0 if args.do_not_continue else serializer.list_known_steps()[-1]

    optimizer, scheduler = construct_optimizer_and_scheduler(args, model, passed_steps=last_step)

    losses = init_losses(get_resolution(args), args.bs, model, device, timers=timers)

    logger = SummaryWriter(str(args.log_path))

    periodic_hooks, hooks = create_hooks(args,
                                         model,
                                         optimizer,
                                         losses,
                                         logger,
                                         serializer)

    if not args.do_not_continue:
        global_step, state = serializer.load_checkpoint(model,
                                                        last_step,
                                                        optimizer=optimizer,
                                                        device=device)
        samples_passed = state.pop('samples_passed', global_step * args.bs)
    else:
        global_step = 0
        samples_passed = 0
        hooks['serialization'](global_step, samples_passed)

    hooks['validation'](global_step, samples_passed)

    if args.profiling:
        prof = torch.autograd.profiler.profile().__enter__()
    try:
        train(model,
              device,
              loader,
              optimizer,
              args.training_steps,
              scheduler=scheduler,
              evaluator=losses,
              logger=logger,
              weights=args.loss_weights,
              is_raw=args.is_raw,
              accumulation_steps=args.accum_step,
              timers=timers,
              hooks=periodic_hooks,
              init_step=global_step,
              init_samples_passed=samples_passed)
    finally:
        if args.profiling:
            prof.__exit__(None, None, None)
            print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'))
            prof.export_chrome_trace('tracefile.json')

    samples = samples_passed + (args.training_steps - global_step) * args.bs
    hooks['serialization'](args.training_steps, samples)
    hooks['validation'](args.training_steps, samples)

if __name__=='__main__':
    main()
