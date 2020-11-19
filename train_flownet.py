from pathlib import Path
import numpy as np
import h5py
import sys
import os
import yaml
import subprocess
from types import SimpleNamespace

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
from utils.dataset import read_info, Dataset, collate_wrapper
from utils.options import train_parser, options2model_kwargs
from utils.training import train, validate
from utils.loss import Losses

def init_losses(shape, batch_size, model, device):
    events = torch.zeros((1, 5), dtype=torch.float32).numpy()
    out = model(events, torch.tensor([0]).numpy(), torch.tensor([0.4]).numpy(), shape, raw=True)
    out_shapes = tuple(tuple(flow.shape[2:]) for flow in out)
    return Losses(out_shapes, batch_size, device)

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
    args = train_parser().parse_args()
    args = choose_data_path(args)
    write_params(args.model, args)
    args.is_raw = not args.ev_images
    return args

def get_resolution(args):
    return args.height, args.width

def get_common_dataset_params(args):
    is_raw = args.is_raw
    collate_fn = collate_wrapper if is_raw else None
    return SimpleNamespace(is_raw=is_raw,
                           collate_fn=collate_fn,
                           shape=get_resolution(args),
                           batch_size=args.bs,
                           pin_memory=True,
                           num_workers=args.bs)

def get_trainset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day2'
    params.augmentation = True
    params.collapse_length = args.cl
    params.shuffle = True
    return params

def get_valset_params(args):
    params = get_common_dataset_params(args)
    params.path = args.data_path/'outdoor_day1'
    params.augmentation = False
    params.collapse_length = 1
    params.shuffle = False
    return params

def get_dataloader(params):
    dataset = Dataset(path=params.path,
                      shape=params.shape,
                      augmentation=params.augmentation,
                      collapse_length=params.augmentation,
                      is_raw=params.is_raw)
    return torch.utils.data.DataLoader(dataset,
                                       collate_fn=params.collate_fn,
                                       batch_size=params.batch_size,
                                       shuffle=params.shuffle,
                                       num_workers=params.num_workers,
                                       pin_memory=params.pin_memory)


def init_model(args, device):
    if args.ev_flownet:
        from EV_FlowNet.net import Model
        model_kwargs = {}
    else:
        assert False, 'Not implemented'
        model_kwargs = options2model_kwargs(args)
    model = Model(device, **model_kwargs)
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

def construct_optimizer_and_scheduler(args, model):
    is_splitted = hasattr(model, 'quantization_layer')
    if is_splitted:
        representation_params = [{'params': model.quantization_layer.parameters(), 'weight_decay': args.wdw}]
        predictor_params = [{'params': model.predictor.parameters()}]
    else:
        representation_params = []
        predictor_params = [{'params': model.parameters(), 'weight_decay': args.wdw}]

    predictor_scheduler = lambda step: args.lr_gamma ** (step // args.step)
    representation_scheduler = lambda step: predictor_scheduler(step) if epoch > args.rs else 0

    optimizer = construct_optimizer(args, representation_params + predictor_params)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lr_lambda=[representation_scheduler] * len(representation_params) + \
                                                      [predictor_scheduler] * len(predictor_params))
    return optimizer, scheduler

def main():
    #torch.autograd.set_detect_anomaly(True)

    args = parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    model = init_model(args, device)

    train_loader = get_dataloader(get_trainset_params(args))
    val_loader = get_dataloader(get_valset_params(args))

    optimizer, scheduler = construct_optimizer_and_scheduler(args, model)

    losses = init_losses(get_resolution(args), args.bs, model, device)

    logger = SummaryWriter(str(args.log_path))

    validate(model, device, val_loader, 0, logger, losses,
             weights=args.loss_weights, is_raw=args.is_raw)
    for epoch in range(args.epochs):
        torch.save(model.state_dict(), args.model/f'{epoch}.pth')
        train(model, device, train_loader, optimizer, epoch,
              evaluator=losses,
              logger=logger,
              weights=args.loss_weights, is_raw=args.is_raw,
              accumulation_step=args.accum_step)
        if (epoch+1) % args.vp == 0:
            validate(model, device, val_loader,
                (epoch + 1) * len(train_loader), logger,
                losses,
                weights=args.loss_weights, is_raw=args.is_raw)
        scheduler.step()
    torch.save(model.state_dict(), args.model/f'{args.epochs}.pth')

if __name__=='__main__':
    main()
