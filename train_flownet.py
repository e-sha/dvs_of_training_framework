from argparse import ArgumentParser
from pathlib import Path
import sys
import torch
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from utils.dataloader import get_trainset_params, get_valset_params
from utils.dataloader import get_dataloader, choose_data_path
from utils.common import collect_execution_info, write_execution_info
from utils.common import check_execution_info
from utils.hooks.validation import ValidationHook
from utils.hooks.serialization import SerializationHook
from utils.loss import init_losses
from utils.model import init_model
from utils.monitors.gpumonitor import GPUMonitor
from utils.options import add_train_arguments
from utils.options import validate_train_args
from utils.profiling import Profiler
from utils.serializer import Serializer
from utils.timer import SynchronizedWallClockTimer, FakeTimer
from utils.training import train, make_hook_periodic


script_dir = Path(__file__).resolve().parent


def parse_args(args, is_write=True):
    parser = ArgumentParser()
    args = add_train_arguments(parser).parse_args(args)
    args = validate_train_args(args)
    args = choose_data_path(args)

    args.model.mkdir(exist_ok=True, parents=True)
    args.log_path = args.model/'log'

    execution_info = collect_execution_info(args)
    check_execution_info(args.model, execution_info, args)
    if is_write:
        write_execution_info(args.model, execution_info)
    return args


def get_params2optimize(model):
    if hasattr(model, 'quantization_layer'):
        return [{'params': model.quantization_layer.parameters()},
                {'params': model.predictor.parameters()}]
    return [{'params': model.parameters()}]


def construct_optimizer(args, params):
    kwargs = {}
    if args.optimizer == 'ADAM':
        opt = optim.AdamW
        kwargs = {'amsgrad': True}
    elif args.optimizer == 'RADAM':
        from RAdam.radam import RAdam
        opt = RAdam
    elif args.optimizer == 'RANGER':
        ranger_path = script_dir/'Ranger-Deep-Learning-Optimizer'
        sys.path.append(str(ranger_path))

        from ranger import Ranger

        opt = Ranger
    else:
        assert hasattr(torch.optim, args.optimizer), 'Unknown optimizer type'
        opt = getattr(torch.optim, args.optimizer)
    return opt(params, lr=args.lr, weight_decay=args.wdw, **kwargs)


def construct_train_tools(args, model, passed_steps=0):
    is_splitted = hasattr(model, 'quantization_layer')
    representation_start = args.training_steps * args.rs
    if is_splitted:
        representation_params = [{
            'params': model.quantization_layer.parameters(),
            'weight_decay': args.wdw}]
        predictor_params = [{'params': model.predictor.parameters()}]
    else:
        representation_params = []
        predictor_params = [{'params': model.parameters(),
                             'weight_decay': args.wdw}]

    def pred_scheduler(step):
        return 2 ** (-step / args.half_life)

    def repr_scheduler(step):
        if step > representation_start:
            return pred_scheduler(step)
        return 0

    optimizer = construct_optimizer(args,
                                    representation_params + predictor_params)
    scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=([repr_scheduler] * len(representation_params) +
                       [pred_scheduler] * len(predictor_params)))
    for _ in range(passed_steps):
        scheduler.step()
    return optimizer, scheduler


def create_hooks(args, model, optimizer, losses, logger, serializer):
    device = torch.device(args.device)
    loader = get_dataloader(get_valset_params(args))
    hooks = {'serialization': SerializationHook(serializer, model, optimizer,
                                                logger)}
    periods = {'serialization': args.checkpointing_interval}
    if not args.skip_validation:
        hooks['validation'] = ValidationHook(model, device, loader, logger,
                                             losses, weights=args.loss_weights,
                                             is_raw=args.is_raw)
        periods['validation'] = args.vp
    periodic_hooks = {k: make_hook_periodic(hooks[k], periods[k])
                      for k in periods}
    return periodic_hooks, hooks


def main():
    # torch.autograd.set_detect_anomaly(True)

    args = parse_args(sys.argv[1:])

    device = torch.device(args.device)
    if device.type == 'cuda':
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

    args.do_not_continue = (args.do_not_continue or
                            len(serializer.list_known_steps()) == 0)
    last_step = (0
                 if args.do_not_continue
                 else serializer.list_known_steps()[-1])

    optimizer, scheduler = construct_train_tools(args,
                                                 model,
                                                 passed_steps=last_step)

    losses = init_losses(args.shape,
                         args.bs, model,
                         device,
                         sequence_length=args.prefix_length +
                         args.suffix_length + 1,
                         timers=timers)

    # allow only manual flush
    logger = SummaryWriter(str(args.log_path),
                           max_queue=100000000,
                           flush_secs=100000000)

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

    if not args.skip_validation:
        hooks['validation'](global_step, samples_passed)

    with Profiler(args.profiling, args.model/'profiling'), \
            GPUMonitor(args.log_path):
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
              init_samples_passed=samples_passed,
              max_events_per_batch=args.max_events_per_batch)

    samples = samples_passed + (args.training_steps - global_step) * args.bs
    hooks['serialization'](args.training_steps, samples)
    if not args.skip_validation:
        hooks['validation'](args.training_steps, samples)


if __name__ == '__main__':
    main()
