import torch
import torch.nn.functional as F

from .common import mean
from .timer import SynchronizedWallClockTimer, FakeTimer


def interpolate(img, shape):
    return F.interpolate(img, size=shape, mode='bilinear', align_corners=True)


def combined_loss(evaluator,
                  flows,
                  flow_ts,
                  flow_sample_idx,
                  images,
                  timestamps,
                  sample_idx,
                  features,
                  weights=[0.5, 1, 1]):
    terms = evaluator(flows, flow_ts, flow_sample_idx, images,
                      timestamps, sample_idx)
    loss = sum(map(lambda v, w: w*mean(v), terms, weights))
    return loss, terms


def make_hook_periodic(hook, checkpointing_interval):
    return lambda step, *args: (None
                                if step % checkpointing_interval
                                else hook(step, *args))


def predictions2tag(predictions):
    return (f'{x.shape[-2]}x{x.shape[-1]}' for x in predictions)


def process_minibatch(model,
                      batch,
                      timers,
                      device,
                      is_raw,
                      evaluator,
                      weights,
                      return_prediction=False):
    timers('batch2gpu').start()
    timestamps, sample_idx, images = map(lambda x: x.to(device),
                                         (batch['timestamps'],
                                          batch['sample_idx'],
                                          batch['images']))
    if is_raw:
        events = batch['events']
        for k in set.difference(set(events.keys()), {'size'}):
            events[k] = events[k].to(device)
    else:
        events = batch['data'].to(device)
    timers('batch2gpu').stop()
    shape = images.size()[-2:]
    timers('forward').start()
    prediction, flow_ts, flow_sample_idx, features = model(events,
                                                           timestamps,
                                                           sample_idx,
                                                           shape,
                                                           raw=is_raw,
                                                           intermediate=True)
    tags = predictions2tag(prediction)
    timers('forward').stop()
    timers('loss').start()
    loss, terms = combined_loss(evaluator,
                                prediction,
                                flow_ts,
                                flow_sample_idx,
                                images,
                                timestamps,
                                sample_idx,
                                features,
                                weights=weights)
    terms = ((y.item() for y in x) for x in terms)
    timers('loss').stop()
    add_info = tuple()
    if return_prediction:
        add_info = (
            {'prediction': prediction,
             'flow_ts': flow_ts,
             'flow_sample_idx': flow_sample_idx,
             'features': features}, )
    return (loss, terms, tags) + add_info


def train(model,
          device,
          loader,
          optimizer,
          num_steps: int,
          scheduler,
          logger,
          evaluator,
          weights=[0.5, 1, 1],
          is_raw=True,
          accumulation_steps=1,
          timers=SynchronizedWallClockTimer(),
          hooks={},
          init_step=0,
          init_samples_passed=0,
          max_events_per_batch: int = 350000):
    ''' Performs training

    Args:
        model (nn.Module): a model to train
        device (torch.device): a device used for training
        loader (torch.utils.data.DataLoader): a training data loader
        optimzer (torch.optim.optimizer.Optimizer): a used optimizer
        num_steps (int): number of training steps
        scheduler (torch.optimi.lr_scheduler): scheduler that updates lr
        logger (class): a class to store logs
        evaluator (class): a class to compute loss
        weights (float, float, float): weights of the loss functions
        is_raw (bool): does use raw event stream
        accumulation_steps (int): gradient accumulation steps
        hooks (dict): hooks that should be called after each step of optimizer.
                      Each hook is a Callable(steps, samples_passed)->None
        max_events_per_batch:
            Maximum number of events in a batch. If any batch has more events,
            it is skipped.
    '''

    model.train()

    samples_passed = init_samples_passed
    loss_sum = 0
    smooth_sum = []
    photo_sum = []
    out_reg_sum = []
    optimizer.zero_grad(set_to_none=True)
    init_batch = init_step * accumulation_steps
    global_step = init_batch
    num_skipped = 0
    timers('batch_construction').start()
    for batch in loader:
        if global_step == num_steps * accumulation_steps:
            break
        num_events = batch['events']['x'].numel() if is_raw else 0
        if num_events > max_events_per_batch:
            num_skipped += 1
            num_processed = global_step - init_batch
            print(f'Skipping batch with {num_events} events')
            print('Augmentation parameters '
                  f'{batch["augmentation_params"]}')
            print('Processing rate is '
                  f'{num_processed / (num_processed + num_skipped):.2f}')
            continue
        global_step += 1
        timers('batch_construction').stop()
        samples_passed += batch['size']
        loss, (smoothness, photometric, out_reg), tags = process_minibatch(
                model, batch, timers, device, is_raw, evaluator, weights)
        loss /= accumulation_steps
        timers('backprop').start()
        loss.backward()
        timers('backprop').stop()

        is_step_boundary = global_step % accumulation_steps == 0
        if is_step_boundary:
            timers('optimizer_step').start()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            timers('optimizer_step').stop()
            scheduler.step()

            timers('logging').start()
            photo_sum = add_loss(photo_sum, photometric)
            smooth_sum = add_loss(smooth_sum, smoothness)
            out_reg_sum = add_loss(out_reg_sum, out_reg)
            loss_sum += loss.item()

            for tag, s, p, o in zip(tags, smooth_sum, photo_sum, out_reg_sum):
                logger.add_scalar(f'Train/photometric loss/{tag}',
                                  p / accumulation_steps,
                                  samples_passed)
                logger.add_scalar(f'Train/smoothness loss/{tag}',
                                  s / accumulation_steps,
                                  samples_passed)
                logger.add_scalar(f'Train/out regularization/{tag}',
                                  o / accumulation_steps,
                                  samples_passed)
            logger.add_scalar('General/Train loss',
                              loss_sum,
                              samples_passed)
            if is_step_boundary:
                for i, lr in enumerate([p['lr']
                                        for p in optimizer.param_groups]):
                    logger.add_scalar(f'General/learning rate/{i}',
                                      lr,
                                      samples_passed)

            loss_sum = 0
            smooth_sum = []
            photo_sum = []
            out_reg_sum = []
            timers('logging').stop()

            step = global_step // accumulation_steps
            for k, hook in hooks.items():
                timers(k).start()
                hook(step, samples_passed)
                timers(k).stop()
            # make sure to return to train after all hooks
            model.train()
        else:
            timers('optimizer_step').start()
            timers('optimizer_step').stop()
            for k, hook in hooks.items():
                timers(k).start()
                timers(k).stop()
            # losses for logging
            timers('logging').start()
            photo_sum = add_loss(photo_sum, photometric)
            smooth_sum = add_loss(smooth_sum, smoothness)
            out_reg_sum = add_loss(out_reg_sum, out_reg)
            loss_sum += loss.item()
            timers('logging').stop()

        timers.log(names=['batch_construction',
                          'batch2gpu',
                          'forward',
                          'loss',
                          'grid_construction',
                          'photometric_loss',
                          'smoothness_loss',
                          'outborder_loss',
                          'backprop',
                          'optimizer_step',
                          'free',
                          'logging'] + list(hooks))
        timers('batch_construction').start()
    timers('batch_construction').stop()


def add_loss(loss_sum, loss_values):
    if len(loss_sum) == 0:
        return list(loss_values)
    return [x + y for x, y in zip(loss_sum, loss_values)]


def validate(model, device, loader, samples_passed,
             logger, evaluator, weights=[0.5, 1, 1], is_raw=True):
    model.eval()

    n = len(loader)
    photo_sum = []
    smooth_sum = []
    out_reg_sum = []
    loss_sum = 0
    with torch.no_grad():
        for batch in loader:
            loss, (smoothness, photometric, out_reg), tags = process_minibatch(
                model, batch, FakeTimer(), device, is_raw, evaluator, weights)
            photo_sum = add_loss(photo_sum, photometric)
            smooth_sum = add_loss(smooth_sum, smoothness)
            out_reg_sum = add_loss(out_reg_sum, out_reg)
            loss_sum += loss.item()
    logger.add_scalar('General/Validation loss', loss_sum/n, samples_passed)
    for tag, s, p, o in zip(tags, smooth_sum, photo_sum, out_reg_sum):
        logger.add_scalar(f'Validation/smoothness loss/{tag}',
                          s/n,
                          samples_passed)
        logger.add_scalar(f'Validation/photometric loss/{tag}',
                          p/n,
                          samples_passed)
        logger.add_scalar(f'Validation/out regularization loss/{tag}',
                          o/n,
                          samples_passed)
