from functools import partial

from .loss import photometric_loss, smoothness_loss
from .timer import SynchronizedWallClockTimer

import torch
import torch.nn.functional as F

def interpolate(img, shape):
    return F.interpolate(img, size=shape, mode='bilinear', align_corners=True)

def mean(v):
    return sum(v) / len(v)

def combined_loss(evaluator, flows, image1, image2, features, weights=[0.5, 1, 1]):
    arths = (features[f'dec_flow_arth_{i}'] for i in range(len(flows)))
    terms = evaluator(flows, image1, image2, arths)
    loss = sum(map(lambda v, w: w*mean(v), terms, weights))
    return loss, terms

def make_hook_periodic(hook, checkpointing_interval):
    return lambda step, *args: (None if (step + 1) % checkpointing_interval else hook(step, *args))

def predictions2tag(predictions):
    return (f'{x.shape[-2]}x{x.shape[-1]}' for x in predictions)

# TODO: add losses object as the input to the train and validate functions
# use losses.losses.H and losses.losses.W to identify image size
# log all the losses

def train(model,
          device,
          loader,
          optimizer,
          num_steps : int,
          scheduler,
          logger,
          evaluator,
          weights=[0.5, 1, 1],
          is_raw=True,
          accumulation_steps=1,
          timers=SynchronizedWallClockTimer(),
          hooks={}):
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
        hooks (dict): hooks that should be called after each step of optimizer. Each hook is a Callable(steps, samples_passed)->None
    '''

    model.train()

    samples_passed = 0
    optimizer.zero_grad()
    timers('batch_construction').start()
    for global_step, (data, start, stop, image1, image2) in enumerate(loader):
        if global_step * accumulation_steps == num_steps:
            break
        timers('batch_construction').stop()
        timers('batch2gpu').start()
        data, start, stop, image1, image2 = map(lambda x: x.to(device), (data, start, stop, image1, image2))
        timers('batch2gpu').stop()
        shape = image1.size()[-2:]
        samples_passed += start.numel()
        timers('forward').start()
        prediction, features = model(data, start, stop, shape, raw=is_raw, intermediate=True)
        tags = predictions2tag(prediction)
        timers('forward').stop()
        timers('loss').start()
        loss, terms = combined_loss(evaluator, prediction, image1, image2, features, weights=weights)
        smoothness, photometric, out_reg = terms
        normalized_loss = loss / accumulation_steps
        timers('loss').stop()
        timers('backprop').start()
        normalized_loss.backward()
        timers('backprop').stop()

        is_step_boundary = (global_step + 1) % accumulation_steps == 0
        if is_step_boundary:
            timers('optimizer_step').start()
            optimizer.step()
            optimizer.zero_grad()
            timers('optimizer_step').stop()
            scheduler.step()

            step = (global_step + 1) // accumulation_steps
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

        # remove the graph
        timers('free').start()
        loss = loss.item()
        smoothness, photometric, out_reg = map(lambda x: (el.item() for el in x),
                                               [smoothness, photometric, out_reg])
        del prediction
        del features
        timers('free').stop()

        timers('logging').start()
        for tag, s, p, o in zip(tags, smoothness, photometric, out_reg):
            logger.add_scalar(f'Train/photometric loss/{tag}', p, samples_passed)
            logger.add_scalar(f'Train/smoothness loss/{tag}', s, samples_passed)
            logger.add_scalar(f'Train/out regularization/{tag}', o, samples_passed)
        logger.add_scalar(f'General/Train loss', loss, samples_passed)
        if is_step_boundary:
            for i, lr in enumerate([p['lr'] for p in optimizer.param_groups]):
                logger.add_scalar(f'General/learning rate/{i}', lr, samples_passed)
        timers('logging').stop()
        timers.log(names=['batch_construction',
                          'batch2gpu',
                          'forward',
                          'loss',
                          'backprop',
                          'optimizer_step',
                          'free',
                          'logging'] + list(hooks))
        timers('batch_construction').start()
    timers('batch_construction').stop()

def add_loss(loss_sum, loss_values):
    if len(loss_sum) == 0:
        return [x.item() for x in loss_values]
    return [x + y.item() for x, y in zip(loss_sum, loss_values)]

def validate(model, device, loader, samples_passed,
             logger, evaluator, weights=[0.5, 1, 1], is_raw=True):
    from utils.timer import SynchronizedWallClockTimer
    model.eval()

    n = len(loader)
    photo_sum = []
    smooth_sum = []
    out_reg_sum = []
    loss_sum = 0
    with torch.no_grad():
        for data, start, stop, image1, image2 in loader:
            data, start, stop, image1, image2 = map(lambda x: x.to(device), (data, start, stop, image1, image2))
            shape = image1.size()[-2:]
            prediction, features = model(data, start, stop, shape, raw=is_raw, intermediate=True)
            tags = predictions2tag(prediction)
            loss, terms = combined_loss(evaluator, prediction, image1, image2, features, weights=weights)
            smoothness, photometric, out_reg = terms
            photo_sum = add_loss(photo_sum, photometric)
            smooth_sum = add_loss(smooth_sum, smoothness)
            out_reg_sum = add_loss(out_reg_sum, out_reg)
            loss_sum += loss.item()
    logger.add_scalar('General/Validation loss', loss_sum/n, samples_passed)
    for tag, s, p, o in zip(tags, smooth_sum, photo_sum, out_reg_sum):
        logger.add_scalar(f'Validation/smoothness loss/{tag}', s/n, samples_passed)
        logger.add_scalar(f'Validation/photometric loss/{tag}', p/n, samples_passed)
        logger.add_scalar(f'Validation/out regularization loss/{tag}', o/n, samples_passed)
