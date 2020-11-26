from functools import partial

from .loss import photometric_loss, smoothness_loss
from .timer import SynchronizedWallClockTimer

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

# TODO: add losses object as the input to the train and validate functions
# use losses.losses.H and losses.losses.W to identify image size
# log all the losses

def train(model, device, train_loader, optimizer, epoch,
        logger, evaluator, weights=[0.5, 1, 1], is_raw=True,
        accumulation_step=1, timers=SynchronizedWallClockTimer()):
    ''' Performs one epoch of training

    Args:
        model (nn.Module): a model to train
        device (torch.device): a device used for training
        train_loader (torch.utils.data.DataLoader): a training data loader
        optimzer (torch.optim.optimizer.Optimizer): a used optimizer
        epoch (int): current epoch
        logger (class): a class to store logs
        evaluator (class): a class to compute loss
        weights (float, float, float): weights of the loss functions
        is_raw (bool): does use raw event stream
        accumulation_step (int): gradient accumulation step
    '''

    model.train()

    n = len(train_loader)
    photo_sum = 0
    smooth_sum = 0
    loss_sum = 0
    optimizer.zero_grad()
    global_step = epoch * n
    timers('batch_construction').start()
    for data, start, stop, image1, image2 in train_loader:
        timers('batch_construction').stop()
        global_step += 1
        timers('batch2gpu').start()
        data, start, stop, image1, image2 = map(lambda x: x.to(device), (data, start, stop, image1, image2))
        timers('batch2gpu').stop()
        shape = image1.size()[-2:]
        timers('forward').start()
        prediction, features = model(data, start, stop, shape, raw=is_raw, intermediate=True)
        timers('forward').stop()
        timers('loss').start()
        loss, terms = combined_loss(evaluator, prediction, image1, image2, features, weights=weights)
        smoothness, photometric, out_reg = terms
        normalized_loss = loss / accumulation_step
        timers('loss').stop()
        timers('backprop').start()
        normalized_loss.backward()
        timers('backprop').stop()
        if global_step % accumulation_step == 0:
            timers('optimizer_step').start()
            optimizer.step()
            optimizer.zero_grad()
            timers('optimizer_step').stop()
        else:
            timers('optimizer_step').start()
            timers('optimizer_step').stop()

        timers('logging').start()
        for i, (s, p, o) in enumerate(zip(smoothness, photometric, out_reg)):
            logger.add_scalar(f'Train/photometric loss/{i}', p.item(), global_step)
            logger.add_scalar(f'Train/smoothness loss/{i}', s.item(), global_step)
            logger.add_scalar(f'Train/out regularization/{i}', o.item(), global_step)
        logger.add_scalar(f'Train/loss', loss.item(), global_step)
        for i, lr in enumerate([p['lr'] for p in optimizer.param_groups]):
            logger.add_scalar(f'learning rate/{i}', lr, global_step)
        timers('logging').stop()
        timers.log(names=['batch_construction',
                          'batch2gpu',
                          'forward',
                          'loss',
                          'backprop',
                          'optimizer_step',
                          'logging'])
        timers('batch_construction').start()
    timers('batch_construction').stop()

def add_loss(loss_sum, loss_values):
    if len(loss_sum) == 0:
        return [x.item() for x in loss_values]
    return [x + y.item() for x, y in zip(loss_sum, loss_values)]

def validate(model, device, loader, global_step,
        logger, evaluator, weights=[0.5, 1, 1], is_raw=True):
    model.eval()

    n = len(loader)
    photo_sum = []
    smooth_sum = []
    out_reg_sum = []
    loss_sum = 0
    for data, start, stop, image1, image2 in loader:
        data, start, stop, image1, image2 = map(lambda x: x.to(device), (data, start, stop, image1, image2))
        shape = image1.size()[-2:]
        prediction, features = model(data, start, stop, shape, raw=is_raw, intermediate=True)
        loss, terms = combined_loss(evaluator, prediction, image1, image2, features, weights=weights)
        smoothness, photometric, out_reg = terms
        photo_sum = add_loss(photo_sum, photometric)
        smooth_sum = add_loss(smooth_sum, smoothness)
        out_reg_sum = add_loss(out_reg_sum, out_reg)
        loss_sum += loss.item()
    logger.add_scalar('Validation/loss', loss_sum/n, global_step)
    for i, (s, p, o) in enumerate(zip(smooth_sum, photo_sum, out_reg_sum)):
        logger.add_scalar(f'Validation/smoothness loss/{i}', s/n, global_step)
        logger.add_scalar(f'Validation/photometric loss/{i}', p/n, global_step)
        logger.add_scalar(f'Validation/out regularization loss/{i}', o/n, global_step)
