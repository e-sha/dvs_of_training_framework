from argparse import ArgumentParser
from imageio import imwrite
import numpy as np
from PIL import ImageDraw, Image
import sys
import torch
from tqdm import tqdm

from EV_FlowNet.test import vis_flow
from utils.common import mean
from utils.dataloader import choose_data_path, get_dataloader
from utils.dataloader import get_valset_params
from utils.loss import init_losses
from utils.model import init_model
from utils.options import add_train_arguments
from utils.options import validate_train_args
from utils.timer import FakeTimer
from utils.training import process_minibatch


def parse_args(args):
    args = add_train_arguments(ArgumentParser()).parse_args(args)
    args = validate_train_args(args)
    args = choose_data_path(args)
    return args


def array2text(data, title):
    data = list(data)
    text = ', '.join([f'{x:.4f}' for x in data])
    return f'{title}: {mean(data):.4f} = [{text}]'


def join_images(images):
    images = images.detach().cpu().numpy().astype(np.uint8)
    assert images.ndim == 4
    images = np.transpose(images, axes=(0, 2, 3, 1))
    images = np.hstack(images)
    if images.shape[-1] == 1:
        images = np.tile(images, (1, 1, 3))
    return images


def get_events_text(args, batch):
    element_index = batch['events']['element_index']
    num_prefix_events = (element_index < args.prefix_length).sum()
    sequence_length = batch['augmentation_params']['sequence_length'].item()
    first_suffix_idx = sequence_length - args.suffix_length
    num_suffix_events = (element_index >= first_suffix_idx).sum()
    num_events = element_index.numel()
    num_prediction_events = num_events - num_prefix_events - num_suffix_events
    prefix_quantile = num_prefix_events * 100 / num_events
    suffix_quantile = num_suffix_events * 100 / num_events
    pred_quantile = num_prediction_events * 100 / num_events
    return f'{num_events} events: ' \
        f'{num_prefix_events} ({prefix_quantile:.2f}%) prefix ' \
        f'+ {num_prediction_events} ({pred_quantile:.2f}%) main + ' \
        f'{num_suffix_events} ({suffix_quantile:.2f}%)'


def prepare_text(args, batch, loss, parts, weights):
    parts = list(map(list, parts))
    loss_text = ' + '.join([f'{y}*{x:.4f}'
                            for x, y in zip(map(mean, parts), weights)])
    return f'loss: {loss:.4f} = {loss_text}\n' + '\n'.join(map(
        array2text, parts, ['smoothness', 'photometric', 'border'])) + \
        '\n' + get_events_text(args, batch)


def put_image(dst, src, x0, y0):
    H, W = src.shape[:2]
    dst[y0: y0 + H, x0: x0 + W] = src


def visualize_prediction(prediction):
    flows = tuple(map(lambda x: np.transpose(x.detach().cpu().numpy(),
                                             (1, 2, 0)),
                      prediction))
    images = tuple(map(vis_flow, flows))
    H, W = images[-1].shape[:2]
    if len(images) > 1:
        H += images[-2].shape[0]
    D = images[-1].shape[2]
    res = np.zeros((H, W, D), dtype=np.uint8)
    put_image(res, images[-1], 0, 0)
    x0 = 0
    y0 = images[-1].shape[0]
    for img in images[-2::-1]:
        put_image(res, img, x0, y0)
        x0 += img.shape[1]
    return res


def visualize_predictions(args, batch, predictions):
    num_predictions = predictions['prediction'][-1].shape[0]
    predictions = [[x[i] for x in predictions['prediction']]
                   for i in range(num_predictions)]
    images = tuple(map(visualize_prediction, predictions))
    image = np.concatenate(images, axis=1)
    image_h, image_w = images[0].shape[:2]
    sequence_length = batch['augmentation_params']['sequence_length'].item()
    res = np.zeros((image_h, image_w * (sequence_length + 1), 3),
                   dtype=np.uint8)
    x_shift = args.prefix_length * image_w + image_w // 2
    put_image(res, image, x_shift, 0)
    return res


def visualize(args, batch, loss, parts, weights, prediction):
    joined_images = join_images(batch['images'])

    res = np.zeros([80, joined_images.shape[1], 3], dtype=np.uint8)
    text = prepare_text(args, batch, loss, parts, weights)
    image = Image.fromarray(res)
    ImageDraw.Draw(image).text((0, 0), text, (255, 255, 255))
    image = np.asarray(image)
    flow_image = visualize_predictions(args, batch, prediction)
    return np.concatenate([image, joined_images, flow_image], axis=0)


def main():
    args = parse_args(sys.argv[1:])
    args.mbs = 1
    model = init_model(args, args.device)
    loader = get_dataloader(get_valset_params(args))
    evaluator = init_losses(
        args.shape, 1, model, args.device,
        sequence_length=args.prefix_length + args.suffix_length + 1)
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            loss, parts, tags, prediction = process_minibatch(
                model, batch, FakeTimer(), args.device, args.is_raw,
                evaluator, args.loss_weights, return_prediction=True)
            visualization = visualize(args, batch, loss, parts,
                                      args.loss_weights, prediction)
            imwrite(f'/tmp/{i:04d}.jpg', visualization)


if __name__ == '__main__':
    main()
