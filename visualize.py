from argparse import ArgumentParser
from imageio import imwrite
import numpy as np
from PIL import ImageDraw, Image
import sys
import torch
from tqdm import tqdm

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


def prepare_text(loss, parts, weights):
    parts = list(map(list, parts))
    loss_text = ' + '.join([f'{y}*{x:.4f}'
                            for x, y in zip(map(mean, parts), weights)])
    return f'loss: {loss:.4f} = {loss_text}\n' + '\n'.join(map(
        array2text, parts, ['smoothness', 'photometric', 'border']))


def visualize(images, loss, parts, weights):
    joined_images = join_images(images)

    res = np.zeros([60, joined_images.shape[1], 3], dtype=np.uint8)
    text = prepare_text(loss, parts, weights)
    image = Image.fromarray(res)
    ImageDraw.Draw(image).text((0, 0), text, (255, 255, 255))
    image = np.asarray(image)
    return np.concatenate([image, joined_images], axis=0)


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
            loss, parts, tags = process_minibatch(
                model, batch, FakeTimer(), args.device, args.is_raw,
                evaluator, args.loss_weights)
            visualization = visualize(batch['images'], loss, parts,
                                      args.loss_weights)
            imwrite(f'/tmp/{i:04d}.jpg', visualization)


if __name__ == '__main__':
    main()
