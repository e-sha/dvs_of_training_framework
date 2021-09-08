from argparse import ArgumentParser
from imageio import imwrite
import numpy as np
from PIL import ImageDraw, Image
import sys
import torch
from tqdm import tqdm

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
            loss, (smoothness, photometric, out_reg), tags = process_minibatch(
                model, batch, FakeTimer(), args.device, args.is_raw,
                evaluator, args.loss_weights)
            res = np.zeros([480, 640, 3], dtype=np.uint8)
            text = f'{sum(smoothness)} {sum(photometric)} {sum(out_reg)}'
            image = Image.fromarray(res)
            ImageDraw.Draw(image).text((0, 0), text, (255, 255, 255))
            res = np.asarray(image)
            imwrite(f'/tmp/{i:04d}.jpg', res)


if __name__ == '__main__':
    main()
