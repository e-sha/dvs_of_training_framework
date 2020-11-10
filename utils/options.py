from argparse import ArgumentParser

import torch
import torch.nn as nn

from mish.mish import Mish

def train_parser():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', help='Directory to store learned weights',
            required=True, type=Path)
    parser.add_argument('-d', '--device', help='Device to run the script',
            default='cuda:0', required=False)
    parser.add_argument('--lr_gamma', help='Value used to decrease learning rate',
            default=0.9, type=float, required=False)
    parser.add_argument('-bs', '--batch_size', help='batch size',
            dest='bs', default=32, type=int, required=False)
    parser.add_argument('-sp', '--starting_point', help='initial weights for the network',
            dest='sp', default=None, required=False)
    parser.add_argument('-wdw', '--weight_decay_weight', help='weight of weight decay',
            dest='wdw', default=1e-4, type=float, required=False)
    parser.add_argument('-ne', '--num_epochs', help='number of epochs to train',
            dest='epochs', default=200, type=int, required=False)
    parser.add_argument('-st', '--step', help='period of time to decrease learning rate',
            dest='step', default=4, type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', help='initial learning rate',
            dest='lr', default=1e-3, type=float, required=False)
    parser.add_argument('-cl', '--collapse_length', help='step for data augmentation',
            dest='cl', default=6, type=int, required=False)
    parser.add_argument('--height', help='height of the training images',
            dest='height', default=256, type=int, required=False)
    parser.add_argument('--width', help='width of the trainging images',
            dest='width', default=256, type=int, required=False)
    parser.add_argument('-vp', '--validation_period', help='validation period',
            dest='vp', default=5, type=int, required=False)
    parser.add_argument('--ev_flownet', help='to train ev_flownet',
            action='store_true')
    parser.add_argument('--optimizer', help='Optimizer to use',
            default='RANGER', choices=['ADAM', 'RADAM', 'RANGER'])
    parser.add_argument('--loss_weights', help='weights of the term in the loss function',
            default=[0.5, 1, 1], nargs=3, type=float)
    parser.add_argument('--ev_images', help='use event images as the input to the network',
            action='store_true')
    parser.add_argument('--representation-start', help='first iteration to start the representation learning',
            dest='rs', default=0, type=int)
    parser.add_argument('--mish', help='use event images as the input to the network',
            action='store_true')
    parser.add_argument('--accum_step', help='Number of batches to process before to accumulate gradients',
            dest='accum_step', default=1, type=int)

    return parser

def options2model_kwargs(parameters):
    kargs = dict()
    if parameters.mish:
        kargs['activation'] = Mish()
    else:
        kargs['activation'] = nn.ReLU()
    return kargs
