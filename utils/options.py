from pathlib import Path
import torch
import torch.nn as nn

from mish.mish import Mish

def add_common_arguments(parser):
    parser.add_argument('--flownet_path', help='relative path to a model to train',
            default=Path('EV_FlowNet'), type=Path, required=False)
    parser.add_argument('--ev_images', help='use event images as the input to the network',
            action='store_true')
    parser.add_argument('--mish', help='use event images as the input to the network',
            action='store_true')
    parser.add_argument('-d', '--device', help='Device to run the script',
            default=torch.device('cuda:0'), type=torch.device, required=False)
    parser.add_argument('-bs', '--batch_size', help='batch size for an optimizer step',
            dest='bs', default=32, type=int, required=False)
    parser.add_argument('--profiling', help='start profiling', action='store_true')
    return parser

def add_test_arguments(parser):
    parser = add_common_arguments(parser)
    parser.add_argument('-m', '--model', help='Path to the learned weights',
            type=Path, required=True)
    parser.add_argument('-o', '--output', help='Path to write test results',
            type=Path, required=True)
    parser.add_argument('-s', '--step', help='step to test',
                        default=None, type=int, required=False)
    parser.add_argument('-ng', '--tests_per_gpu', help='Number of tests to launch per GPU',
                        default=2, type=int, required=False)
    return parser

def add_train_arguments(parser):
    parser = add_common_arguments(parser)
    parser.add_argument('-m', '--model', help='Directory to store learned weights',
            required=True, type=Path)
    parser.add_argument('--half_life', help='Half-life of a learning rate',
            dest='half_life', default=100000, type=float, required=False)
    parser.add_argument('-mbs', '--micro_batch_size', help='batch size for a single forward-backward pass',
            dest='mbs', default=32, type=int, required=False)
    parser.add_argument('-sp', '--starting_point', help='initial weights for the network',
            dest='sp', default=None, required=False)
    parser.add_argument('-wdw', '--weight_decay_weight', help='weight of weight decay',
            dest='wdw', default=1e-4, type=float, required=False)
    parser.add_argument('-ne', '--num_training_steps', help='number of steps to train',
            dest='training_steps', default=1000000, type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', help='initial learning rate',
            dest='lr', default=1e-3, type=float, required=False)
    parser.add_argument('-cl', '--collapse_length', help='step for data augmentation',
            dest='cl', default=6, type=int, required=False)
    parser.add_argument('--height', help='height of the training images',
            dest='height', default=256, type=int, required=False)
    parser.add_argument('--width', help='width of the trainging images',
            dest='width', default=256, type=int, required=False)
    parser.add_argument('-vp', '--validation_period', help='validation period',
            dest='vp', default=1000, type=int, required=False)
    parser.add_argument('--optimizer', help='Optimizer to use',
            default='RANGER', choices=['ADAM', 'RADAM', 'RANGER'])
    parser.add_argument('--loss_weights', help='weights of the term in the loss function',
            default=[0.5, 1, 1], nargs=3, type=float)
    parser.add_argument('--representation-start', help='proportion of training steps without the representation learning',
            dest='rs', default=0.5, type=float)
    parser.add_argument('--num_checkpoints', help='Number of last checkpoints to store',
            dest='num_checkpoints', default=2, type=int)
    parser.add_argument('--permanent_interval', help='Periodicity of making checkpoints that will not be removed',
            dest='permanent_interval', default=10000, type=int)
    parser.add_argument('--checkpointing_interval', help='Periodicity of making checkpoints',
            dest='checkpointing_interval', default=1000, type=int)
    parser.add_argument('--timers', help='Print information from timers',
            dest='timers', action='store_true')
    parser.add_argument('--num_workers', help='Number of workers to read data',
            dest='num_workers', default=32, type=int)
    parser.add_argument('--do_not_continue', help='Do not continue training from checkpoints',
            dest='do_not_continue', action='store_true')

    return parser

def validate_common_args(args):
    args.is_raw = not args.ev_images
    return args

def validate_train_args(args):
    args = validate_common_args(args)
    assert args.bs > 0
    assert args.mbs > 0
    assert args.bs % args.mbs == 0
    args.accum_step = args.bs // args.mbs
    assert args.permanent_interval % args.checkpointing_interval == 0
    return args

def validate_test_args(args):
    args = validate_common_args(args)
    return args

def options2model_kwargs(parameters):
    kargs = dict()
    if parameters.mish:
        kargs['activation'] = Mish()
    else:
        kargs['activation'] = nn.ReLU()
    return kargs
