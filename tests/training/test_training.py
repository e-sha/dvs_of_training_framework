from tests.utils import test_path
import torch
import tempfile
from types import SimpleNamespace


from utils.dataset import DatasetImpl, collate_wrapper
from utils.loss import Losses
from utils.timer import FakeTimer
from utils.training import train, validate
from train_flownet import init_model, construct_train_tools


def test_trainloop():
    args = SimpleNamespace(wdw=0.01,
                           training_steps=1,
                           rs=0,
                           optimizer='ADAM',
                           lr=0.01,
                           half_life=1,
                           device=torch.device('cpu'))
    data_path = test_path/'data/seq'
    shape = [256, 256]
    dataset = DatasetImpl(path=data_path,
                          shape=shape,
                          augmentation=False,
                          collapse_length=1,
                          is_raw=True,
                          max_seq_length=1)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=2,
                                              pin_memory=True,
                                              shuffle=False)
    model = init_model(
            SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                            mish=False, sp=None, prefix_length=0,
                            suffix_length=0, max_sequence_length=1,
                            dynamic_sample_length=False),
            device=args.device)
    optimizer, scheduler = construct_train_tools(args, model)
    evaluator = Losses([tuple(map(lambda x: x // 2 ** i, shape))
                        for i in range(4)][::-1], 2, args.device)
    with tempfile.TemporaryDirectory() as td:
        logger = torch.utils.tensorboard.SummaryWriter(log_dir=td)
        train(model=model, device=args.device, loader=data_loader,
              optimizer=optimizer, num_steps=args.training_steps,
              scheduler=scheduler, logger=logger, evaluator=evaluator,
              timers=FakeTimer())
        del logger


def test_validation():
    args = SimpleNamespace(wdw=0.01,
                           training_steps=1,
                           rs=0,
                           optimizer='ADAM',
                           lr=0.01,
                           half_life=1,
                           device=torch.device('cpu'))
    data_path = test_path/'data/seq'
    shape = [256, 256]
    dataset = DatasetImpl(path=data_path,
                          shape=shape,
                          augmentation=False,
                          collapse_length=1,
                          is_raw=True,
                          max_seq_length=1)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=2,
                                              pin_memory=True,
                                              shuffle=False)
    model = init_model(
            SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                            mish=False, sp=None, prefix_length=0,
                            suffix_length=0, max_sequence_length=1,
                            dynamic_sample_length=False),
            device=args.device)
    optimizer, scheduler = construct_train_tools(args, model)
    evaluator = Losses([tuple(map(lambda x: x // 2 ** i, shape))
                        for i in range(4)][::-1], 2, args.device)
    with tempfile.TemporaryDirectory() as td:
        logger = torch.utils.tensorboard.SummaryWriter(log_dir=td)
        validate(model=model, device=args.device, loader=data_loader,
                 samples_passed=0, logger=logger, evaluator=evaluator)
        del logger
