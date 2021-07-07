from tests.utils import test_path
import torch
from types import SimpleNamespace


from utils.dataset import DatasetImpl, collate_wrapper
from utils.loss import Losses
from utils.timer import FakeTimer
from utils.training import train
from train_flownet import init_model, construct_train_tools


def test_dummy():
    data_path = test_path/'data/seq'
    shape = [256, 256]
    dataset = DatasetImpl(path=data_path,
                          shape=shape,
                          augmentation=False,
                          collapse_length=1,
                          is_raw=True,
                          return_aug=True,
                          max_seq_length=1)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=2,
                                              pin_memory=True,
                                              shuffle=False)
    events, timestamps, sample_idx, images, augmentation_params = next(iter(data_loader))
    model = init_model(
            SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                            mish=False, sp=None),
            device='cpu')
    prediction, timestamps, sample_idx, features = model(events,
                                                         timestamps,
                                                         sample_idx,
                                                         shape,
                                                         raw=True,
                                                         intermediate=True)


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
                          return_aug=False,
                          max_seq_length=1)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=2,
                                              pin_memory=True,
                                              shuffle=False)
    model = init_model(
            SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                            mish=False, sp=None),
            device=args.device)
    optimizer, scheduler = construct_train_tools(args, model)
    evaluator = Losses([shape], 2, args.device)
    train(model=model, device=args.device, loader=data_loader, optimizer=optimizer,
          num_steps=args.training_steps, scheduler=scheduler, logger=None,
          evaluator=evaluator, timers=FakeTimer())
