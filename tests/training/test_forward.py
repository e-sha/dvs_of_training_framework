import h5py
import numpy as np
from pathlib import Path
import sys
import torch
from types import SimpleNamespace


test_path = Path(__file__).parent.resolve()
while test_path.name != 'tests':
    test_path = test_path.parent
sys.path.append(str(test_path.parent))


try:
    from utils.dataset import Dataset, DatasetImpl, collate_wrapper
    from train_flownet import init_model
except ImportError:
    raise


def test_forward():
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
    events, timestamps, images, augmentation_params = next(iter(data_loader))
    model = init_model(
            SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                            mish=False, sp=None),
                            device='cpu')
    prediction, features = model(events,
                                 timestamps,
                                 shape,
                                 raw=True,
                                 intermediate=True)
