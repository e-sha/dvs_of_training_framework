from tests.utils import test_path
import torch
from types import SimpleNamespace


from train_flownet import init_model, init_losses
from utils.dataset import DatasetImpl, collate_wrapper
from utils.training import combined_loss


def test_backward():
    data_path = test_path/'data/seq'
    shape = [256, 256]
    batch_size = 2
    dataset = DatasetImpl(path=data_path,
                          shape=shape,
                          augmentation=False,
                          collapse_length=1,
                          is_raw=True,
                          return_aug=True,
                          max_seq_length=1)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False)
    events, timestamps, sample_idx, images, augmentation_params = \
            next(iter(data_loader))
    model = init_model(
            SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                            mish=False, sp=None),
            device='cpu')
    evaluator = init_losses(shape, batch_size, model, device='cpu')
    prediction, flow_ts, sample_idx, features = model(events,
                                                      timestamps,
                                                      sample_idx,
                                                      shape,
                                                      raw=True,
                                                      intermediate=True)
    loss, terms = combined_loss(evaluator,
                                prediction,
                                flow_ts,
                                sample_idx,
                                images,
                                timestamps,
                                features)
