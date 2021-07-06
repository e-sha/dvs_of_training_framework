from tests.utils import read_test_elem
import torch


from utils.loss import Losses


def test_no_changes():
    B, H, W = 1, 5, 6
    dtype = torch.float32
    images = torch.zeros((2 * B, 1, H, W), dtype=dtype)
    timestamps = torch.tensor([0, 0.4], dtype=dtype)
    sample_idx = torch.LongTensor([0, 0])
    flow_sample_idx = torch.LongTensor([0])
    flow = torch.zeros((B, 2, H, W), dtype=dtype)
    evaluator = Losses([(H, W)], 1, 'cpu')
    loss = evaluator([flow], timestamps.view(1, 2), flow_sample_idx,
                     images, timestamps, sample_idx)
    assert len(loss) == 3
    for i, (l, gt) in enumerate(zip(loss, [0.002, 0.002, 0])):
        assert len(l) == 1
        assert (l[0] - gt).abs() < 5e-6, i


def test_zero_flow():
    #x0, y0 = 230, 200
    #B, H, W = 1, 5, 6
    x0, y0 = 0, 0
    B, H, W = 1, 246, 340
    dtype = torch.float32
    events, start, stop, image1, image2 = read_test_elem(1, box=[y0, x0, H, W],
                                                         is_torch=True)
    images = torch.cat([image1[None, None], image2[None, None]],
                       axis=0).to(torch.float32)
    timestamps = torch.tensor([0, stop - start], dtype=dtype)
    sample_idx = torch.LongTensor([0, 0])
    flow_sample_idx = torch.LongTensor([0])
    flow = torch.zeros((B, 2, H, W), dtype=dtype)
    evaluator = Losses([(H, W)], 1, 'cpu')
    loss = evaluator([flow], timestamps.view(1, 2), flow_sample_idx,
                     images, timestamps, sample_idx)
    assert len(loss) == 3
    for i, (l, gt) in enumerate(zip(loss, [0.002, 0.622660, 0])):
        assert len(l) == 1
        assert (l[0] - gt).abs() < 5e-6, f'[{i}] {l} vs {gt}'


def test_pred_flow():
    #x0, y0 = 230, 200
    #B, H, W = 1, 5, 6
    x0, y0 = 0, 0
    B, H, W = 1, 246, 340
    dtype = torch.float32
    events, start, stop, image1, image2, flow = \
            read_test_elem(1, box=[y0, x0, H, W],
                           is_torch=True, read_pred=True)
    images = torch.cat([image1[None, None], image2[None, None]],
                       axis=0).to(torch.float32)
    timestamps = torch.tensor([0, stop - start], dtype=dtype)
    sample_idx = torch.LongTensor([0, 0])
    flow_sample_idx = torch.LongTensor([0])
    flow = flow.permute(2, 0, 1)[None]
    evaluator = Losses([(H, W)], 1, 'cpu')
    loss = evaluator([flow], timestamps.view(1, 2), flow_sample_idx,
                     images, timestamps, sample_idx)
    assert len(loss) == 3
    for i, (l, gt) in enumerate(zip(loss, [0.002120, 0.652659, 0.007802])):
        assert len(l) == 1
        assert (l[0] - gt).abs() < 5e-6, f'[{i}] {l} vs {gt}'
