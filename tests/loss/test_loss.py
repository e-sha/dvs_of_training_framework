from tests.utils import read_test_elem
import torch


from utils.loss import Loss


def test_no_changes():
    B, H, W = 1, 5, 6
    dtype = torch.float32
    images = torch.zeros((2 * B, 1, H, W), dtype=dtype)
    timestamps = torch.tensor([[0, 0], [0.4, 0]], dtype=dtype)
    flow = torch.zeros((B, 2, H, W), dtype=dtype)
    flow_arth = torch.zeros((B, 2, H, W), dtype=dtype)
    evaluator = Loss((H, W), 1, 'cpu')
    loss = evaluator(images, timestamps, flow, flow_arth)
    assert len(loss) == 3
    for i, (l, gt) in enumerate(zip(loss, [0.002, 0.002, 0])):
        assert (l - gt).abs() < 5e-6, i


def test_zero_flow():
    x0, y0 = 230, 200
    B, H, W = 1, 5, 6
    dtype = torch.float32
    events, start, stop, image1, image2 = read_test_elem(1, box=[y0, x0, H, W],
                                                         is_torch=True)
    images = torch.cat([image1[None, None], image2[None, None]],
                       axis=0).to(torch.float32)
    timestamps = torch.tensor([[0, 0], [stop - start, 0]], dtype=dtype)
    flow = torch.zeros((B, 2, H, W), dtype=dtype)
    flow_arth = torch.zeros((B, 2, H, W), dtype=dtype)
    evaluator = Loss((H, W), 1, 'cpu')
    loss = evaluator(images, timestamps, flow, flow_arth)
    assert len(loss) == 3
    for i, (l, gt) in enumerate(zip(loss, [0.002, 0.878210, 0])):
        assert (l - gt).abs() < 5e-6, f'[{i}] {l} vs {gt}'


def test_pred_flow():
    x0, y0 = 230, 200
    B, H, W = 1, 5, 6
    dtype = torch.float32
    events, start, stop, image1, image2, flow = \
            read_test_elem(1, box=[y0, x0, H, W],
                           is_torch=True, read_pred=True)
    images = torch.cat([image1[None, None], image2[None, None]],
                       axis=0).to(torch.float32)
    timestamps = torch.tensor([[0, 0], [stop - start, 0]], dtype=dtype)
    flow = flow.permute(2, 0, 1)[None]
    flow_arth = torch.zeros((B, 2, H, W), dtype=dtype)
    evaluator = Loss((H, W), 1, 'cpu')
    loss = evaluator(images, timestamps, flow, flow_arth)
    assert len(loss) == 3
    for i, (l, gt) in enumerate(zip(loss, [0.004497, 0.928744, 0.009104])):
        assert (l - gt).abs() < 5e-6, f'[{i}] {l} vs {gt}'
