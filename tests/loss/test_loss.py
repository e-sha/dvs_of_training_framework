from tests.utils import test_path
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
