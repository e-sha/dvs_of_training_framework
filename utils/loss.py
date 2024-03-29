from functools import partial
import torch
import torch.nn.functional as F
from typing import Optional

from .timer import FakeTimer

torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version[0] > 1 or torch_version[0] == 1 and torch_version[1] > 2:
    grid_sample = partial(F.grid_sample, align_corners=True)
else:
    grid_sample = F.grid_sample

if torch_version[0] > 1 or torch_version[0] == 1 and torch_version[1] >= 10:
    meshgrid = partial(torch.meshgrid, indexing='ij')
else:
    meshgrid = torch.meshgrid


def interpolate(img, shape):
    return F.interpolate(img, size=shape, mode='bilinear', align_corners=True)


def charbonier_loss(delta,
                    alpha: float = 0.45,
                    epsilon: float = 1e-3,
                    denominator: Optional[torch.Tensor] = None,
                    device: Optional[torch.device] = torch.device('cpu')):
    if delta.numel() == 0:
        return torch.tensor(0, dtype=torch.float32, device=device)
    delta = (delta.pow(2) + epsilon*epsilon).pow(alpha)
    if denominator is None:
        return delta.mean()
    assert denominator.numel() == delta.numel()
    return (delta / denominator).sum()


class Loss:
    def __init__(self, pred_shape, batch_size, device, timers=FakeTimer()):
        self.N = batch_size
        self.H, self.W = pred_shape

        grid = meshgrid(torch.arange(self.H,
                                     dtype=torch.float32,
                                     device=device),
                        torch.arange(self.W,
                                     dtype=torch.float32,
                                     device=device))
        # 2, H, W <-- (x, y)
        grid = torch.cat(tuple(map(lambda x: x.unsqueeze(0),
                                   grid[::-1])), dim=0)
        # N, 2, H, W
        self.grid = grid.unsqueeze(0).expand(self.N, -1, -1, -1)
        # used to store (self.grid + flow)
        self.grid_holder = torch.empty_like(self.grid)
        self.timers = timers

    def warp_images_with_flow(self, images, warp_grid):
        N, C, H, W = images.size()
        assert self.N >= N, 'This object should be used for batch of at ' \
                            f'most {self.N} samples, but {N} samples ' \
                            'are given'
        assert self.H == H, 'This object should be used for images of ' \
                            f'height {self.H}, but image of height {H} ' \
                            'are given'
        assert self.W == W, 'This object should be used for images of ' \
                            f'height {self.W}, but image of height {W} ' \
                            'are given'

        return grid_sample(images, warp_grid.permute(0, 2, 3, 1))

    def photometric_loss(self, prev_images, next_images, warp_grid):
        warped = self.warp_images_with_flow(next_images, warp_grid)
        return charbonier_loss(warped - prev_images, device=next_images.device)

    def smoothness_loss(self, flow):
        ucrop = flow[..., 1:, :]
        dcrop = flow[..., :-1, :]
        lcrop = flow[..., 1:]
        rcrop = flow[..., :-1]

        ulcrop = flow[..., 1:, 1:]
        drcrop = flow[..., :-1, :-1]
        dlcrop = flow[..., :-1, 1:]
        urcrop = flow[..., 1:, :-1]

        return (charbonier_loss(lcrop - rcrop) +
                charbonier_loss(ucrop - dcrop) +
                charbonier_loss(ulcrop - drcrop) +
                charbonier_loss(dlcrop - urcrop)) / 4

    def outborder_mask(self, warp_grid):
        # x or y coordinate points out of border
        return ((warp_grid < -1) | (warp_grid > 1)).sum(1) > 0

    def outborder_regularization_loss(self, flow, warp_grid):
        N = warp_grid.size()[0]
        with torch.no_grad():
            mask = self.outborder_mask(warp_grid)
            # number of bad values in each sample [N_1, N_2, ..., N_N]
            denominators = mask.view(N, -1).sum(dim=1) * 2
            mask = mask.unsqueeze_(1).expand(-1, 2, -1, -1)

            # at which positions values of ith sample ends from
            stop = torch.cumsum(denominators, dim=0)
            # number of bad values in a whole batch
            num_points = denominators.sum()
            # indices of samples in a batch
            idx = torch.searchsorted(stop,
                                     torch.arange(num_points,
                                                  device=flow.device),
                                     right=True)
            denominators = denominators[idx] * N

        values = flow[mask]
        loss = charbonier_loss(values,
                               denominator=denominators,
                               device=flow.device)
        return loss

    def __call__(self, prev_images, next_images, flow):
        N, C, H, W = prev_images.size()
        assert self.N >= N, 'This object should be used for batch of ' \
                            f'at most {self.N} samples, but {N} samples ' \
                            'are given'
        assert self.H == H, 'This object should be used for images of ' \
                            f'height {self.H}, but image of height {H} ' \
                            'are given'
        assert self.W == W, 'This object should be used for images of ' \
                            f'height {self.W}, but image of height {W} ' \
                            'are given'

        NN, NC, NH, NW = next_images.size()
        assert NN == N
        assert NC == C
        assert NH == H
        assert NW == W

        FN, FC, FH, FW = flow.size()
        assert FN == N, 'Number of images and flows should ' \
                        f'be the same {N} vs {FN}'
        assert FC == 2, 'Flow should contain 2 channels (dx and dy)'
        assert FH == H, 'Height of images and flows should ' \
                        f'be the same {H} vs {FH}'
        assert FW == W, 'Width of images and flows should ' \
                        f'be the same {W} vs {FW}'

        self.timers('grid_construction').start()
        # compute grid to sample data (non-normalized)
        self.grid_holder = self.grid[:N] + flow
        # normalize [0, W-1] -> [0, 2]
        self.grid_holder[:, 0] /= (W-1)/2.
        # normalize [0, H-1] -> [0, 2]
        self.grid_holder[:, 1] /= (H-1)/2.
        # normalize [0, 2] -> [-1, 1]
        self.grid_holder -= 1
        self.timers('grid_construction').stop()
        self.timers('photometric_loss').start()

        photometric = self.photometric_loss(prev_images,
                                            next_images,
                                            self.grid_holder)
        self.timers('photometric_loss').stop()
        self.timers('smoothness_loss').start()
        smoothness = self.smoothness_loss(flow)
        self.timers('smoothness_loss').stop()
        self.timers('outborder_loss').start()
        outborder = self.outborder_regularization_loss(flow,
                                                       self.grid_holder)
        self.timers('outborder_loss').stop()
        return smoothness, photometric, outborder


class Losses:
    def __init__(self, shapes, batch_size, device, timers=FakeTimer()):
        self.losses = [Loss(shape, batch_size, device, timers)
                       for shape in shapes]

    def __call__(self, flows, flow_ts, flow_sample_idx, images, timestamps,
                 sample_idx):
        result = []
        with torch.no_grad():
            P = flow_sample_idx.shape[0]
            # [2, D, P]
            # D is a number of timestamps in the input data
            # P is a number of predictions
            sample_mask = sample_idx.view(1, -1, 1) == \
                flow_sample_idx.view(1, 1, -1)
            image_ts_mask = timestamps.view(1, -1, 1) == \
                flow_ts.T.view(2, 1, -1)
            image_mask = torch.logical_and(image_ts_mask, sample_mask)
            # for each prediction has to be exactly one image in data
            assert (image_mask.sum(1) == 1).all()
            # (iFoS, D, P)
            # iFoS is first or second image
            indices = torch.nonzero(image_mask, as_tuple=True)
            device = indices[2].device
            assert indices[0].numel() == 2 * P
            assert (indices[0][:P] == 0).all()
            assert (indices[0][P:] == 1).all()
            assert (indices[2][:P] == torch.arange(P,
                                                   device=device)).all()
            assert (indices[2][P:] == torch.arange(P,
                                                   device=device)).all()
            start_indices = indices[1][:P]
            stop_indices = indices[1][P:]
        for loss, flow in zip(self.losses, flows):
            cur_shape = flow.size()[-2:]
            with torch.no_grad():
                images = interpolate(images, cur_shape)
            result.append(loss(images[start_indices],
                               images[stop_indices],
                               flow))
        return tuple(zip(*result))


def init_losses(shape, batch_size, model, device, sequence_length,
                timers=FakeTimer()):
    events = {'x': torch.tensor([], dtype=torch.long, device=device),
              'y': torch.tensor([], dtype=torch.long, device=device),
              'timestamp': torch.tensor([], dtype=torch.float32,
                                        device=device),
              'polarity': torch.tensor([], dtype=torch.long,
                                       device=device),
              'element_index': torch.tensor([], dtype=torch.long,
                                            device=device),
              'sample_index': torch.tensor([], dtype=torch.long,
                                           device=device)}
    with torch.no_grad():
        num_timestamps = sequence_length + 1
        out = model(events,
                    torch.tensor([0.04 * i for i in range(num_timestamps)],
                                 dtype=torch.float32,
                                 device=device),
                    torch.tensor([0] * num_timestamps,
                                 dtype=torch.long,
                                 device=device),
                    shape, raw=True)
    out_shapes = tuple(tuple(flow.shape[2:]) for flow in out[0])
    return Losses(out_shapes, batch_size, device, timers=timers)
