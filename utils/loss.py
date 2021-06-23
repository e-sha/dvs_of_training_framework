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


def interpolate(img, shape):
    return F.interpolate(img, size=shape, mode='bilinear', align_corners=True)


def charbonier_loss(delta,
                    alpha: float = 0.45,
                    epsilon: float = 1e-3,
                    denominator: Optional[torch.Tensor] = None,
                    device: Optional[torch.device] = torch.device('cpu')):
    if delta.numel() == 0:
        return torch.zeros([1], dtype=torch.float32, device=device)
    delta = (delta.pow(2) + epsilon*epsilon).pow(alpha)
    if denominator is None:
        return delta.mean()
    assert denominator.numel() == delta.numel()
    return (delta / denominator).sum()


class Loss:
    def __init__(self, pred_shape, batch_size, device, timers=FakeTimer()):
        self.N = batch_size
        self.H, self.W = pred_shape

        grid = torch.meshgrid(torch.arange(self.H,
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

    def outborder_regularization_loss(self, flow_arth, warp_grid):
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
                                                  device=flow_arth.device),
                                     right=True)
            denominators = denominators[idx] * N

        values = flow_arth[mask]
        loss = charbonier_loss(values,
                               denominator=denominators,
                               device=flow_arth.device)
        return loss

    def __call__(self, images, timestamps, flow, flow_arth):
        num_images, C, H, W = images.size()
        batch_size = int(timestamps[-1, 1].item()) + 1
        N = num_images - batch_size
        assert self.N >= N, 'This object should be used for batch of ' \
                            f'at most {self.N} samples, but {N} samples ' \
                            'are given'
        assert self.H == H, 'This object should be used for images of ' \
                            f'height {self.H}, but image of height {H} ' \
                            'are given'
        assert self.W == W, 'This object should be used for images of ' \
                            f'height {self.W}, but image of height {W} ' \
                            'are given'

        FN, FC, FH, FW = flow.size()
        assert FN == N, 'Number of images and flows should ' \
                        f'be the same {N} vs {FN}'
        assert FC == 2, 'Flow should contain 2 channels (dx and dy)'
        assert FH == H, 'Height of images and flows should ' \
                        f'be the same {H} vs {FH}'
        assert FW == W, 'Width of images and flows should ' \
                        f'be the same {W} vs {FW}'

        AN, AC, AH, AW = flow_arth.size()
        assert AN == N, 'Number of images and flow\'s hyperbolic ' \
                        f'arctangenses should be the same {N} vs {AN}'
        assert AC == 2, 'Flow\'s hyperbolic arctangenses should ' \
                        'contain 2 channels (dx and dy)'
        assert AH == H, 'Height of images and flow\'s hyperbolic ' \
                        f'arctangenses should be the same {H} vs {AH}'
        assert AW == W, 'Width of images and flow\'s hyperbolic ' \
                        f'arctangenses should be the same {W} vs {AW}'

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

        prev_image_indices = torch.full_like(timestamps[:, 1],
                                             False, dtype=torch.bool)
        next_image_indices = torch.full_like(timestamps[:, 1],
                                             False, dtype=torch.bool)
        prev_image_indices[:-1] = next_image_indices[1:] = \
                timestamps[:-1, 1] == timestamps[1:, 1]

        photometric = self.photometric_loss(images[prev_image_indices],
                                            images[next_image_indices],
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

    def __call__(self, flows, images, timestamps, flow_arths):
        result = []
        for loss, flow, flow_arth in zip(self.losses, flows, flow_arths):
            cur_shape = flow.size()[-2:]
            images = interpolate(images, cur_shape)
            result.append(loss(images, timestamps, flow, flow_arth))
        return tuple(zip(*result))
