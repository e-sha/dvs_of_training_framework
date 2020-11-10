import torch
import torch.nn.functional as F
from functools import partial

torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version[0] > 1 or torch_version[0] == 1 and torch_version[1] > 2:
    grid_sample = partial(F.grid_sample, align_corners=True)
else:
    grid_sample = F.grid_sample

def interpolate(img, shape):
    return F.interpolate(img, size=shape, mode='bilinear', align_corners=True)

def charbonier_loss(delta, alpha: float=0.45, epsilon: float=1e-3):
    if delta.numel() == 0:
        return 0
    return (delta.pow(2) + epsilon*epsilon).pow(alpha).mean()

class Loss:
    def __init__(self, pred_shape, batch_size, device):
        self.N = batch_size
        self.H, self.W = pred_shape

        grid = torch.meshgrid(torch.arange(self.H, dtype=torch.float32, device=device),
                              torch.arange(self.W, dtype=torch.float32, device=device))
        # 2, H, W <-- (x, y)
        grid = torch.cat(tuple(map(lambda x: x.unsqueeze(0), grid[::-1])), dim=0)
        # N, 2, H, W
        self.grid = grid.unsqueeze(0).expand(self.N, -1, -1, -1)
        # used to store (self.grid + flow)
        self.grid_holder = torch.empty_like(self.grid)

    def warp_images_with_flow(self, images, warp_grid):
        N, C, H, W = images.size()
        assert self.N >= N, f'This object should be used for batch of at most {self.N} samples, but {N} samples are given'
        assert self.H == H, f'This object should be used for images of height {self.H}, but image of height {H} are given'
        assert self.W == W, f'This object should be used for images of height {self.W}, but image of height {W} are given'

        return grid_sample(images, warp_grid.permute(0, 2, 3, 1))

    def photometric_loss(self, prev_images, next_images, warp_grid):
        warped = self.warp_images_with_flow(next_images, warp_grid)
        return charbonier_loss(warped - prev_images)

    def smoothness_loss(self, flow):
        ucrop = flow[..., 1:, :]
        dcrop = flow[..., :-1, :]
        lcrop = flow[..., 1:]
        rcrop = flow[..., :-1]

        ulcrop = flow[..., 1:, 1:]
        drcrop = flow[..., :-1, :-1]
        dlcrop = flow[..., :-1, 1:]
        urcrop = flow[..., 1:, :-1]

        return (charbonier_loss(lcrop - rcrop) + charbonier_loss(ucrop - dcrop) +\
                charbonier_loss(ulcrop - drcrop) + charbonier_loss(dlcrop - urcrop)) / 4

    def outborder_mask(self, warp_grid):
        # x or y coordinate points out of border
        return ((warp_grid < -1) | (warp_grid > 1)).sum(1) > 0

    def outborder_regularization_loss(self, flow_arth, warp_grid):
        mask = self.outborder_mask(warp_grid).unsqueeze_(1).expand(-1, 2, -1, -1)
        N = mask.size()[0]
        losses = []
        for i in range(N):
            values = flow_arth[i][mask[i]]
            if values.numel() == 0:
                losses.append(0)
            losses.append(charbonier_loss(values))
        return sum(losses) / N

    def __call__(self, prev_images, next_images, flow, flow_arth):
        N, C, H, W = prev_images.size()
        assert self.N >= N, f'This object should be used for batch of at most {self.N} samples, but {N} samples are given'
        assert self.H == H, f'This object should be used for images of height {self.H}, but image of height {H} are given'
        assert self.W == W, f'This object should be used for images of height {self.W}, but image of height {W} are given'

        NN, NC, NH, NW = next_images.size()
        assert NN == N, f'Number of previous and next images should be the same {N} vs {NN}'
        assert NC == C, f'Number of channels of previous and next images should be the same {C} vs {NC}'
        assert NH == H, f'Height of previous and next images should be the same {H} vs {NH}'
        assert NW == W, f'Width of images and flows should be the same {W} vs {NW}'

        FN, FC, FH, FW = flow.size()
        assert FN == N, f'Number of images and flows should be the same {N} vs {FN}'
        assert FC == 2, f'Flow should contain 2 channels (dx and dy)'
        assert FH == H, f'Height of images and flows should be the same {H} vs {FH}'
        assert FW == W, f'Width of images and flows should be the same {W} vs {FW}'

        AN, AC, AH, AW = flow.size()
        assert AN == N, f'Number of images and flow\'s hyperbolic arctangenses should be the same {N} vs {AN}'
        assert AC == 2, f'Flow\'s hyperbolic arctangenses should contain 2 channels (dx and dy)'
        assert AH == H, f'Height of images and flow\'s hyperbolic arctangenses should be the same {H} vs {AH}'
        assert AW == W, f'Width of images and flow\'s hyperbolic arctangenses should be the same {W} vs {AW}'

        # compute grid to sample data (non-normalized)
        #torch.add(self.grid, flow, out=self.grid_holder)
        self.grid_holder = self.grid[:N] + flow
        # normalize [0, W-1] -> [0, 2]
        self.grid_holder[:, 0] /= (W-1)/2.
        # normalize [0, H-1] -> [0, 2]
        self.grid_holder[:, 1] /= (H-1)/2.
        # normalize [0, 2] -> [-1, 1]
        self.grid_holder -= 1
        photometric = self.photometric_loss(prev_images, next_images, self.grid_holder)
        smoothness = self.smoothness_loss(flow)
        outborder = self.outborder_regularization_loss(flow_arth, self.grid_holder)
        return smoothness, photometric, outborder

class Losses:
    def __init__(self, shapes, batch_size, device):
        self.losses = [Loss(shape, batch_size, device) for shape in shapes]

    def __call__(self, flows, prev_images, next_images, flow_arths):
        result = []
        for loss, flow, flow_arth in zip(self.losses, flows, flow_arths):
            cur_shape = flow.size()[-2:]
            im1, im2 = map(interpolate, (prev_images, next_images), [cur_shape]*2)
            result.append(loss(im1, im2, flow, flow_arth))
        return tuple(zip(*result))

def warp_images_with_flow(images, flow):
    N, C, H, W = images.size()
    assert flow.size()[0] == N, f'Number of images and flows should be the same {N} vs {flow.size()[0]}'
    assert flow.size()[1] == 2, f'Flow should contain 2 channels (dx and dy)'
    assert flow.size()[2] == H, f'Height of images and flows should be the same {H} vs {flow.size()[2]}'
    assert flow.size()[3] == W, f'Width of images and flows should be the same {W} vs {flow.size()[3]}'
    # H, W
    device = images.device
    grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                          torch.arange(W, dtype=torch.float32, device=device))
    # H, W, 2 <-- (x, y)
    grid = torch.cat(tuple(map(lambda x: x.unsqueeze(2), grid[::-1])), dim=2)
    # N, H, W, 2
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
    # N, H, W, 2
    flow = flow.permute(0, 2, 3, 1)
    grid = grid + flow
    # normalize [0, W-1] -> [0, 2]
    grid[..., 0] /= (W-1)/2.
    # normalize [0, H-1] -> [0, 2]
    grid[..., 1] /= (H-1)/2.
    # normalize [0, 2] -> [-1, 1]
    grid -= 1
    return F.grid_sample(images, grid, align_corners=True)

def photometric_loss(prev_images, next_images, flow):
    total_photometric_loss = 0.
    looss_weight_sum = 0.

    warped = warp_images_with_flow(next_images, flow)
    return charbonier_loss(warped - prev_images)

def smoothness_loss(flow):
    ucrop = flow[..., 1:, :]
    dcrop = flow[..., :-1, :]
    lcrop = flow[..., 1:]
    rcrop = flow[..., :-1]

    ulcrop = flow[..., 1:, 1:]
    drcrop = flow[..., :-1, :-1]
    dlcrop = flow[..., :-1, 1:]
    urcrop = flow[..., 1:, :-1]

    return (charbonier_loss(lcrop - rcrop) + charbonier_loss(ucrop - dcrop) +\
            charbonier_loss(ulcrop - drcrop) + charbonier_loss(dlcrop - urcrop)) / 4

def get_int_and_remainder(x, is_floor):
    if is_floor:
        cx = x.floor()
        return cx, cx + 1 - x
    cx = x.ceil()
    return cx, x - (cx - 1)

def contrast_image(events, flow, ts, pol):
    mask = events[:, 3].long() == pol

    x = events[mask, 0]
    y = events[mask, 1]
    t = events[mask, 2]
    b = events[mask, 4]

    n = events.size()[0]

    device = flow.device

    bs, nc, height, width = flow.size()
    assert nc == 2

    res = torch.zeros([bs, height, width], dtype=torch.float32, device=device)
    if n == 0:
        return res
    acc = torch.full_like(res, eps)

    ui = x + width * (y + height * (0 + b * nc)).long()
    vi = x + width * (y + height * (1 + b * nc)).long()

    u = flow.ravel()[ui]
    v = flow.ravel()[vi]

    x1 = x + (ts - t) * u
    y1 = y + (ts - t) * v

    eps = 1e-6
    for i in range(2):
        cx, dx = get_int_and_remainder(x1, bool(i))
        for j in range(2):
            cy, dy = get_int_and_remainder(y1, bool(j))
            i = x + width * (y + b * height).long()
            res.put_(i, t, accumulate=True)
            acc.put_(i, np.ones(n), accumulate=True)
    return res / acc

def contrast_loss(events, flow, ts):
    loss = 0
    for t in [0, ts]:
        for p in [-1, 1]:
            img = contast_image(events, flow, t, p)
            loss += img.pow_(2)
    return loss
