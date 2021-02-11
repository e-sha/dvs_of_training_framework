import numpy as np
from .transformation import map as event_map


def central_shift(in_shape, out_shape):
    ''' Returns top left corner of central bounding box in the image

    Args:
        in_shift (tuple): shape of the original image
        out_shape (tuple): shape of the resulting image
    '''
    return tuple(map(lambda x, y: (x-y)//2, in_shape, out_shape))


class EventCrop:
    ''' Leaves only events in the specified bounding box
    and modifies pixel coordinates
    '''
    def __init__(self, box):
        self.box = box

    def __call__(self, events, box=None):
        if box is None:
            box = self.box
        x, y, t, p = events.T
        mask = np.logical_and(
                np.logical_and(x >= box[1], x < box[1] + box[3]),
                np.logical_and(y >= box[0], y < box[0] + box[2])
                )
        return np.array([x[mask] - box[1],
                         y[mask] - box[0],
                         t[mask],
                         p[mask]]).T


class ImageCrop:
    ''' Crop images
    '''
    def __init__(self, box):
        self.box = box

    def __call__(self, flow):
        return flow[self.box[0]:self.box[0] + self.box[2],
                    self.box[1]:self.box[1] + self.box[3]]


class ImageCentralCrop:
    ''' Crop images
    '''
    def __init__(self, shape, return_box):
        self.shape = shape
        self.return_box = return_box

    def __call__(self, img, box=None):
        if box is None:
            start = list(central_shift(img.shape[:2], self.shape))
            box = start + list(self.shape)
        res = img[box[0]:box[0] + box[2],
                  box[1]:box[1] + box[3]]
        if self.return_box:
            return res, box
        return res


class ImageRandomCrop:
    ''' Crop images
    '''
    def __init__(self, shape, return_box):
        self.shape = shape
        self.return_box = return_box

    def __randint(self, x):
        if x == 0:
            return 0
        return np.random.randint(x)

    def __call__(self, img, box=None):
        if box is None:
            start = list(map(lambda x, y: self.__randint(x - y),
                             img.shape[:2], self.shape))
            box = start + list(self.shape)
        res = img[box[0]:box[0] + box[2],
                  box[1]:box[1] + box[3]]
        if self.return_box:
            return res, box
        return res


def get_count_image(events, imsize):
    ''' Counts number of events in each pixel

    Args:
        events (list): a list of sorted events in form of [x, y, t, p],
                       where each component is an array
        imsize (typle): height and width of the image

    Return:
        (np.array): an image that counts number of events in each pixel
    '''

    x, y = [np.array(v).astype(int) for v in events[:2]]
    idx = np.ravel_multi_index([y, x], imsize)
    res = np.zeros(imsize, dtype=np.uint64).ravel()
    np.add.at(res, idx, np.ones(idx.size))
    return res.reshape(imsize)


def frame_generator(events, frames):
    ''' Generate events cresponding to frames

    Args:
        events (list): a list of sorted events in form of [x, y, t, p],
                       where each component is an array
        frames (list): a list of timestamp pairs that generate frame
                       in form of [(start_ts, stop_ts)]
    '''
    frames = np.array(frames)
    t = events[2]
    idx = np.searchsorted(t, frames.ravel(), side='right').reshape(-1, 2)
    for (start, stop), (i_start, i_stop) in zip(frames, idx):
        yield [p[i_start:i_stop] for p in events], start, stop


def RandomRotation(interval, shape):
    x, y = np.meshgrid(range(shape[1]), range(shape[0]))
    x, y = map(lambda x: x.ravel(), (x, y))
    idx = np.ravel_multi_index([y, x], shape)
    assert np.all(idx == np.arange(np.prod(shape))), 'Sanity check'

    x, y = map(lambda x, s: x.astype(float) - s, (x, y),
               (shape[1] / 2, shape[0] / 2))
    multi_idx = np.vstack((x[None], y[None]))

    if not hasattr(interval, '__len__'):
        interval = abs(interval)
        interval = (-interval, interval)
    assert len(interval) == 2, "You should specify at most two anges"
    assert interval[0] <= interval[1], "The first interval should " \
                                       "be lower or equal to the second"

    def grad2rad(x):
        return x * np.pi / 180

    def rotation(image1, image2, events, angle=None):
        if angle is None:
            angle = (np.random.rand() * (interval[1] - interval[0]) +
                     interval[0])
        rad_angle = grad2rad(angle)
        mat = np.array([
            [np.cos(rad_angle), -np.sin(rad_angle)],
            [np.sin(rad_angle), np.cos(rad_angle)]
            ])
        idx1 = mat.dot(multi_idx)
        x1 = np.rint(idx1[0] + shape[1]/2)
        y1 = np.rint(idx1[1] + shape[0]/2)
        x1, y1 = map(lambda x: x.astype(int), (x1, y1))

        mask = np.logical_and(
                np.logical_and(x1 >= 0, x1 < shape[1]),
                np.logical_and(y1 >= 0, y1 < shape[0])
                )

        cur_idx = idx[mask]
        cur_ridx = np.ravel_multi_index([y1[mask], x1[mask]], shape)

        # rotate image image[y, x] = image[y1, x1]
        rimage1 = np.zeros_like(image1).ravel()
        rimage1[cur_idx] = image1.ravel()[cur_ridx]
        rimage2 = np.zeros_like(image2).ravel()
        rimage2[cur_idx] = image2.ravel()[cur_ridx]
        rimage1, rimage2 = map(lambda x: x.reshape(image1.shape),
                               (rimage1, rimage2))

        # rotate events
        revents = event_map(events.astype(np.float32).copy(),
                            image1.shape,
                            cur_ridx.astype(np.uint64),
                            cur_idx.astype(np.uint64))

        return rimage1, rimage2, revents, angle

    return rotation
