import h5py
from pathlib import Path
import numpy as np
import random
import torch
import torch.utils.data

from EV_FlowNet.net import compute_event_image
from .data import EventCrop, ImageRandomCrop
from .data import RandomRotation, ImageCentralCrop


def read_info(filename):
    with h5py.File(filename, 'r') as f:
        sets = list(map(lambda x: x.decode(), f['set_name']))
        start_times = list(f['start_time'])
    return dict(zip(sets, start_times))


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self._shuffle = kwargs.pop('shuffle', False)
        self._dataset = DatasetImpl(**kwargs)

    def __iter__(self):
        def iterate(dataset, start, end, shuffle, **kwargs):
            shuffle_fun = random.shuffle if shuffle else lambda x: None
            order = list(range(start, end))
            shuffle_fun(order)
            i = 0
            while True:
                yield dataset[order[i]]
                i += 1
                if i == len(order):
                    i = 0
                    shuffle_fun(order)

        iter_start = 0
        iter_end = len(self._dataset)
        if False:  # if worker_info is not None:
            # split workload
            worker_info = torch.utils.data.get_worker_info()
            num_elems = iter_end - iter_start
            per_worker = num_elems // worker_info.num_workers
            residual = num_elems % worker_info.num_workers
            worker_id = worker_info.id
            iter_start = ((per_worker + 1) * min(worker_id, residual) +
                          per_worker * max(worker_id - residual, 0))
            iter_end = (iter_start + per_worker +
                        (1 if worker_id >= residual else 0))
            assert iter_start < len(self._dataset)
            assert iter_end <= len(self._dataset)
        return iterate(self._dataset, iter_start, iter_end, self._shuffle)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self._dataset = DatasetImpl(**kwargs)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


class DatasetImpl:
    def __init__(self,
                 path,  # path to the dataset
                 shape,  # shape of the images to load
                 augmentation=False,  # does apply augmentaion
                 collapse_length=6,  # maximum index distance between images
                                     # used for a single Optical Flow
                                     # predictions
                 max_seq_length=1, # maximum number of expected OF
                                   # predictions per sample
                 is_static_seq_length = True,
                 return_aug=False,  # does return augmentation parameters
                 is_raw=True,  # does return raw events or event images
                 is_align=True,  # shift timestamps to get a 0 start timestamp
                 angle=30):  # maximum angle for rotation
        self.path = Path(path)
        self.files = sorted(list(self.path.glob('*.hdf5')),
                            key=lambda x: int(x.stem))
        assert len(self.files) > 0, f"No hdf5 files found in {self.path}"
        self.augmentation = augmentation
        self.shape = shape
        self.collapse_length = collapse_length
        self.max_seq_length = max_seq_length
        self.is_static_seq_length = is_static_seq_length
        self.return_aug = return_aug
        self.is_raw = is_raw
        self.is_align = is_align
        self.angle = angle
        self.random_rotation = None  # RandomRotation(30, shape)

        self.event_crop_fun = EventCrop(box=None)
        kwargs = dict(shape=shape, return_box=True, channel_first=True)
        if self.augmentation:
            self.img_crop_fun = ImageRandomCrop(**kwargs)
        else:
            self.img_crop_fun = ImageCentralCrop(**kwargs)

    def __len__(self):
        n = len(self.files)
        if self.is_static_seq_length:
            return n - self.max_seq_length + 1
        return n

    def _get_k_elems(self, idx, k):
        events = []
        stop = -1
        for i in range(k):
            with h5py.File(self.files[idx+i], 'r') as f:
                events.append(np.array(f['events']))
                if i == 0:
                    image1 = np.array(f['image1'])
                    start = float(f['start'][()])
                else:
                    assert stop == float(f['start'][()])
                image2 = np.array(f['image2'])
                stop = float(f['stop'][()])
        events = np.vstack(events)
        # convert float to list with one element
        start, stop = map(lambda x: [x], (start, stop))

        return events, start, stop, image1, image2

    def _rotate(self, images, events, angle):
        # we have to know image shape to initialize random_rotation
        if self.random_rotation is None:
            self.random_rotation = RandomRotation(self.angle, images.shape[-2:])
        return self.random_rotation(images, events, angle)

    def __getitem__(self,
                    idx,
                    k=None,
                    is_flip=None,
                    angle=None,
                    box=None,
                    seq_length=None):
        ''' returns events and corresponding images
        Args:
            idx (int): index

        Return:
            events (ndarray of floats): event image
            start (float): first timestamps of the sliding window
            stop (float): last timestamp of the sliding window
            image1 (ndarray of floats): the image immediately
                                        before the sliding window
            image2 (ndarray of floats): the image immediately
                                        after the sliding window
        '''

        if seq_length is None:
            if self.augmentation:
                if self.is_static_seq_length:
                    seq_length = self.max_seq_length
                else:
                    choices = min(len(self.files) - idx, self.max_seq_length)
                    seq_length = np.random.randint(choices) + 1
            else:
                seq_length = self.max_seq_length

        # choose collapse length from 1 to self.collapse_length
        if k is None:
            if self.augmentation:
                # collapse events for several images
                max_k = (len(self.files) - idx) // seq_length
                choices = min(self.collapse_length, max_k)
                k = np.random.randint(choices) + 1
            else:
                k = 1

        assert idx + k * seq_length <= len(self.files)

        # read data
        events = None
        image_ts = None
        images = None
        for i in range(seq_length):
            _events, _start, _stop, _image1, _image2 = self._get_k_elems(idx + i * k, k)
            assert _image1.ndim == _image2.ndim
            assert all([x == y for x, y in zip(_image1.shape, _image2.shape)])
            if _image1.ndim == 2:
                _image1 = _image1[None]
                _image2 = _image2[None]
            else:
                assert _image1.ndim == 3
                _image1 = np.rollaxis(_image1, 2, 0)
                _image2 = np.rollaxis(_image2, 2, 0)
            _events = add_sample_index(_events, i)
            if events is None:
                events = [_events]
                image_ts = [_start, _stop]
                images = [_image1, _image2]
            else:
                events.append(_events)
                image_ts.append(_stop)
                images.append(_image2)
        events = np.vstack(events)
        image_ts = np.array(image_ts)
        images = np.concatenate(images, axis=0)

        # align timestamps
        # It fixes a problem with to huge timestamps stored as fp32
        if self.is_align:  # subtract start timestamp
            start_ts = image_ts[0]
            events[:, 2] -= start_ts
            image_ts -= start_ts

        # convert events to float32
        events = events.astype(np.float32)

        if self.augmentation:
            is_flip = np.random.rand() < 0.5 if is_flip is None else is_flip
            if is_flip:
                # horizontal flip
                images = images[..., ::-1]
                events[:, 0] = images.shape[-1] - events[:, 0] - 1
            # rotate image
            images, events, angle = self._rotate(images,
                                                 events,
                                                 angle)
        else:
            is_flip = False

        # crop. The input box is None if it isn't specified
        images, box = self.img_crop_fun(images, box=box)
        events = self.event_crop_fun(events, box=box)

        # convert images to float32 with channels as a first dimension
        images = images.astype(np.float32)
        assert all(events[:, 2] >= image_ts[0])
        assert all(events[:, 2] <= image_ts[-1])

        if self.is_raw:
            samples = events
        else:
            with torch.no_grad():
                samples = compute_event_image(events,
                                              image_ts[:-1],
                                              image_ts[1:],
                                              self.shape,
                                              device='cpu',
                                              dtype=torch.float32)[0]

        if self.return_aug:
            box = np.array(box, dtype=int)
            is_flip = np.array([is_flip], dtype=bool)
            return (samples,
                    image_ts,
                    images,
                    (idx, seq_length, k, box, angle, is_flip))
        else:
            return (samples,
                    image_ts,
                    images)  # add channel dimension


def add_sample_index(events, i):
    return np.hstack((events, np.full_like(events[:, [0]], i)))


def collate_wrapper(batch):
    def to_tensor(x):
        return torch.FloatTensor(x)

    #     0          1        2            3
    # (events, timestamps, images, augmentation_params)
    # add sample index to events
    events = np.vstack([add_sample_index(sample[0], i)
                        for i, sample in enumerate(batch)])
    timestamps = np.vstack([add_sample_index(sample[1], i)
                            for i, sample in enumerate])
    images = np.vstack([x[2] for x in batch])
    add_info = tuple()
    if len(batch) > 0 and len(batch[0]) > 3:
        # process augmentation parameters
        augmentation_params = [x[3] for x in batch]
        #   0        1       2    3     4       5
        # (idx, seq_length,  k,  box, angle, is_flip)
        idx = np.array([x[0] for x in augmentation_params])
        seq_length = np.array([x[1] for x in augmentation_params])
        k = np.array([x[2] for x in augmentation_params])
        box = np.vstack([x[3].reshape(1, -1) for x in augmentation_params])
        angle = np.array([x[4] for x in augmentation_params])
        is_flip = np.array([x[5] for x in augmentation_params])
        add_info = (tuple(map(to_tensor, (idx, k, box, angle, is_flip))), )

    return tuple(map(to_tensor, (events,
                                 start,
                                 stop,
                                 image1,
                                 image2))) + add_info
