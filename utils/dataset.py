import h5py
from pathlib import Path
import numpy as np
import torch

from EV_FlowNet.net import compute_event_image
from .data import central_shift, EventCrop, ImageCrop, ImageRandomCrop, RandomRotation, ImageCentralCrop

def read_info(filename):
    with h5py.File(filename, 'r') as f:
        sets = list(map(lambda x: x.decode(), f['set_name']))
        start_times = list(f['start_time'])
    return dict(zip(sets, start_times))

class Dataset:
    def __init__(self,
            path, # path to the dataset
            shape, # shape of the images to load
            augmentation=False, # does apply augmentaion
            collapse_length=6, # maximum number of images to combine in a single sample
            return_aug=False, # does return augmentation parameters
            is_raw=True, # does return raw events or event images
            is_align = True, # shift timestamps to get a 0 start timestamp
            angle = 30 # maximum angle for rotation
            ):
        self.path = Path(path)
        self.files = sorted(list(self.path.glob('*.hdf5')), key=lambda x: int(x.stem))
        assert len(self.files) > 0, f"No hdf5 files found in {self.path}"
        self.augmentation = augmentation
        self.shape = shape
        self.collapse_length = collapse_length
        self.return_aug = return_aug
        self.is_raw = is_raw
        self.is_align = is_align
        self.angle = angle
        self.random_rotation = None#RandomRotation(30, shape)

        self.event_crop_fun = EventCrop(box=None)
        if self.augmentation:
            self.img_crop_fun = ImageRandomCrop(shape=shape, return_box=True)
        else:
            self.img_crop_fun = ImageCentralCrop(shape=shape, return_box=True)

    def __len__(self):
        return len(self.files)

    def _get_k_elems(self, idx, k):
        events = []
        for i in range(k):
            with h5py.File(self.files[idx+i], 'r') as f:
                events.append(np.array(f['events']))
                if i == 0:
                    image1 = np.array(f['image1'])
                    start = float(f['start'][()])
                else:
                    assert(stop == float(f['start'][()]))
                image2 = np.array(f['image2'])
                stop = float(f['stop'][()])
        events = np.vstack(events)
        # convert float to list with one element
        start, stop = map(lambda x: [x], (start, stop))

        return events, start, stop, image1, image2

    def _rotate(self, image1, image2, events, angle):
         # we have to know image shape to initialize random_rotation
        if self.random_rotation is None:
            self.random_rotation = RandomRotation(self.angle, image1.shape[:2])
        return self.random_rotation(image1, image2, events, angle)

    def __getitem__(self, idx, k=None, is_flip=None, angle=None, box=None):
        ''' returns events and corresponding images
        Args:
            idx (int): index

        Return:
            events (ndarray of floats): event image
            start (float): first timestamps of the sliding window
            stop (float): last timestamp of the sliding window
            image1 (ndarray of floats): the image immediately before the sliding window
            image2 (ndarray of floats): the image immediately after the sliding window
        '''

        # choose collapse length from 1 to self.collapse_length
        if k is None:
            if self.augmentation:
                # collapse events for several images
                max_k = len(self) - idx
                choises = min(self.collapse_length, max_k)
                k = np.random.randint(choises) + 1
            else:
                k = 1

        # read data
        events, start, stop, image1, image2 = self._get_k_elems(idx, k)

        # align timestamps (it fix problem with to huge timestamps stored as fp32)
        if self.is_align: # subtract start timestamp
            events[:, 2] -= start[0]
            stop[0] -= start[0]
            start[0] = 0

        # convert events to float32
        events = events.astype(np.float32)

        if self.augmentation:
            is_flip = np.random.rand() < 0.5 if is_flip is None else is_flip
            if is_flip:
                # horizontal flip
                image1 = image1[:,::-1]
                image2 = image2[:,::-1]
                events[:, 0] = image1.shape[1] - events[:, 0] - 1
            # rotate image
            image1, image2, events, angle = self._rotate(image1, image2, events, angle)
        else:
            is_flip = False

        # crop
        image1, box = self.img_crop_fun(image1, box=box) # input box is None if it isn't specified
        image2, _   = self.img_crop_fun(image2, box=box)
        events      = self.event_crop_fun(events, box=box)

        # convert images to float32 with channels as a first dimension
        image1, image2 = map(lambda x: x[None].astype(np.float32), (image1, image2))
        assert all(events[:, 2] >= start[0])
        assert all(events[:, 2] <= stop[0])

        if self.is_raw:
            samples = events
        else:
            events = add_sample_index(events, 0)
            samples = compute_event_image(events, np.array(start), np.array(stop), self.shape)[0]

        if self.return_aug:
            box = np.array(box, dtype=int)
            is_flip = np.array([is_flip], dtype=bool)
            return samples, start, stop, image1, image2, (idx, k, box, angle, is_flip)
        else:
            return samples, start, stop, image1, image2 # add channel dimension

def add_sample_index(events, i):
    return np.hstack((events, np.full(events[:, [0]].shape, i, dtype=np.float32)))

def collate_wrapper(batch):
    to_tensor = lambda x: torch.FloatTensor(x)

    #    0        1     2     3        4
    # (events, start, stop, image1, image2)
    # add sample index to events
    events = np.vstack([add_sample_index(sample[0], i) for i, sample in enumerate(batch)])
    start = np.hstack([sample[1] for sample in batch])
    stop = np.hstack([sample[2] for sample in batch])
    image1 = np.vstack([sample[3][None] for sample in batch])
    image2 = np.vstack([sample[4][None] for sample in batch])
    add_info = tuple()
    if len(batch) > 0 and len(batch[0]) > 5:
        #    0        1     2      3       4     5[0] 5[1] 5[2]  5[3]    5[4]
        # (events, start, stop, image1, image2, (idx,  k,  box, angle, is_flip))
        idx = np.array([sample[5][0] for sample in batch])
        k = np.array([sample[5][1] for sample in batch])
        box = np.vstack([sample[5][2][None] for sample in batch])
        angle = np.array([sample[5][3] for sample in batch])
        is_flip = np.array([sample[5][4] for sample in batch])
        add_info = (tuple(map(to_tensor, (idx, k, box, angle, is_flip))), )
        
    return tuple(map(to_tensor, (events, start, stop, image1, image2))) + add_info
