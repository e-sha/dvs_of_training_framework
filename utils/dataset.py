import h5py
from pathlib import Path
import numpy as np
import random
import torch
import torch.utils.data
import typing

from EV_FlowNet.net import compute_event_image
from .common import cumsum_with_prefix
from .data import EventCrop, ImageRandomCrop
from .data import RandomRotation, ImageCentralCrop


Augmentation_t = typing.Dict[str, torch.Tensor]


def read_info(filename):
    with h5py.File(filename, 'r') as f:
        sets = list(map(lambda x: x.decode(), f['set_name']))
        start_times = list(f['start_time'])
    return dict(zip(sets, start_times))


def select_encoded_ranges(events_per_element: torch.Tensor,
                          elements_per_sample: torch.Tensor,
                          sample_begin: int,
                          sample_end: int):
    """Computes begin and end indices requred to subsed an encoded batch of
    [sample_begin, sample_end) samples.

    Args:
        events_per_elment:
            Number of events stored for each element.
        elements_per_sample:
            Number of elements in each sample.
        sample_begin:
            Index of the first sample in the batch.
        sample_end:
            Index of the next to the last sample in the batch.

    Returns:
        A dictionary of begin and end indices for each tensor in the encoded
        samples. The result follows the structure of the encoded batch.
        For example, begin and end indices for x coordinate of events are
        located at ['events']['x']['begin'] and ['events']['x']['end'].
    """
    assert isinstance(sample_begin, int)
    assert isinstance(sample_end, int)
    assert sample_end > sample_begin

    events_shift = cumsum_with_prefix(events_per_element)
    elements_shift = cumsum_with_prefix(elements_per_sample)
    timestamps_shift = cumsum_with_prefix(elements_per_sample + 1)

    events_per_element_begin = elements_shift[sample_begin].item()
    events_per_element_end = elements_shift[sample_end].item()
    events_begin = events_shift[events_per_element_begin].item()
    events_end = events_shift[events_per_element_end].item()
    timestamp_begin = timestamps_shift[sample_begin].item()
    timestamp_end = timestamps_shift[sample_end].item()
    return {'events': {'x': {'begin': events_begin,
                             'end': events_end},
                       'y': {'begin': events_begin,
                             'end': events_end},
                       'timestamp': {'begin': events_begin,
                                     'end': events_end},
                       'polarity': {'begin': events_begin,
                                    'end': events_end},
                       'events_per_element': {
                           'begin': events_per_element_begin,
                           'end': events_per_element_end},
                       'elements_per_sample': {'begin': sample_begin,
                                               'end': sample_end}},
            'timestamps': {'begin': timestamp_begin, 'end': timestamp_end},
            'images': {'begin': timestamp_begin, 'end': timestamp_end},
            'augmentation_params': {
                'idx': {'begin': sample_begin, 'end': sample_end},
                'sequence_length': {'begin': sample_begin, 'end': sample_end},
                'collapse_length': {'begin': sample_begin, 'end': sample_end},
                'box': {'begin': sample_begin, 'end': sample_end},
                'angle': {'begin': sample_begin, 'end': sample_end},
                'is_flip': {'begin': sample_begin, 'end': sample_end}}}


def join_batches(batches: typing.List[typing.Dict]):
    """ Joins encoded batches to build a bigger batch

    Args:
        batches:
            List of encoded batches. Each batch is a dict with keys
            (events, timestamps, images, augmentation_params).

    Returns:
        A joined batch as a dict with keys
        (events, timestamps, images, augmentation_params).
    """

    if len(batches) == 0:
        return {'events': {'x': torch.tensor([], dtype=torch.short),
                           'y': torch.tensor([], dtype=torch.short),
                           'timestamp': torch.tensor([], dtype=torch.float32),
                           'polarity': torch.tensor([], dtype=torch.bool),
                           'events_per_element':
                           torch.tensor([], dtype=torch.short),
                           'elements_per_sample':
                           torch.tensor([], dtype=torch.short)},
                'timestamps': torch.tensor([], dtype=torch.float32),
                'images': torch.tensor([], dtype=torch.uint8),
                'augmentation_params': {}}
    if len(batches) == 1:
        return batches[0]
    result = {}
    for k in batches[0].keys():
        if isinstance(batches[0][k], dict):
            result[k] = {}
            for sk in batches[0][k].keys():
                result[k][sk] = torch.cat([el[k][sk] for el in batches])
        elif batches[0][k] is None:
            assert k == 'augmentation_params'
            assert all([el[k] is None for el in batches])
            result[k] = None
        else:
            assert isinstance(batches[0][k], torch.Tensor)
            result[k] = torch.cat([el[k] for el in batches])
    return result


def encode_batch(events: torch.Tensor,
                 timestamps: torch.Tensor,
                 sample_idx: torch.Tensor,
                 images: torch.Tensor,
                 augmentation_params: Augmentation_t,
                 size: int):
    """Encodes a batch to decrease storage space

    Args:
        events:
            Events for a batch in [x, y, timestamp, polarity, element, sample]
        timestamps:
            Timestamps of images in the batch
        sample_idx:
            Sample indices of the timestamps and images
        images:
            Images at the given timestamps
        augmentation_params:
            Augmentation parameters as a dictionary
        size:
            A number of samples in the batch

    Returns:
        A dictionary representation of the encoded batch with keys
        (events, timestamps, images, augmentation_params).
        events is a dictionary with keys
        (x, y, timestamp, polarity, events_per_element, elements_per_sample),
        where events.x is one-dimensional tensor of x coordinates as
        torch.int16;
        events.y is one-dimensional tensor of y coordinates as torch.int16;
        events.timestamps is a one-dimensional tensor of timestamps as
        torch.float32;
        events.polarities is a one-dimensional boolean tensor of polarities;
        events.events_per_element is a one-dimensional short tensor
        representing a number of events in each element;
        events.elements_per_sample is a one-dimensional short tensor
        representing a number of elements in each sample;
        timestamps is one-dimensional float tensor representing
        timestamps of images;
        sample_idx is one-dimensional short tensor representing;
        images is a uint8 tensor representing images;
        augmentation_params is dictionary of augmentation parameters.
    """
    x = events[:, 0].to(torch.short)
    y = events[:, 1].to(torch.short)
    t = events[:, 2]
    p = ((events[:, 3] + 1) / 2).to(torch.bool)
    e = events[:, 4].to(torch.long).numpy()
    s = events[:, 5].to(torch.short).numpy()

    elements_per_sample = np.zeros(size, dtype=np.short) - 1
    np.add.at(elements_per_sample, sample_idx, np.ones(sample_idx.numel()))
    elements_per_sample = torch.tensor(elements_per_sample)
    new_e = np.zeros(e.size, dtype=np.uint8)
    element_shift = np.array([0] + elements_per_sample.tolist(), dtype=np.long)
    element_shift = np.cumsum(element_shift)
    new_e = e + element_shift[s]
    total_elements = int(new_e[-1]) + 1

    events_per_element = np.zeros(total_elements, dtype=np.long)
    np.add.at(events_per_element, new_e, np.ones_like(new_e))
    events_per_element = torch.tensor(events_per_element)
    return {'events': {'x': x, 'y': y, 'timestamp': t, 'polarity': p,
                       'events_per_element': events_per_element,
                       'elements_per_sample': elements_per_sample},
            'timestamps': timestamps, 'images': images.to(torch.uint8),
            'augmentation_params': augmentation_params}


def decode_batch(encoded_batch):
    """Decodes a batch of encoded images

    Args:
        encoded_batch:
            A dictionary representation of the encoded batch as in
            encode_batch.

    Returns:
        Batch of data as a tuple of
        (events, timestamps, sample_idx, images, augmentation_params, size)
    """
    events = encoded_batch['events']
    timestamps = encoded_batch['timestamps']
    images = encoded_batch['images']
    augmentation_params = encoded_batch['augmentation_params']
    polarity = events['polarity'].view(-1, 1).to(torch.float32) * 2 - 1
    sample_idx = torch.cat([
        torch.full([n.item() + 1], i, dtype=torch.long)
        for i, n in enumerate(events['elements_per_sample'])])
    batch_size = events['elements_per_sample'].numel()
    sample_shift = cumsum_with_prefix(events['elements_per_sample'],
                                      dtype=torch.long)
    num_elements = events['events_per_element'].numel()
    element_index = []
    sample_index = []
    for i, num_elements in enumerate(events['elements_per_sample']):
        current_events_per_element = \
                events['events_per_element'][sample_shift[i]:
                                             sample_shift[i + 1]]
        num_events = sum(current_events_per_element).item()
        element_index.append(torch.cat(
            [torch.full([n], j, dtype=torch.float32)
             for j, n in enumerate(current_events_per_element)]))
        sample_index.append(torch.full([num_events], i, dtype=torch.float32))
    element_index = torch.cat(element_index)
    sample_index = torch.cat(sample_index)
    out_events = torch.cat([events['x'].view(-1, 1).to(torch.float32),
                            events['y'].view(-1, 1).to(torch.float32),
                            events['timestamp'].view(-1, 1),
                            polarity,
                            element_index.view(-1, 1),
                            sample_index.view(-1, 1)], dim=1)
    return out_events, timestamps.to(torch.float32), \
        sample_idx, images.to(torch.float32), augmentation_params, batch_size


def write_encoded_batch(path: Path,
                        batch: typing.Dict):
    """Writes encoded batch to a file in hdf5 file

    Args:
        path:
            A file path to write
        batch:
            Same as an output of encode_batch
    """
    def write_element(descriptor, data, name):
        if isinstance(data, torch.Tensor):
            descriptor.create_dataset(name, data=data)
            return
        assert isinstance(data, dict), name
        subgroup = descriptor.create_group(name)
        for k, v in data.items():
            write_element(subgroup, v, k)

    with h5py.File(path, 'w') as f:
        for k, v in batch.items():
            write_element(f, v, k)


def read_encoded_batch(descriptor: h5py.File,
                       events_per_element: torch.Tensor,
                       elements_per_sample: torch.Tensor,
                       sample_begin: int,
                       sample_end):
    """Reads batch of encoded samples in range [sample_begin, sample_end).

    Args:
        descriptor:
            A descriptor of an open hdf5 file with samples.
        events_per_element:
            Same as in select_encoded_ranges.
        elements_per_sample:
            Same as in select_encoded_ranges.
        sample_begin:
            Same as in select_encoded_ranges.
        sample_end:
            Same as in select_encoded_ranges.

    Returns:
        An encoded batch of samples from sample_begin to sample_end-1.
    """
    def is_final(element):
        assert isinstance(element, dict), element
        return 'begin' in element and isinstance(element['begin'], int) and \
               'end' in element and isinstance(element['end'], int)

    def read_data(descriptor, ranges):
        assert isinstance(ranges, dict)
        result = {}
        for k, v in ranges.items():
            if is_final(v):
                result[k] = torch.tensor(descriptor[k][v['begin']:v['end']])
            else:
                result[k] = read_data(descriptor[k], v)
        return result

    ranges = select_encoded_ranges(events_per_element,
                                   elements_per_sample,
                                   sample_begin,
                                   sample_end)
    return read_data(descriptor, ranges)


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
                 max_seq_length=1,   # maximum number of expected OF
                                     # predictions per sample
                 is_static_seq_length=True,
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

        return events, start, stop, image1, image2

    def _rotate(self, images, events, angle):
        # we have to know image shape to initialize random_rotation
        if self.random_rotation is None:
            self.random_rotation = RandomRotation(self.angle,
                                                  images.shape[-2:])
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
            _events, _start, _stop, _image1, _image2 = \
                self._get_k_elems(idx + i * k, k)
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
            angle = 0

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

        box = np.array(box, dtype=int)
        is_flip = np.array([is_flip], dtype=bool)
        return (samples,
                image_ts,
                images,
                (idx, seq_length, k, box, angle, is_flip))


class PreprocessedDataloader:
    """Dataloader that reads preprocessed data

    It iterates over files with preprocessed encoded data and sequently
    returns data.

    Attributes:
        file_index: An index of the current file to read
        sample_index: An index of the next sample in the current file
        batch_size: Number of samples in a batch
        files: Files with the preprocessed datasets
        length: Number of samples in the preprocessed dataset
        num_samples_per_file: Number of samples in each file
    """

    def __init__(self,
                 path: Path,
                 batch_size: int):
        """Inits PreprocessedDataloader with path to preprocessed dataset
        and batch size

        Args:
            path:
                A path to the preprocessed dataset
            batch_size:
                Number of samples per batch
        """
        self.batch_size = batch_size
        self.files = sorted(path.glob('*.hdf5'), key=lambda x: int(x.stem))
        self.file_index = 0
        self.sample_index = 0
        self.num_samples_per_file = []
        for file in self.files:
            with h5py.File(file, 'r') as f:
                self.num_samples_per_file.append(len(
                    f['events']['elements_per_sample']))
        self.length = sum(self.num_samples_per_file)

    def set_index(self,
                  idx: int):
        """Moves sample iterator to the specified index

        Args:
            idx:
                An index of the sample to start
        """
        idx = idx % self.length
        cs = torch.cumsum(torch.tensor(self.num_samples_per_file), 0)
        # cs[i-1] <= idx < cs[i]
        self.file_index = torch.searchsorted(cs, idx + 1).item()
        self.sample_index = idx if self.file_index == 0 \
            else idx - cs[self.file_index - 1].item()

    def __len__(self):
        """Returns number of samples in the preprocessed dataset"""
        return self.length

    def __iter__(self):
        """Returns an iterator over the dataset"""
        return self

    def __next__(self):
        """Returns the next batch"""
        num2read = self.batch_size
        batches = []
        while num2read > 0:
            left = self.num_samples_per_file[self.file_index] - \
                    self.sample_index
            cur_num2read = min(left, num2read)
            next_sample_index = self.sample_index + cur_num2read
            if cur_num2read > 0:
                with h5py.File(self.files[self.file_index], 'r') as f:
                    events_per_element = torch.tensor(
                            f['events']['events_per_element'])
                    elements_per_sample = torch.tensor(
                            f['events']['elements_per_sample'])
                    batches.append(read_encoded_batch(f, events_per_element,
                                                      elements_per_sample,
                                                      self.sample_index,
                                                      next_sample_index))
            self.sample_index = next_sample_index
            num2read -= cur_num2read
            if num2read > 0:
                self.file_index = (self.file_index + 1) % len(self.files)
                self.sample_index = 0
        encoded_batch = join_batches(batches)
        return decode_batch(encoded_batch)


def add_sample_index(events, i):
    return np.hstack((events, np.full_like(events[:, [0]], i)))


def collate_wrapper(batch):
    def to_tensor(x):
        if isinstance(x, np.ndarray) and x.dtype == np.int_:
            return torch.LongTensor(x)
        return torch.FloatTensor(x)

    #     0          1        2            3
    # (events, timestamps, images, augmentation_params)
    # add sample index to events
    events = np.vstack([add_sample_index(sample[0], i)
                        for i, sample in enumerate(batch)])
    sample_idx = np.hstack([np.full_like(sample[1], i, dtype=np.int_)
                            for i, sample in enumerate(batch)])
    timestamps = np.hstack([sample[1]
                            for sample in batch])
    images = np.vstack([x[2] for x in batch])
    images = np.expand_dims(images, axis=1)
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
        info_dict = {'idx': idx, 'sequence_length': seq_length,
                     'collapse_length': k, 'box': box, 'angle': angle,
                     'is_flip': is_flip}
        add_info = ({k: to_tensor(v) for k, v in info_dict.items()}, )

    events, timestamps, sample_idx, images = tuple(map(to_tensor, (events,
                                                                   timestamps,
                                                                   sample_idx,
                                                                   images)))
    return {'events': events, 'timestamps': timestamps,
            'sample_idx': sample_idx, 'images': images,
            'augmentation_params': add_info[0], 'size': len(batch)}
