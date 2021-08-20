import numpy as np
import torch


from utils.dataset import Dataset, DatasetImpl, collate_wrapper
from tests.utils import data_path, read_test_elem, concat_events, compare


def test_read():
    dataset = Dataset(path=data_path,
                      shape=[256, 256],
                      augmentation=True,
                      collapse_length=2,
                      is_raw=True)
    assert len(dataset) > 0
    events, timestamps, images, augmentation_parameters = dataset[0]
    assert isinstance(events, dict)
    assert set.intersection(
            set(events.keys()),
            {'x', 'y', 'timestamp', 'polarity', 'element_index'})
    assert isinstance(events['x'], np.ndarray)
    assert isinstance(events['y'], np.ndarray)
    assert isinstance(events['timestamp'], np.ndarray)
    assert isinstance(events['polarity'], np.ndarray)
    assert isinstance(events['element_index'], np.ndarray)
    assert events['x'].dtype == np.int64
    assert events['y'].dtype == np.int64
    assert events['timestamp'].dtype == np.float32
    assert events['polarity'].dtype == np.int64
    assert events['element_index'].dtype == np.int64
    n = events['x'].size
    for k, v in events.items():
        assert v.size == n, k
    assert (events['element_index'] != 0).sum() == 0, 'Sample is a sequence ' \
                                                      'of more than 1 element'
    assert images.ndim == 3
    assert images.shape == (2, 256, 256)
    assert timestamps.shape == (2,)
    assert timestamps[0] < timestamps[1]


def test_data_augmentation_collapse():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True)
    gt_idx, gt_k, gt_flip, gt_angle = 1, 2, False, 0
    gt_box, gt_seq_length = np.array([0, 0, 260, 346]), 1
    events, timestamps, images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip,
            angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]

    element1 = tuple(read_test_elem(1, element_index=0, box=gt_box))
    element2 = tuple(read_test_elem(2, element_index=0, box=gt_box))
    gt_events = concat_events(element1[0], element2[0])
    gt_timestamps = np.array([0, element2[2] - element1[1]])
    gt_events['timestamp'] -= element1[1]
    assert element1[2] == element2[1]
    assert (element1[4] == element2[3]).all()
    gt_images = np.concatenate([element1[3][None], element2[4][None]],
                               axis=0).astype(np.float32)

    compare(events, gt_events)
    assert (timestamps == gt_timestamps).all()
    print(timestamps)
    print(gt_timestamps)
    print(type(timestamps), type(gt_timestamps))
    print(type(images == gt_images))
    print(type(images), type(gt_images))
    print(images.shape, gt_images.shape)
    assert (images == gt_images).all()


def test_data_augmentation_flip():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True)
    gt_idx, gt_k, gt_flip, gt_angle = 1, 1, True, 0
    gt_box, gt_seq_length = np.array([0, 0, 260, 346]), 1
    events, timestamps, first_images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip, angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]

    # (x, y) -> linear index
    first_indices = np.ravel_multi_index(
            np.vstack([events['y'][None], events['x'][None]]),
            first_images[0].shape)

    gt_flip = not gt_flip
    events, timestamps, second_images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip, angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]

    # (x, y) -> linear index
    second_indices = np.ravel_multi_index(
            np.vstack([events['y'][None], events['x'][None]]),
            second_images[0].shape)

    assert (first_images != second_images).any()
    assert first_images.shape == second_images.shape
    for i in range(first_images.shape[0]):
        assert (first_images[i].ravel()[first_indices] ==
                second_images[i].ravel()[second_indices]).all()


def test_data_augmentation_angle():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True)
    gt_idx, gt_k, gt_flip, gt_angle = 1, 1, False, 90
    gt_box, gt_seq_length = np.array([0, 0, 260, 346]), 1
    events, timestamps, rotated_images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip, angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]

    # (x, y) -> linear index
    rotated_indices = np.ravel_multi_index(
            np.vstack([events['y'][None], events['x'][None]]),
            rotated_images[0].shape)
    H, W = rotated_images.shape[-2:]
    assert W % 2 == 0
    assert H % 2 == 0
    x = -(events['y'][None] - H // 2) + W // 2
    y = (events['x'][None] - W // 2) + H // 2
    assert (y < H).all()
    assert (y >= 0).all()
    assert (x < W).all()
    assert (x >= 0).all()
    original_indices = np.ravel_multi_index(np.vstack([y, x]), [H, W])

    gt_angle = 0
    _, _, original_images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip, angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]

    assert (original_images != rotated_images).any()
    assert original_images.shape == rotated_images.shape
    for i in range(original_images.shape[0]):
        assert (original_images[i].ravel()[original_indices] ==
                rotated_images[i].ravel()[rotated_indices]).all()


def test_data_augmentation_crop():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True)
    gt_idx, gt_k, gt_flip, gt_angle = 1, 1, False, 0
    gt_box, gt_seq_length = np.array([1, 2, 100, 150]), 1
    events, timestamps, images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip, angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]
    assert images.shape[-2:] == tuple(gt_box[-2:])
    assert (events['x'] >= 0).all()
    assert (events['y'] >= 0).all()
    assert (events['x'] < gt_box[-1]).all()
    assert (events['y'] < gt_box[-2]).all()

    gt_events, _, _, gt_image1, gt_image2 = read_test_elem(
            gt_idx, element_index=0)
    gt_images = np.concatenate([gt_image1[None], gt_image2[None]], axis=0)

    box_stop = [gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
    assert (gt_images[:,
                      gt_box[0]:box_stop[0],
                      gt_box[1]:box_stop[1]] == images).all()
    mask = np.logical_and(np.logical_and(gt_events['x'] >= gt_box[1],
                                         gt_events['x'] < box_stop[1]),
                          np.logical_and(gt_events['y'] >= gt_box[0],
                                         gt_events['y'] < box_stop[0]))
    # (x, y) -> linear index
    cropped_indices = np.ravel_multi_index(
            np.vstack([events['y'][None], events['x'][None]]),
            images.shape[-2:])
    original_indices = np.ravel_multi_index(
            np.vstack([gt_events['y'][mask][None],
                       gt_events['x'][mask][None]]),
            gt_images.shape[-2:])
    for i in range(images.shape[0]):
        assert (images[i].ravel()[cropped_indices] ==
                gt_images[i].ravel()[original_indices]).all()


def test_data_augmentation_sequence():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True)
    gt_idx, gt_k, gt_flip, gt_angle = 1, 1, False, 0
    gt_box, gt_seq_length = np.array([0, 0, 260, 346]), 2
    events, timestamps, images, aug_params = dataset.__getitem__(
            idx=gt_idx, k=gt_k, is_flip=gt_flip,
            angle=gt_angle, box=gt_box,
            seq_length=gt_seq_length)
    assert gt_idx == aug_params[0]
    assert gt_seq_length == aug_params[1]
    assert gt_k == aug_params[2]
    assert (gt_box == aug_params[3]).all()
    assert gt_angle == aug_params[4]
    assert gt_flip == aug_params[5]

    element1 = tuple(read_test_elem(gt_idx, element_index=0))
    element2 = tuple(read_test_elem(gt_idx + 1, element_index=1))
    gt_events = concat_events(element1[0], element2[0])
    gt_events['timestamp'] -= element1[1]
    gt_timestamps = np.array([element1[1],
                              element1[2],
                              element2[2]]) - element1[1]
    assert element1[2] == element2[1]
    assert (element1[4] == element2[3]).all()
    gt_images = np.concatenate([element1[3][None],
                                element1[4][None],
                                element2[4][None]], axis=0)
    compare(events, gt_events)
    assert (timestamps == gt_timestamps).all()
    assert (images[0] == gt_images[0]).all()
    assert (images[1] == gt_images[1]).all()
    assert np.max(np.abs(images - gt_images).reshape(-1)) == 0
    assert (images == gt_images).all()


def test_dataloader():
    dataset = DatasetImpl(path=data_path,
                          shape=[260, 346],
                          augmentation=False,
                          collapse_length=1,
                          is_raw=True)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=2,
                                              pin_memory=True,
                                              shuffle=False)
    batch = next(iter(data_loader))

    element1 = tuple(read_test_elem(0, element_index=0, is_torch=True))
    element2 = tuple(read_test_elem(1, element_index=0, is_torch=True))
    element1[0]['timestamp'] -= element1[1]
    element2[0]['timestamp'] -= element2[1]
    gt_events = concat_events(element1[0], element2[0])
    gt_events['sample_index'] = np.hstack([
        np.full_like(element1[0]['x'], 0),
        np.full_like(element2[0]['x'], 1)])
    gt_events = {k: torch.tensor(v) for k, v in gt_events.items()}
    gt_timestamps = torch.tensor(
            [0, element1[2] - element1[1], 0, element2[2] - element1[1]],
            dtype=torch.float32)
    gt_sample_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    image00 = torch.tensor(element1[3], dtype=torch.float32)[None, None]
    image01 = torch.tensor(element1[4], dtype=torch.float32)[None, None]
    image10 = torch.tensor(element2[3], dtype=torch.float32)[None, None]
    image11 = torch.tensor(element2[4], dtype=torch.float32)[None, None]
    gt_images = torch.cat([image00, image01, image10, image11], dim=0) \
                     .to(torch.float32)

    assert torch.equal(batch['events'], gt_events)
    assert torch.equal(batch['timestamps'], gt_timestamps)
    assert torch.equal(batch['sample_idx'], gt_sample_idx)
    assert (batch['images'] == gt_images).all()
    assert batch['size'] == 2
