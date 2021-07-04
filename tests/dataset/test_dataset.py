import h5py
import numpy as np
from tests.utils import data_path
import torch


from utils.dataset import Dataset, DatasetImpl, collate_wrapper


def test_read():
    dataset = Dataset(path=data_path,
                      shape=[256, 256],
                      augmentation=True,
                      collapse_length=2,
                      is_raw=True)
    assert len(dataset) > 0
    events, timestamps, images = dataset[0]
    assert events.shape[1] == 5, 'Events are the matrix of ' \
                                 '5 columns [x, y, p, t, k]'
    assert (events[:, 4] != 0).sum() == 0, 'Sample is a sequence ' \
                                           'of 2 images'
    assert images.ndim == 3
    assert images.shape == (2, 256, 256)
    assert timestamps.shape == (2,)
    assert timestamps[0] < timestamps[1]


def test_data_augmentation_collapse():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True,
                          return_aug=True)
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

    with h5py.File(data_path/'000001.hdf5', 'r') as f1, \
            h5py.File(data_path/'000002.hdf5', 'r') as f2:
        gt_events = np.vstack([f1['events'], f2['events']])
        gt_events = np.hstack([gt_events, np.full_like(gt_events[:, [0]], 0)])
        start = float(f1['start'][()])
        stop = float(f2['stop'][()])
        gt_events[:, 2] -= start
        gt_timestamps = np.array([0, stop - start])
        image1 = np.array(f1['image1'])[None]
        image2 = np.array(f2['image2'])[None]
        assert float(f1['stop'][()]) == float(f2['start'][()])
        assert (np.array(f1['image2']) == np.array(f2['image1'])).all()
        gt_images = np.concatenate([image1, image2], axis=0).astype(np.float32)
    assert (events == gt_events).all()
    assert (timestamps == gt_timestamps).all()
    assert (images == gt_images).all()


def test_data_augmentation_flip():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True,
                          return_aug=True)
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
    first_indices = np.ravel_multi_index(events[:, 1::-1].T.astype(int),
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
    second_indices = np.ravel_multi_index(events[:, 1::-1].T.astype(int),
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
                          is_raw=True,
                          return_aug=True)
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
    rotated_indices = np.ravel_multi_index(events[:, 1::-1].T.astype(int),
                                           rotated_images[0].shape)
    H, W = rotated_images.shape[-2:]
    assert W % 2 == 0
    assert H % 2 == 0
    x = -(events[:, [1]] - H // 2) + W // 2
    y = (events[:, [0]] - W // 2) + H // 2
    assert (y < H).all()
    assert (y >= 0).all()
    assert (x < W).all()
    assert (x >= 0).all()
    original_indices = np.ravel_multi_index(np.vstack([y.T, x.T]).astype(int),
                                            [H, W])

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
                          is_raw=True,
                          return_aug=True)
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
    assert (events[:, :2] >= 0).all()
    assert (events[:, 0] < gt_box[-1]).all()
    assert (events[:, 1] < gt_box[-2]).all()

    with h5py.File(data_path/'000001.hdf5', 'r') as f:
        gt_events = np.array(f['events'])
        image1 = np.array(f['image1'])[None]
        image2 = np.array(f['image2'])[None]
        gt_images = np.concatenate([image1, image2], axis=0).astype(np.float32)

    box_stop = [gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]
    assert (gt_images[:,
                      gt_box[0]:box_stop[0],
                      gt_box[1]:box_stop[1]] == images).all()
    mask = np.logical_and(np.logical_and(gt_events[:, 0] >= gt_box[1],
                                         gt_events[:, 0] < box_stop[1]),
                          np.logical_and(gt_events[:, 1] >= gt_box[0],
                                         gt_events[:, 1] < box_stop[0]))
    # (x, y) -> linear index
    cropped_indices = np.ravel_multi_index(events[:, 1::-1].T.astype(int),
                                           images.shape[-2:])
    original_indices = np.ravel_multi_index(
            gt_events[mask, 1::-1].T.astype(int),
            gt_images.shape[-2:])
    for i in range(images.shape[0]):
        assert (images[i].ravel()[cropped_indices] ==
                gt_images[i].ravel()[original_indices]).all()


def test_data_augmentation_sequence():
    dataset = DatasetImpl(path=data_path,
                          shape=[256, 256],
                          augmentation=True,
                          collapse_length=2,
                          is_raw=True,
                          return_aug=True)
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

    with h5py.File(data_path/'000001.hdf5', 'r') as f1, \
            h5py.File(data_path/'000002.hdf5', 'r') as f2:
        gt_events1 = np.array(f1['events'])
        gt_events2 = np.array(f2['events'])
        gt_events = map(lambda x, y: np.hstack((x,
                                                np.full_like(x[:, [0]], y))),
                        (gt_events1, gt_events2), (0, 1))
        gt_events = np.vstack(list(gt_events))
        start = float(f1['start'][()])
        intermediate = float(f1['stop'][()])
        stop = float(f2['stop'][()])
        gt_events[:, 2] -= start
        gt_timestamps = np.array([start, intermediate, stop]) - start
        image1 = np.array(f1['image1'])[None]
        image2 = np.array(f1['image2'])[None]
        image3 = np.array(f2['image2'])[None]
        assert float(f1['stop'][()]) == float(f2['start'][()])
        assert (np.array(f1['image2']) == np.array(f2['image1'])).all()
        gt_images = np.concatenate([image1, image2, image3], axis=0) \
                      .astype(np.float32)
    assert (events == gt_events).all()
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
                          is_raw=True,
                          return_aug=True)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              collate_fn=collate_wrapper,
                                              batch_size=2,
                                              pin_memory=True,
                                              shuffle=False)
    events, timestamps, sample_idx, images, augmentation_params = next(iter(data_loader))

    with h5py.File(data_path/'000000.hdf5', 'r') as f0, \
            h5py.File(data_path/'000001.hdf5', 'r') as f1:
        gt_events0 = np.array(f0['events'])
        start0 = float(f0['start'][()])
        gt_events0[:, 2] -= start0
        gt_events1 = np.array(f1['events'])
        start1 = float(f1['start'][()])
        gt_events1[:, 2] -= start1
        gt_events = map(lambda x, y: np.hstack((x,
                                                np.full_like(x[:, [0]], 0),
                                                np.full_like(x[:, [0]], y))),
                        (gt_events0, gt_events1), (0, 1))
        gt_events = torch.tensor(np.vstack(list(gt_events)),
                                 dtype=torch.float32)
        stop0 = float(f0['stop'][()]) - start0
        stop1 = float(f1['stop'][()]) - start1
        gt_timestamps = torch.tensor([0, stop0, 0, stop1], dtype=torch.float32)
        gt_sample_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        image00 = torch.tensor(f0['image1'], dtype=torch.float32)[None, None]
        image01 = torch.tensor(f0['image2'], dtype=torch.float32)[None, None]
        image10 = torch.tensor(f1['image1'], dtype=torch.float32)[None, None]
        image11 = torch.tensor(f1['image2'], dtype=torch.float32)[None, None]
        gt_images = torch.cat([image00, image01, image10, image11], dim=0) \
                         .to(torch.float32)

    assert torch.equal(events, gt_events)
    assert torch.equal(timestamps, gt_timestamps)
    print(sample_idx)
    print(gt_sample_idx)
    assert torch.equal(sample_idx, gt_sample_idx)
    assert (images == gt_images).all()
