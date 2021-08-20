import h5py
import numpy as np
from pathlib import Path
import sys
import torch


from utils.common import to_tensor


test_path = Path(__file__).parent.resolve()
while test_path.name != 'tests':
    test_path = test_path.parent
sys.path.append(str(test_path.parent))
data_path = test_path/'data/seq'
pred_path = test_path/'data/pred'


try:
    from utils.data import ImageCrop, EventCrop
except ImportError:
    raise


def read_test_elem(i,
                   element_index=None,
                   box=[0, 0, np.inf, np.inf],
                   is_torch=False,
                   read_pred=False):
    def map_function(data):
        if not is_torch:
            return data
        return to_tensor(data)

    filename = f'{i:06d}.hdf5'
    with h5py.File(data_path/filename, 'r') as f:
        events, start, stop, image1, image2 = np.array(f['events']), \
                float(f['start'][()]), float(f['stop'][()]), \
                np.array(f['image1']), np.array(f['image2'])
    images = (image1, image2)
    if read_pred:
        with h5py.File(pred_path/filename, 'r') as f:
            images = (*images, np.array(f['flow']))
    box = np.array(box)
    shape = np.array(images[0].shape[:2])
    box[:2] = np.minimum(box[:2], shape)
    box[2:] = np.minimum(shape - box[:2], box[2:])
    box = box.astype(int)
    events = EventCrop(box=box)(events)
    events = {'x': events[:, 0].astype(np.int64),
              'y': events[:, 1].astype(np.int64),
              'timestamp': events[:, 2],
              'polarity': events[:, 3].astype(np.int64)}
    if element_index is not None:
        events['element_index'] = np.full_like(events['x'],
                                               element_index,
                                               dtype=np.int_)
    image_crop = ImageCrop(box=box, return_box=False, channel_first=False)
    images = tuple(map(image_crop, images))
    return map(map_function, (events, start, stop, *images))


def concat_events(*argv):
    keys = {'x', 'y', 'polarity', 'timestamp', 'element_index'}
    if len(argv) == 0:
        return {k: [] for k in keys}
    result = {}
    for k in keys:
        data = [x[k] for x in argv]
        if isinstance(argv[0][k], torch.Tensor):
            result[k] = torch.cat(data)
        else:
            result[k] = np.hstack(data)
    return result


def compare(computed, groundtruth, prefix=''):
    if isinstance(computed, torch.Tensor):
        assert isinstance(groundtruth, torch.Tensor), prefix
        assert torch.equal(computed, groundtruth), prefix
        return
    if isinstance(computed, np.ndarray):
        assert isinstance(groundtruth, np.ndarray), prefix
        assert (computed == groundtruth).all(), prefix
        return
    if isinstance(computed, int):
        assert isinstance(groundtruth, int), prefix
        assert computed == groundtruth, prefix
        return
    if isinstance(computed, tuple):
        assert isinstance(groundtruth, tuple)
        computed = {f'{i}': v for i, v in enumerate(computed)}
        groundtruth = {f'{i}': v for i, v in enumerate(groundtruth)}
    assert isinstance(computed, dict) and isinstance(groundtruth, dict),\
        prefix
    assert len(computed) == len(groundtruth),\
        f'{prefix}: {computed.keys()} {groundtruth.keys()}'
    for k in computed.keys():
        assert k in groundtruth
        compare(computed[k], groundtruth[k], prefix=prefix+f'.{k}')
