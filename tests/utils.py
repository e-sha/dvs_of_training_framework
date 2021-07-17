import h5py
import numpy as np
from pathlib import Path
import sys
import torch


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
                   box=[0, 0, np.inf, np.inf],
                   is_torch=False,
                   read_pred=False):
    map_function = torch.tensor if is_torch else lambda x: x
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
    image_crop = ImageCrop(box=box, return_box=False, channel_first=False)
    images = tuple(map(image_crop, images))
    return map(map_function, (events, start, stop, *images))
