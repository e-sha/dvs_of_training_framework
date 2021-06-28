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


try:
    from utils.data import ImageCrop, EventCrop
except ImportError:
    raise


def read_test_elem(i, box=[0, 0, np.inf, np.inf], is_torch=False):
    map_function = torch.tensor if is_torch else lambda x: x
    with h5py.File(data_path/f'{i:06d}.hdf5', 'r') as f:
        events, start, stop, image1, image2 = np.array(f['events']), \
                float(f['start'][()]), float(f['stop'][()]), \
                np.array(f['image1']), np.array(f['image2'])
        box = np.array(box)
        shape = np.array(image1.shape[:2])
        box[:2] = np.minimum(box[:2], image1.shape[:2])
        box[2:] = np.minimum(shape - box[:2], box[2:])
        box = box.astype(int)
        events = EventCrop(box=box)(events)
        image_crop = ImageCrop(box=box, return_box=False, channel_first=False)
        image1, image2 = map(image_crop, (image1, image2))
        return map(map_function, (events, start, stop, image1, image2))
