import json
import torch
import pandas as pd
import math

from pathlib import Path
from itertools import repeat
from collections import OrderedDict

def create_bbox_mask(bbox_ratio, feature_size=7):
    """
    Create a binary mask based on bounding box ratio.
    
    Args:
    - bbox_ratio (tuple): (x, y, w, h) in ratio format.
    - feature_size (int): Size of the feature map (default is 7 for a 7x7 feature map).
    
    Returns:
    - mask (tensor): Binary mask of shape [feature_size, feature_size].
    """
    x, y, w, h = bbox_ratio
    # Convert ratio to actual coordinates
    x = x * feature_size
    y = y * feature_size
    w = w * feature_size
    h = h * feature_size

    min_x, min_y, max_x, max_y = int(x - w / 2), int(y - h / 2), math.ceil(x + w / 2), math.ceil(y + h / 2)
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, feature_size)
    max_y = min(max_y, feature_size)
    
    mask = torch.zeros((feature_size, feature_size))
    mask[min_y:max_y, min_x:max_x] = 1
    return mask

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
