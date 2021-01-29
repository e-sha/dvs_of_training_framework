import re
import torch


def get_torch_version():
    version = re.match(r'^([0-9.]*)', torch.__version__).groups()[0]
    return tuple(map(int, version.split('.')))
