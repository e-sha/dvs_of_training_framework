from pathlib import Path
import torch

from .common import get_torch_version

class Profiler():
    def __init__(self, enabled=True, logdir=Path('.')):
        self._prof = None
        self._enabled = enabled
        self._logdir = Path(logdir)
        self._logdir.mkdir(exist_ok=True, parents=True)
        major, minor, _ = get_torch_version()
        self._prof_kwargs = dict(enabled=self._enabled, profile_memory=True)
        self._group_kwargs = dict()
        if major > 1 or major == 1 and minor > 6:
            self._prof_kwargs.update(dict(with_stack=True))
            self._group_kwargs.update(dict(group_by_stack_n=15))

    def __enter__(self):
        self._prof = torch.autograd.profiler.profile(**self._prof_kwargs).__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._prof.__exit__(exc_type, exc_val, exc_tb)
        if self._enabled:
            self._prof.export_chrome_trace(self._logdir/'tracefile.json')
            table_path = self._logdir/'table.txt'
            table_path.write_text(self._prof.key_averages(**self._group_kwargs).table(sort_by='self_cpu_time_total'))
