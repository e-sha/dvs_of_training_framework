from pathlib import Path
import torch

from .common import get_torch_version


is_pyprof_initialized = False


class Profiler():
    def __init__(self, profiler_type, logdir=Path('.')):
        if profiler_type == 'None':
            enabled = False
            nvtx = False
        elif profiler_type == 'CPU':
            enabled = True
            nvtx = False
        elif profiler_type == 'NVTX':
            import pyprof
            enabled = True
            nvtx = True
        else:
            assert False, f'Unknown profiler type {profiler_type}'
        self._prof = None
        self._enabled = enabled
        self._nvtx = nvtx
        self._logdir = Path(logdir)
        self._logdir.mkdir(exist_ok=True, parents=True)
        major, minor, _ = get_torch_version()
        self._prof_kwargs = dict(enabled=self._enabled, profile_memory=True)
        self._group_kwargs = dict()
        if major > 1 or major == 1 and minor > 6:
            self._prof_kwargs.update(dict(with_stack=True))
            self._group_kwargs.update(dict(group_by_stack_n=15))
        global is_pyprof_initialized
        if not is_pyprof_initialized and self._enabled and self._nvtx:
            pyprof.init()
            is_pyprof_initialized = True

    def __enter__(self):
        if self._nvtx:
            self._prof = torch.autograd \
                              .profiler \
                              .emit_nvtx(self._enabled) \
                              .__enter__()
        else:
            self._prof = torch.autograd \
                              .profiler \
                              .profile(**self._prof_kwargs) \
                              .__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._enabled:
            self._prof.__exit__(exc_type, exc_val, exc_tb)
        if self._enabled and not self._nvtx:
            self._prof.export_chrome_trace(self._logdir/'tracefile.json')
            table_path = self._logdir/'table.txt'
            table = self._prof \
                        .key_averages(**self._group_kwargs) \
                        .table(sort_by='self_cpu_time_total')
            table_path.write_text(table)
