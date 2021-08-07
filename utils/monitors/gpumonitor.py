from gpuutils import GpuUtils
from multiprocessing import Process
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


def monitor(path: Path,
            period: int):
    """Performs monitoring of GPU utilization

    Args:
        path:
            Path to write monitors.
        period:
            Period of writting information.
    """
    logger = SummaryWriter(str(path))
    k = 0
    while True:
        monitors = GpuUtils.analyzeSystem(pandas_format=False)
        if len(monitors['gpu_index']) == 0:
            print('No GPUs found')
            break
        for i, utilization, memory_available, memory_utilization \
                in zip(monitors['gpu_index'],
                       monitors['utilizations'],
                       monitors['available_memories_in_mb'],
                       monitors['memory_usage_percentage']):
            logger.add_scalar(f'Monitoring/GPU{i}/utilization',
                              utilization, k)
            logger.add_scalar(f'Monitoring/GPU{i}/MB left',
                              memory_available, k)
            logger.add_scalar(f'Monitoring/GPU{i}/memory utilization',
                              memory_utilization, k)
            k += 1
        time.sleep(period)


class GPUMonitor:
    """Writes information about GPU utilization using SummaryWriter.

    Attributes:
        path:
            Path to write monitorings.
        period:
            Period of writting information.
        process:
            Process that writes monitors.
    """

    def __init__(self,
                 path: Path,
                 period: int = 30):
        """Inits GPUMonitor using path and period"""
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        self.period = period
        self.process = None

    def __enter__(self):
        self.process = Process(target=monitor, args=(self.path, self.period))
        self.process.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.terminate()
        self.process = None
