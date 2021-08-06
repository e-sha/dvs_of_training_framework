import torch
from torch import nn

from ..serializer import Serializer


class SerializationHook:
    """Serializes model, optimizer and writes summary on a disk."""
    def __init__(self,
                 serializer: Serializer,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 logger: torch.utils.tensorboard.SummaryWriter):
        """Inits SerializationHook

        Args:
            serializer:
                A serializer module to use.
            model:
                A model to validate.
            optimizer:
                An optimizer to serialize.
            logger:
                A summary writer those data should be flushed.
        """
        self.serializer = serializer
        self.model = model
        self.optimizer = optimizer
        self.logger = logger

    def __call__(self,
                 steps: int,
                 samples: int):
        """Performs serialization.

        Args:
            steps:
                Number of passed steps.
            samples:
                Number of passed samples.
        """
        self.serializer.checkpoint_model(
            self.model,
            self.optimizer,
            global_step=steps,
            samples_passed=samples)
        self.logger.flush()
        print(f'Flushed logs for step {steps} ({samples} passed)')
