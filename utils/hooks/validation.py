from copy import deepcopy
import torch
from torch import nn
from typing import List


from ..training import validate
from ..loss import Losses


class ValidationHook:
    """Performs validation during the training process"""
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 loader: torch.utils.data.DataLoader,
                 logger: torch.utils.tensorboard.SummaryWriter,
                 losses: Losses,
                 weights: List[float],
                 is_raw: bool):
        """Inits ValidationHook

        Args:
            model:
                A model to validate.
            device:
                A device to perform computation.
            loader:
                A dataset loader.
            logger:
                A summary writer to log validation results.
            weights:
                Weights of losses in the combined loss.
            is_raw:
                A flag of raw events in the dataset.
        """
        self.model = model
        self.device = device
        self.loader = loader
        self.logger = logger
        self.losses = losses
        self.weights = deepcopy(weights)
        self.is_raw = is_raw

    def __call__(self,
                 steps: int,
                 samples: int):
        """Performs validation

        Args:
            steps:
                Number of passed steps.
            samples:
                Number of passed samples.
        """
        validate(self.model, self.device, self.loader, samples,
                 self.logger, self.losses, weights=self.weights,
                 is_raw=self.is_raw)
