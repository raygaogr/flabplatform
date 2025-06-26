import torch
from torch import nn
from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
from PIL import Image
from abc import ABCMeta, abstractmethod
from flabplatform.core.config import Config

class BaseRunner(nn.Module, metaclass=ABCMeta):
    """Base class for all runners.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def __call__(
        self, 
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        **kwargs: Any) -> List[Union[Dict]]:
        """Run the model."""
        return self.predict(source, **kwargs)

    @abstractmethod
    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        **kwargs: Any
    ) -> List[Union[Dict]]:
        """Predict the model."""
        pass
    
    @abstractmethod
    def val(self, **kwargs: Any) -> None:
        """Validate the model."""
        pass

    @abstractmethod
    def export(self, format="onnx", **kwargs: Any) -> None:
        """Export the model."""
        pass

    @classmethod
    @abstractmethod
    def from_cfg(cls, cfg: Union[Dict, Config], **kwargs: Any):
        pass


