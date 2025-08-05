from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from flabplatform.core.registry import MODELS


def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


@MODELS.register_module()
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


@MODELS.register_module()
class Dropout(nn.Dropout):
    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


def build_dropout(cfg: Dict, default_args: Optional[Dict] = None) -> Any:
    """Builder for drop out layers."""
    return MODELS.build(cfg, default_args=default_args)
