
"""Collecting some commonly used type hint in mmflow."""
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from flabplatform.core.config import ConfigDict

from flabplatform.flabseg.structures import SegDataSample

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, Sequence[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

SampleList = Sequence[SegDataSample]
OptSampleList = Optional[SampleList]

# Type hint of Tensor
TensorDict = Dict[str, torch.Tensor]
TensorList = Sequence[torch.Tensor]

ForwardResults = Union[Dict[str, torch.Tensor], List[SegDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
