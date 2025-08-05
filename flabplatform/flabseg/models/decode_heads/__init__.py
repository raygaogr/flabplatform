
from .fcn_head import FCNHead
from .segformer_head import SegformerHead
from .uper_head import UPerHead
from .decode_head import BaseDecodeHead
from .psp_head import PSPHead

__all__ = ['FCNHead', 'UPerHead', 'SegformerHead','BaseDecodeHead','PSPHead']
