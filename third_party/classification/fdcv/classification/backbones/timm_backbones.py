"""
Copyright (C) 2025 dsl.
"""
import torch
from torch import nn
from typing import Optional, Union, Tuple
from torch import Tensor
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ['TimmBackbone']


class AdaptiveConcatPool2d(nn.Module):
    """
    Concatenated pooling layer that combines max pooling and average pooling.

    Args:
        sz (Optional[Union[int, Tuple[int, int]]]): Output size of the pooling operation.
            If None, defaults to 1. Can be an integer for square output or tuple for rectangular.
    """
    def __init__(self, sz: Optional[Union[int, Tuple[int, int]]] = None):
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the pooling layer.

        Args:
            x (Tensor): Input tensor of shape (batch, channles, height, width)

        Returns:
            Tensor: Concatenated output of max and average pooling
                    Shape: (Batch, channels * 2, output_size, output_size)
        """
        return torch.cat([self.mp(x), self.ap(x)], 1)


class TimmBackbone(nn.Module):
    """
    Wrapper for timm model backbones with flexible pooling options.

    Args:
        backbone_name (str): Name of the timm model architecture
        img_c (int): Number of input image channels (default: 3)
        pretrained (bool): Whether to use pretained weights (default: True)
        concat_pool (bool): Whether to use concatenated pooling (default: True)
        **kwargs: Additional arguments passed to timm.create_model
    """
    def __init__(self, backbone_name: str, img_c: int = 3, pretrained: bool = True, concat_pool: bool = True, **kwargs):
        super().__init__()
        import timm
        # model = timm.create_model(
        #     'mobilenetv4_conv_medium',
        #     pretrained=True,
        #     pretrained_cfg_overlay={'file': '/root/.cache/huggingface/hub/models--timm--mobilenetv4_conv_medium.e500_r256_in1k/snapshots/ad66898c045c1b5223ea3f2c0830b74cf2e75bac/model.safetensors'},
        #     num_classes=1000  # 根据您的实际情况设置类别数
        # )
        self.model = timm.create_model(backbone_name, pretrained=pretrained, in_chans=img_c,features_only=True, **kwargs)
        self.pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
    
    def forward(self, x: Tensor) -> Tensor:
        # Extract features only, output shape:(batch_size, num_features, height, width)
        #x = self.model.forward_features(x)
        x = self.model(x)[-1]  
        x = self.pool(x)
        x = self.flatten(x)
        return x
