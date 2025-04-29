from typing import Optional, Tuple

import torch
import torch.nn as nn

from flabplatform.flabcls.registry import MODELS
from .cls_head import ClsHead


@MODELS.register_module()
class LinearClsHead(ClsHead):
    """线性分类器头部。

    参数:
        num_classes (int): 类别数量（不包括背景类别）。
        in_channels (int): 输入特征图的通道数。
        loss (dict): 分类损失的配置。默认值为 ``dict(type='CrossEntropyLoss', loss_weight=1.0)``。
        topk (int | Tuple[int]): Top-k 准确率。默认值为 ``(1, )``。
        cal_acc (bool): 是否在训练期间计算准确率。
            如果在训练期间使用了 Mixup 或 CutMix 等批量增强方法，计算准确率是没有意义的。
            默认值为 False。
        init_cfg (dict, optional): 控制初始化的配置。
            默认值为 ``dict(type='Normal', layer='Linear', std=0.01)``。
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        # The LinearClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
