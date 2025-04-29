from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from flabplatform.flabcls.registry import BATCH_AUGMENTS
from .cutmix import CutMix


@BATCH_AUGMENTS.register_module()
class ResizeMix(CutMix):
    """ResizeMix 随机粘贴层，用于一批数据的增强。

    参数:
        alpha (float): 用于 Beta 分布生成混合比例的参数。它应该是一个正数。
            更多细节请参考 :class:`Mixup`。
        lam_min (float): 混合比例 `lam` 的最小值。默认为 0.1。
        lam_max (float): 混合比例 `lam` 的最大值。默认为 0.8。
        interpolation (str): 用于上采样的算法：
            'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'。
            默认为 'bilinear'。
        prob (float): 执行 ResizeMix 的概率。范围应在 [0, 1] 之间。默认为 1.0。
        cutmix_minmax (List[float], optional): 裁剪区域的最小/最大面积比例。
            如果不为 None，则在该比例范围内均匀采样裁剪区域的边界框，
            并忽略 ``alpha`` 参数。否则，边界框根据 ``alpha`` 生成。
            默认为 None。
        correct_lam (bool): 当裁剪区域被图像边界裁剪时，是否应用混合比例修正。
            默认为 True。
        **kwargs: 其他可被 :class:`CutMix` 接受的参数。
    """

    def __init__(self,
                 alpha: float,
                 lam_min: float = 0.1,
                 lam_max: float = 0.8,
                 interpolation: str = 'bilinear',
                 cutmix_minmax: Optional[List[float]] = None,
                 correct_lam: bool = True):
        super().__init__(
            alpha=alpha, cutmix_minmax=cutmix_minmax, correct_lam=correct_lam)
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.interpolation = interpolation

    def mix(self, batch_inputs: torch.Tensor,
            batch_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对批量输入和批量标签进行混合。

        参数:
            batch_inputs (Tensor): 一批图像张量，形状为 ``(N, C, H, W)``。
            batch_scores (Tensor): 一批独热格式的标签，形状为 ``(N, num_classes)``。

        返回:
            Tuple[Tensor, Tensor]: 混合后的输入和标签。
        """
        lam = np.random.beta(self.alpha, self.alpha)
        lam = lam * (self.lam_max - self.lam_min) + self.lam_min
        img_shape = batch_inputs.shape[-2:]
        batch_size = batch_inputs.size(0)
        index = torch.randperm(batch_size)

        (y1, y2, x1, x2), lam = self.cutmix_bbox_and_lam(img_shape, lam)
        batch_inputs[:, :, y1:y2, x1:x2] = F.interpolate(
            batch_inputs[index],
            size=(int(y2 - y1), int(x2 - x1)),
            mode=self.interpolation,
            align_corners=False)
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return batch_inputs, mixed_scores
