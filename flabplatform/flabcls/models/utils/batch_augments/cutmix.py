from typing import List, Optional, Tuple

import numpy as np
import torch

from flabplatform.flabcls.registry import BATCH_AUGMENTS
from .mixup import Mixup


@BATCH_AUGMENTS.register_module()
class CutMix(Mixup):
    r"""CutMix 批量增强方法。

    参数:
        alpha (float): 用于 Beta 分布生成混合比例的参数。它应该是一个正数。
            更多细节请参考 :class:`Mixup`。
        cutmix_minmax (List[float], optional): 裁剪区域的最小/最大面积比例。
            如果不为 None，则在该比例范围内均匀采样裁剪区域的边界框，
            并忽略 ``alpha`` 参数。否则，边界框根据 ``alpha`` 生成。
            默认为 None。
        correct_lam (bool): 当裁剪区域被图像边界裁剪时，是否应用混合比例修正。
            默认为 True。
    """

    def __init__(self,
                 alpha: float,
                 cutmix_minmax: Optional[List[float]] = None,
                 correct_lam: bool = True):
        super().__init__(alpha=alpha)

        self.cutmix_minmax = cutmix_minmax
        self.correct_lam = correct_lam

    def rand_bbox_minmax(
            self,
            img_shape: Tuple[int, int],
            count: Optional[int] = None) -> Tuple[int, int, int, int]:
        """生成基于最小/最大面积比例的随机裁剪区域。

        此方法受 Darknet 的 CutMix 实现启发。它根据输入图像的每个维度，
        基于最小/最大比例生成一个随机矩形边界框。

        参数:
            img_shape (tuple): 图像的形状，格式为 (高度, 宽度)。
            count (int, optional): 要生成的边界框数量。默认为 None。

        返回:
            Tuple[int, int, int, int]: 裁剪区域的左上角和右下角坐标。
        """
        assert len(self.cutmix_minmax) == 2
        img_h, img_w = img_shape
        cut_h = np.random.randint(
            int(img_h * self.cutmix_minmax[0]),
            int(img_h * self.cutmix_minmax[1]),
            size=count)
        cut_w = np.random.randint(
            int(img_w * self.cutmix_minmax[0]),
            int(img_w * self.cutmix_minmax[1]),
            size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    def rand_bbox(self,
                  img_shape: Tuple[int, int],
                  lam: float,
                  margin: float = 0.,
                  count: Optional[int] = None) -> Tuple[int, int, int, int]:
        """生成基于混合比例的标准裁剪区域。

        此方法根据混合比例生成一个随机的正方形裁剪区域，并支持通过边界百分比
        限制裁剪区域的边界。

        参数:
            img_shape (tuple): 图像的形状，格式为 (高度, 宽度)。
            lam (float): CutMix 的混合比例。
            margin (float): 边界框尺寸的百分比，用于限制裁剪区域的边界。
                默认为 0。
            count (int, optional): 要生成的边界框数量。默认为 None。

        返回:
            Tuple[int, int, int, int]: 裁剪区域的左上角和右下角坐标。
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    def cutmix_bbox_and_lam(self,
                            img_shape: Tuple[int, int],
                            lam: float,
                            count: Optional[int] = None) -> tuple:
        """生成裁剪区域并应用混合比例修正。

        参数:
            img_shape (tuple): 图像的形状，格式为 (高度, 宽度)。
            lam (float): CutMix 的混合比例。
            count (int, optional): 要生成的边界框数量。默认为 None。

        返回:
            Tuple[Tuple[int, int, int, int], float]: 裁剪区域的坐标和修正后的混合比例。
        """
        if self.cutmix_minmax is not None:
            yl, yu, xl, xu = self.rand_bbox_minmax(img_shape, count=count)
        else:
            yl, yu, xl, xu = self.rand_bbox(img_shape, lam, count=count)
        if self.correct_lam or self.cutmix_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[0] * img_shape[1])
        return (yl, yu, xl, xu), lam

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
        batch_size = batch_inputs.size(0)
        img_shape = batch_inputs.shape[-2:]
        index = torch.randperm(batch_size)

        (y1, y2, x1, x2), lam = self.cutmix_bbox_and_lam(img_shape, lam)
        batch_inputs[:, :, y1:y2, x1:x2] = batch_inputs[index, :, y1:y2, x1:x2]
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return batch_inputs, mixed_scores
