from typing import Tuple

import numpy as np
import torch

from flabplatform.flabcls.registry import BATCH_AUGMENTS


@BATCH_AUGMENTS.register_module()
class Mixup:
    r"""Mixup 批量增强方法。

    参数:
        alpha (float): 用于 Beta 分布生成混合比例的参数。它应该是一个正数。
            更多细节请参考下面的说明。
    """

    def __init__(self, alpha: float):
        assert isinstance(alpha, float) and alpha > 0

        self.alpha = alpha

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
        index = torch.randperm(batch_size)

        mixed_inputs = lam * batch_inputs + (1 - lam) * batch_inputs[index, :]
        mixed_scores = lam * batch_scores + (1 - lam) * batch_scores[index, :]

        return mixed_inputs, mixed_scores

    def __call__(self, batch_inputs: torch.Tensor, batch_score: torch.Tensor):
        assert batch_score.ndim == 2, \
            'The input `batch_score` should be a one-hot format tensor, '\
            'which shape should be ``(N, num_classes)``.'

        mixed_inputs, mixed_score = self.mix(batch_inputs, batch_score.float())
        return mixed_inputs, mixed_score
