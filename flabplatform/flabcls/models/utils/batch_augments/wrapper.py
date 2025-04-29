from typing import Callable, Union

import numpy as np
import torch

from flabplatform.flabcls.registry import BATCH_AUGMENTS


class RandomBatchAugment:
    """随机选择一个批量增强方法并应用。

    参数:
        augments (Callable | dict | list): 批量增强方法的配置，可以是一个可调用对象、字典或列表。
        probs (float | List[float] | None): 每种批量增强方法的概率。如果为 None，则均匀选择。
            默认为 None。
    """

    def __init__(self, augments: Union[Callable, dict, list], probs=None):
        if not isinstance(augments, (tuple, list)):
            augments = [augments]

        self.augments = []
        for aug in augments:
            if isinstance(aug, dict):
                self.augments.append(BATCH_AUGMENTS.build(aug))
            else:
                self.augments.append(aug)

        if isinstance(probs, float):
            probs = [probs]

        if probs is not None:
            assert len(augments) == len(probs), \
                '``augments`` and ``probs`` must have same lengths. ' \
                f'Got {len(augments)} vs {len(probs)}.'
            assert sum(probs) <= 1, \
                'The total probability of batch augments exceeds 1.'
            self.augments.append(None)
            probs.append(1 - sum(probs))

        self.probs = probs

    def __call__(self, batch_input: torch.Tensor, batch_score: torch.Tensor):
        """Randomly apply batch augmentations to the batch inputs and batch
        data samples."""
        aug_index = np.random.choice(len(self.augments), p=self.probs)
        aug = self.augments[aug_index]

        if aug is not None:
            return aug(batch_input, batch_score)
        else:
            return batch_input, batch_score.float()
