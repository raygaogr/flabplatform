import math
from numbers import Number
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from mmengine.model import (BaseDataPreprocessor, stack_batch)

from flabplatform.flabcls.registry import MODELS
from flabplatform.flabcls.structures import (DataSample, 
                                   batch_label_to_onehot, cat_batch_labels,
                                   tensor_split)
from .batch_augments import RandomBatchAugment


@MODELS.register_module()
class ClsDataPreprocessor(BaseDataPreprocessor):
    """用于分类任务的图像预处理器。

    它提供以下数据预处理功能：

    - 整理数据并将其移动到目标设备。
    - 使用定义的 ``pad_value`` 将输入填充到当前批次的最大尺寸。
    填充后的尺寸可以被定义的 ``pad_size_divisor`` 整除。
    - 将输入堆叠为批量输入。
    - 如果输入的形状为 (3, H, W)，将图像从 BGR 转换为 RGB。
    - 使用定义的均值和标准差对图像进行归一化。
    - 在训练期间执行批量增强（如 Mixup 和 Cutmix）。

    参数:
        mean (Sequence[Number], optional): R、G、B 通道的像素均值。默认为 None。
        std (Sequence[Number], optional): R、G、B 通道的像素标准差。默认为 None。
        pad_size_divisor (int): 填充后的图像尺寸应可被 ``pad_size_divisor`` 整除。默认为 1。
        pad_value (Number): 填充值。默认为 0。
        to_rgb (bool): 是否将图像从 BGR 转换为 RGB。默认为 False。
        to_onehot (bool): 是否生成独热格式的标签并设置到数据样本中。默认为 False。
        num_classes (int, optional): 类别数量。默认为 None。
        batch_augments (dict, optional): 批量增强设置，包括 "augments" 和 "probs"。
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Number = 0,
                 to_rgb: bool = False,
                 to_onehot: bool = False,
                 num_classes: Optional[int] = None,
                 batch_augments: Optional[dict] = None):
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.to_rgb = to_rgb
        self.to_onehot = to_onehot
        self.num_classes = num_classes

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                'preprocessing, please specify both `mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        if batch_augments:
            self.batch_augments = RandomBatchAugment(**batch_augments)
            if not self.to_onehot:
                from flabplatform.core.logging import MMLogger
                MMLogger.get_current_instance().info(
                    'Because batch augmentations are enabled, the data '
                    'preprocessor automatically enables the `to_onehot` '
                    'option to generate one-hot format labels.')
                self.to_onehot = True
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """基于 ``BaseDataPreprocessor`` 执行归一化、填充、BGR 转 RGB 转换和批量增强。

        参数:
            data (dict): 从数据加载器中采样的数据。
            training (bool): 是否启用训练时增强。

        返回:
            dict: 与模型输入格式相同的数据。
        """
        inputs = self.cast_data(data['inputs'])

        if isinstance(inputs, torch.Tensor):
            # ------ To RGB ------
            if self.to_rgb and inputs.size(1) == 3:
                inputs = inputs.flip(1)

            # -- Normalization ---
            inputs = inputs.float()
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

            # ------ Padding -----
            if self.pad_size_divisor > 1:
                h, w = inputs.shape[-2:]

                target_h = math.ceil(
                    h / self.pad_size_divisor) * self.pad_size_divisor
                target_w = math.ceil(
                    w / self.pad_size_divisor) * self.pad_size_divisor
                pad_h = target_h - h
                pad_w = target_w - w
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), 'constant',
                               self.pad_value)
        else:
            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if self.to_rgb and input_.size(0) == 3:
                    input_ = input_.flip(0)

                # -- Normalization ---
                input_ = input_.float()
                if self._enable_normalize:
                    input_ = (input_ - self.mean) / self.std

                processed_inputs.append(input_)
            # Combine padding and stack
            inputs = stack_batch(processed_inputs, self.pad_size_divisor,
                                 self.pad_value)

        data_samples = data.get('data_samples', None)
        sample_item = data_samples[0] if data_samples is not None else None

        if isinstance(sample_item, DataSample):
            batch_label = None
            batch_score = None

            if 'gt_label' in sample_item:
                gt_labels = [sample.gt_label for sample in data_samples]
                batch_label, label_indices = cat_batch_labels(gt_labels)
                batch_label = batch_label.to(self.device)
            if 'gt_score' in sample_item:
                gt_scores = [sample.gt_score for sample in data_samples]
                batch_score = torch.stack(gt_scores).to(self.device)
            elif self.to_onehot and 'gt_label' in sample_item:
                assert batch_label is not None, \
                    'Cannot generate onehot format labels because no labels.'
                num_classes = self.num_classes or sample_item.get(
                    'num_classes')
                assert num_classes is not None, \
                    'Cannot generate one-hot format labels because not set ' \
                    '`num_classes` in `data_preprocessor`.'
                batch_score = batch_label_to_onehot(
                    batch_label, label_indices, num_classes).to(self.device)

            # ----- Batch Augmentations ----
            if (training and self.batch_augments is not None
                    and batch_score is not None):
                inputs, batch_score = self.batch_augments(inputs, batch_score)

            # ----- scatter labels and scores to data samples ---
            if batch_label is not None:
                for sample, label in zip(
                        data_samples, tensor_split(batch_label,
                                                   label_indices)):
                    sample.set_gt_label(label)
            if batch_score is not None:
                for sample, score in zip(data_samples, batch_score):
                    sample.set_gt_score(score)

        return {'inputs': inputs, 'data_samples': data_samples}

