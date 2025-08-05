from itertools import product
from typing import List, Optional, Sequence, Union

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric

from flabplatform.flabcls.registry import METRICS


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


def _precision_recall_f1_support(pred_positive, gt_positive, average):
    """calculate base classification task metrics, such as  precision, recall,
    f1_score, support."""
    average_options = ['micro', 'macro', None]
    assert average in average_options, 'Invalid `average` argument, ' \
        f'please specify from {average_options}.'
    ignored_index = gt_positive == -1
    pred_positive[ignored_index] = 0
    gt_positive[ignored_index] = 0

    class_correct = (pred_positive & gt_positive)
    if average == 'micro':
        tp_sum = class_correct.sum()
        pred_sum = pred_positive.sum()
        gt_sum = gt_positive.sum()
    else:
        tp_sum = class_correct.sum(0)
        pred_sum = pred_positive.sum(0)
        gt_sum = gt_positive.sum(0)

    precision = tp_sum / torch.clamp(pred_sum, min=1).float() * 100
    recall = tp_sum / torch.clamp(gt_sum, min=1).float() * 100
    f1_score = 2 * precision * recall / torch.clamp(
        precision + recall, min=torch.finfo(torch.float32).eps)
    if average in ['macro', 'micro']:
        precision = precision.mean(0)
        recall = recall.mean(0)
        f1_score = f1_score.mean(0)
        support = gt_sum.sum(0)
    else:
        support = gt_sum
    return precision, recall, f1_score, support


@METRICS.register_module()
class Accuracy(BaseMetric):
    """准确率评估指标。

    对于二分类或多分类任务，准确率是所有预测中正确预测的比例：

    .. math::

        \text{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{all}}}

    参数:
        topk (int | Sequence[int]): 如果真实标签与前 **k** 个最佳预测之一匹配，则样本被视为正预测。
            如果参数是一个元组，则会同时计算并输出所有的 top-k 准确率。默认值为 1。
        thrs (Sequence[float | None] | float | None): 如果是浮点数，则分数低于阈值的预测将被视为负预测。
            如果为 None，则不应用阈值。如果参数是一个元组，则会同时计算并输出基于所有阈值的准确率。
            默认值为 0。
        collect_device (str): 在分布式训练期间用于收集结果的设备名称。必须是 'cpu' 或 'gpu'。
            默认值为 'cpu'。
        prefix (str, optional): 将添加到指标名称中的前缀，用于区分不同评估器的同名指标。
            如果未在参数中提供前缀，则使用 `self.default_prefix`。默认值为 None。

    """
    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 topk: Union[int, Sequence[int]] = (1, ),
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(topk, int):
            self.topk = (topk, )
        else:
            self.topk = tuple(topk)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)

    def process(self, data_batch, data_samples: Sequence[dict]):
        """处理一批数据样本。

        处理后的结果应存储在 `self.results` 中，稍后将使用这些结果计算指标。

        参数:
            data_batch: 从数据加载器中获取的一批数据。
            data_samples (Sequence[dict]): 模型输出的一批数据样本。
        """

        for data_sample in data_samples:
            result = dict()
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                result['pred_label'] = data_sample['pred_label'].cpu()
            result['gt_label'] = data_sample['gt_label'].cpu()
            # Save the result to `self.results`.
            self.results.append(result)

    def compute_metrics(self, results: List):
        """从处理后的结果中计算指标。

        参数:
            results (list): 每批次处理后的结果。

        返回:
            dict: 计算出的指标。字典的键是指标名称，值是对应的结果。
        """
        # NOTICE: don't access `self.results` from the method.
        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])

            try:
                acc = self.calculate(pred, target, self.topk, self.thrs)
            except ValueError as e:
                # If the topk is invalid.
                raise ValueError(
                    str(e) + ' Please check the `val_evaluator` and '
                    '`test_evaluator` fields in your config file.')

            multi_thrs = len(self.thrs) > 1
            for i, k in enumerate(self.topk):
                for j, thr in enumerate(self.thrs):
                    name = f'top{k}'
                    if multi_thrs:
                        name += '_no-thr' if thr is None else f'_thr-{thr:.2f}'
                    metrics[name] = acc[i][j].item()
        else:
            # If only label in the `pred_label`.
            pred = torch.cat([res['pred_label'] for res in results])
            acc = self.calculate(pred, target, self.topk, self.thrs)
            metrics['top1'] = acc.item()

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        topk: Sequence[int] = (1, ),
        thrs: Sequence[Union[float, None]] = (0., ),
    ) -> Union[torch.Tensor, List[List[torch.Tensor]]]:
        """计算准确率。

        参数:
            pred (torch.Tensor | np.ndarray | Sequence): 预测结果。可以是标签 (N, )，也可以是每个类别的分数 (N, C)。
            target (torch.Tensor | np.ndarray | Sequence): 每个预测的目标标签，形状为 (N, )。
            topk (Sequence[int]): 计算 top-k 准确率的 k 值。默认值为 (1, )。
            thrs (Sequence[float | None]): 如果是浮点数，则分数低于阈值的预测将被视为负预测。
                仅在 `pred` 是分数时使用。None 表示不应用阈值。默认值为 (0., )。

        返回:
            torch.Tensor | List[List[torch.Tensor]]: 准确率。

            - torch.Tensor: 如果 `pred` 是标签序列（维度为 1）。仅返回 top-1 准确率张量，忽略 `topk` 和 `thrs` 参数。
            - List[List[torch.Tensor]]: 如果 `pred` 是分数序列（维度为 2）。返回每个 `topk` 和 `thrs` 的准确率。
            第一维是 `topk`，第二维是 `thrs`。
        """

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        num = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).float().sum(0, keepdim=True)
            acc = correct.mul_(100. / num)
            return acc
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = max(topk)

            if maxk > pred.size(1):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(1)}.')

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            results = []
            for k in topk:
                results.append([])
                for thr in thrs:
                    # Only prediction values larger than thr are counted
                    # as correct
                    _correct = correct
                    if thr is not None:
                        _correct = _correct & (pred_score.t() > thr)
                    correct_k = _correct[:k].reshape(-1).float().sum(
                        0, keepdim=True)
                    acc = correct_k.mul_(100. / num)
                    results[-1].append(acc)
            return results

