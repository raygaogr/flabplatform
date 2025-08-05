from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.utils import is_str

if hasattr(torch, 'tensor_split'):
    tensor_split = torch.tensor_split
else:
    # A simple implementation of `tensor_split`.
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs


LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """创建一个独立的图形对象。

    与 :func:`plt.figure` 不同，此函数创建的图形不会被 matplotlib 管理。
    它具有 :obj:`matplotlib.backends.backend_agg.FigureCanvasAgg`，因此您可以通过
    ``canvas`` 属性访问绘制的图像。

    参数:
        *args: :class:`matplotlib.figure.Figure` 的所有位置参数。
        margin (bool): 是否保留图形的白色边缘。默认为 False。
        **kwargs: :class:`matplotlib.figure.Figure` 的所有关键字参数。

    返回:
        matplotlib.figure.Figure: 创建的图形对象。
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def format_score(value: SCORE_TYPE) -> torch.Tensor:
    """将多种 Python 类型转换为分数格式的张量。

    支持的类型包括：:class:`numpy.ndarray`、:class:`torch.Tensor`、
    :class:`Sequence`。

    参数:
        value (torch.Tensor | numpy.ndarray | Sequence): 分数值。

    返回:
        :obj:`torch.Tensor`: 格式化后的分数张量。
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not is_str(value):
        value = torch.tensor(value).float()
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')
    assert value.ndim == 1, \
        f'The dims of value should be 1, but got {value.ndim}.'

    return value


def cat_batch_labels(elements: List[torch.Tensor]):
    """将一批标签张量拼接为一个张量。

    参数:
        elements (List[tensor]): 一批标签。

    返回:
        Tuple[torch.Tensor, List[int]]: 第一个元素是拼接后的标签张量，
        第二个元素是每个样本的分割索引。
    """
    labels = []
    splits = [0]
    for element in elements:
        labels.append(element)
        splits.append(splits[-1] + element.size(0))
    batch_label = torch.cat(labels)
    return batch_label, splits[1:-1]


def batch_label_to_onehot(batch_label, split_indices, num_classes):
    """将拼接的标签张量转换为one-hot编码。

    参数:
        batch_label (torch.Tensor): 从多个样本拼接的标签张量。
        split_indices (List[int]): 每个样本的分割索引。
        num_classes (int): 类别数量。

    返回:
        torch.Tensor: one-hot格式的标签张量。
    """
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [
        sparse_onehot.sum(0)
        for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)
    ]
    return torch.stack(onehot_list)


def label_to_onehot(label: LABEL_TYPE, num_classes: int):
    """将标签转换为one-hot格式的张量。

    参数:
        label (LABEL_TYPE): 标签值。
        num_classes (int): 类别数量。

    返回:
        torch.Tensor: one-hot格式的标签张量。
    """
    label = format_label(label)
    sparse_onehot = F.one_hot(label, num_classes)
    return sparse_onehot.sum(0)
