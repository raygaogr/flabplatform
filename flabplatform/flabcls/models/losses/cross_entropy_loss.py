import torch.nn as nn
import torch.nn.functional as F

from flabplatform.flabcls.registry import MODELS
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """计算交叉熵损失。

    参数:
        pred (torch.Tensor): 预测值，形状为 (N, C)，C 是类别数量。
        label (torch.Tensor): 预测的真实标签。
        weight (torch.Tensor, optional): 样本级别的损失权重。
        reduction (str): 用于减少损失的方法。
        avg_factor (int, optional): 用于平均损失的因子。默认值为 None。
        class_weight (torch.Tensor, optional): 每个类别的权重，形状为 (C)，C 是类别数量。默认值为 None。

    返回:
        torch.Tensor: 计算出的损失。
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       class_weight=None,
                       avg_factor=None):
    """计算 Soft 版本的交叉熵损失。标签可以是浮点数。

    参数:
        pred (torch.Tensor): 预测值，形状为 (N, C)，C 是类别数量。
        label (torch.Tensor): 预测的真实标签，形状为 (N, C)。当使用 "mixup" 时，标签可以是浮点数。
        weight (torch.Tensor, optional): 样本级别的损失权重。
        reduction (str): 用于减少损失的方法。
        avg_factor (int, optional): 用于平均损失的因子。默认值为 None。
        class_weight (torch.Tensor, optional): 每个类别的权重，形状为 (C)，C 是类别数量。默认值为 None。

    返回:
        torch.Tensor: 计算出的损失。
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None):
    """计算带有 Logits 的二元交叉熵损失。

    参数:
        pred (torch.Tensor): 预测值，形状为 (N, *)。
        label (torch.Tensor): 真实标签，形状为 (N, *)。
        weight (torch.Tensor, optional): 每个元素的损失权重，形状为 (N, )。默认值为 None。
        reduction (str): 用于减少损失的方法。选项包括 "none"、"mean" 和 "sum"。
            如果 reduction 为 'none'，损失的形状与 pred 和 label 相同。默认值为 'mean'。
        avg_factor (int, optional): 用于平均损失的因子。默认值为 None。
        class_weight (torch.Tensor, optional): 每个类别的权重，形状为 (C)，C 是类别数量。默认值为 None。
        pos_weight (torch.Tensor, optional): 每个类别的正样本权重，形状为 (C)，C 是类别数量。默认值为 None。

    返回:
        torch.Tensor: 计算出的损失。
    """
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    assert pred.dim() == label.dim()

    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label.float(),  # only accepts float type tensor
        weight=class_weight,
        pos_weight=pos_weight,
        reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """交叉熵损失。

    参数:
        use_sigmoid (bool): 预测是否使用 Sigmoid 或 Softmax。默认值为 False。
        use_soft (bool): 是否使用 Soft 版本的交叉熵损失。默认值为 False。
        reduction (str): 用于减少损失的方法。选项包括 "none"、"mean" 和 "sum"。默认值为 'mean'。
        loss_weight (float): 损失的权重。默认值为 1.0。
        class_weight (List[float], optional): 每个类别的权重，形状为 (C)，C 是类别数量。默认值为 None。
        pos_weight (List[float], optional): 每个类别的正样本权重，形状为 (C)，C 是类别数量。
            仅在使用 Sigmoid 的二元交叉熵损失时启用。默认值为 None。
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_soft=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        assert not (
            self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
