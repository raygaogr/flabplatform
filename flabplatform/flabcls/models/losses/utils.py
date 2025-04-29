import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """根据指定的方式减少损失。

    参数:
        loss (Tensor): 元素级别的损失张量。
        reduction (str): 减少方式，可选值为 "none"、"mean" 和 "sum"。

    返回:
        Tensor: 经过减少的损失张量。
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """应用元素级别的权重并减少损失。

    参数:
        loss (Tensor): 元素级别的损失。
        weight (Tensor): 元素级别的权重。
        reduction (str): 与 PyTorch 内置损失的减少方式相同，可选值为 "none"、"mean" 和 "sum"。
        avg_factor (float): 计算损失均值时的平均因子。

    返回:
        Tensor: 处理后的损失值。
    """
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss