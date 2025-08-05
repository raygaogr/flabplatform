import inspect
from typing import Dict, Tuple, Union

import torch.nn as nn
from flabplatform.core.registry import MODELS
from mmengine.utils import is_tuple_of
from mmengine.utils.dl_utils.parrots_wrapper import (SyncBatchNorm, _BatchNorm,
                                                     _InstanceNorm)

MODELS.register_module('BN', module=nn.BatchNorm2d)
MODELS.register_module('BN1d', module=nn.BatchNorm1d)
MODELS.register_module('BN2d', module=nn.BatchNorm2d)
MODELS.register_module('BN3d', module=nn.BatchNorm3d)
MODELS.register_module('SyncBN', module=SyncBatchNorm)
MODELS.register_module('GN', module=nn.GroupNorm)
MODELS.register_module('LN', module=nn.LayerNorm)
MODELS.register_module('IN', module=nn.InstanceNorm2d)
MODELS.register_module('IN1d', module=nn.InstanceNorm1d)
MODELS.register_module('IN2d', module=nn.InstanceNorm2d)
MODELS.register_module('IN3d', module=nn.InstanceNorm3d)


def _infer_abbr(class_type):
    """从类名推断归一化层的缩写。

    在使用 `build_norm_layer()` 构建归一化层时，希望在变量名称中保留归一化类型，
    例如 `self.bn1` 或 `self.gn`。此方法将推断类类型与缩写的映射关系。

    规则：
    1. 如果类具有属性 `_abbr_`，则返回该属性。
    2. 如果父类是 `_BatchNorm`、`GroupNorm`、`LayerNorm` 或 `InstanceNorm`，
    则缩写分别为 "bn"、"gn"、"ln" 和 "in"。
    3. 如果类名包含 "batch"、"group"、"layer" 或 "instance"，
    则缩写分别为 "bn"、"gn"、"ln" 和 "in"。
    4. 否则，缩写默认为 "norm"。

    参数:
        class_type (type): 归一化层的类型。

    返回:
        str: 推断的缩写。
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """构建归一化层。

    参数:
        cfg (dict): 归一化层的配置字典，应该包含以下内容：
            - type (str): 层的类型。
            - layer args: 实例化归一化层所需的参数。
            - requires_grad (bool, optional): 是否允许梯度更新。
        num_features (int): 输入通道数。
        postfix (int | str): 附加到归一化层缩写后的后缀，用于创建命名层。

    返回:
        tuple[str, nn.Module]: 第一个元素是由缩写和后缀组成的层名称，例如 `bn1` 或 `gn`。
        第二个元素是创建的归一化层。
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        # Switch registry to the target scope. If `norm_layer` cannot be found
        # in the registry, fallback to search `norm_layer` in the
        # mmengine.MODELS.
        with MODELS.switch_scope_and_registry(None) as registry:
            norm_layer = registry.get(layer_type)
        if norm_layer is None:
            raise KeyError(f'Cannot find {norm_layer} in registry under '
                           f'scope name {registry.scope}')
    abbr = _infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm(layer: nn.Module,
            exclude: Union[type, tuple, None] = None) -> bool:
    """检查一个层是否为归一化层。

    参数:
        layer (nn.Module): 要检查的层。
        exclude (type | tuple[type] | None): 要排除的类型。

    返回:
        bool: 是否为归一化层。
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)
