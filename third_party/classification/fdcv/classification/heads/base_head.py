"""
Copyright (C) 2025 dsl.
"""
import torch
from torch import nn
from typing import List, Optional, Union, Tuple, Any
from torch import Tensor

__all__ = ['create_base_head', 'create_fcn_head', 'num_features_model']


def children(m: nn.Module) -> List[nn.Module]:
    """
    Get children modules of the input module.

    Args:
        m (nn.Module): Input module

    Returns:
        List[nn.Module]: List of child modules
    """
    return list(m.children())


def num_children(m: nn.Module) -> int:
    """
    Get number of children modules in the input module.

    Args:
        m (nn.Module): Input module

    Returns:
        int: Number of child modules
    """
    return len(children(m))


class ParameterModule(nn.Module):
    """
    Register a lone parameter as a module.

    Args:
        p (nn.Parameter): Parameter to be registered
    """
    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that returns input unchanged."""
        return x


def children_and_parameters(m: nn.Module) -> List[nn.Module]:
    """
    Return the children of module and its direct parameters not registered in modules.

    Args:
        m (nn.Module): Input module

    Returns:
        List[nn.Module]: List of children modules and parameter modules
    """
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()]
                     for c in m.children()], [])
    for p in m.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children


def flatten_model(m: nn.Module) -> List[nn.Module]:
    """
    Flatten a model's hierarchy into a list of leaf modules.

    Args:
        m (nn.Module): Input module

    Returns:
        List[nn.Module]: Flattened list of leaf modules
    """
    return sum(map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]


def one_param(m: nn.Module) -> nn.Parameter:
    """
    Return the first parameter of a module.

    Args:
        m (nn.Module): Input module

    Returns:
        nn.Parameter: First parameter of the module
    """
    return next(m.parameters())


def in_channels(m: nn.Module) -> int:
    """
    Return the number of input channels from the first weight layer.

    Args:
        m (nn.Module): Input module

    Returns:
        int: Number of input channels

    Raises:
        Exception: If no weight layer is found

    """
    for l in flatten_model(m):
        if hasattr(l, 'weight'):
            return l.weight.shape[1]
    raise Exception('No weight layer')


def dummy_batch(m: nn.Module, size: Tuple[int, int] = (64, 64)) -> Tensor:
    """
    Create a dummy batch to test module with specified size.

    Args:
        m (nn.Module): Input module
        size (Tuple[int, int], optional): Size of dummy input. Defaults to (64, 64)

    Returns:
        Tensor: Dummy batch tensor
    """
    ch_in = 3
    try:
        ch_in = in_channels(m)
    except Exception as e:
        pass
    return one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1., 1.)
    
def dummy_eval(m: nn.Module, size: Tuple[int, int] = (64, 64)) -> Tensor:
    """
    Pass a dummy batch through module in evaluation mode.

    Args:
        m (nn.Module): Input module
        size (Tuple[int, int], optional): Size of dummy input. Defaults to (64, 64)

    Returns:
        Tensor: Output tensor from module
    """
    return m.eval()(dummy_batch(m, size))
    
def num_features_model(m: nn.Module) -> int:
    """
    Calculate the number of output features for a model.

    Args:
        m (nn.Module): Input module

    Returns:
        int: Number of output features

    Raises:
        Exception: If unable to determine output features with any input size
    """
    try_sz = [128, 224, 256, 384, 448, 512, 1024]
    for sz in try_sz:
        try:
            x = dummy_eval(m, (sz, sz))
            return x.shape[1]
        except Exception as e:
            if sz >= 1024:
                raise


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: Optional[nn.Module] = None) -> List[nn.Module]:
    """
    Create a sequence of batchnorm, dropout, and linear layers.

    Args:
        n_in (int): Number of input features
        n_out (int): Number of output features
        bn (bool, optional): Whether to include BatchNorm. Defaults to True
        p (float, optional): Dropout probability. Defaults to 0.
        actn (Optional[nn.Module], optional): Activation function. Defaults to None

    Returns:
        List[nn.Module]: List of layers
    """
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: 
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: 
        layers.append(actn)
    return layers


def create_base_head(num_features: int, num_classes: int, lin_ftrs: Optional[List[int]] = None, ps: float = 0.5, bn_final: bool = False) -> nn.Sequential:
    """
    Create a classification head with configurable architecture.

    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        lin_ftrs (Optional[List[int]], optional): Sizes of hidden layers. Defaults to None
        ps (float, optional): Dropout probabilities. Defaults to 0.5
        bn_final (bool, optional): Whether to add final BatchNorm. Defaults to False

    Returns:
        nn.Sequential: Classification head module
    """
    lin_ftrs = [num_features, 512, num_classes] if lin_ftrs is None else [num_features] + lin_ftrs + [num_classes]
    ps = [ps/2] * (len(lin_ftrs) - 2) + [ps]
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    
    layers = []
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn=True, p=p, actn=actn)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


def bn_drop_conv2d(n_in: int, n_out: int, kernel_size: int = 1, bn: bool = True, p: float = 0., actn: Optional[nn.Module] = None) -> List[nn.Module]:
    """
    Create a sequence of batchnorm, dropout, and conv2d layers.

    Args:
        n_in (int): Number of input channels
        n_out (int): Number of output channels
        kernel_size (int, optional): Size of conv kernel. Defaults to 1
        bn (bool, optional): Whether to include BatchNorm. Defaults to True
        p (float, optional): Dropout probability. Defaults to 0.
        actn (Optional[nn.Module], optional): Activation function. Defaults to None
        
    Returns:
        List[nn.Module]: List of layers
    """
    layers = [nn.BatchNorm2d(n_in)] if bn else []
    if p != 0: 
        layers.append(nn.Dropout2d(p))
    layers.append(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=kernel_size//2))
    if actn is not None: 
        layers.append(actn)
    return layers


def create_fcn_head(num_features: int, num_classes: int, lin_ftrs: Optional[List[int]]=None, ps: float = 0.5, kernel_size: int = 1) -> nn.Sequential:
    """
    Create a Fully Convolutional classification head.
    
    Args:
        num_features (int): Number of input features.
        num_classes (int): Number of output classes.
        lin_ftrs (Optional[List[int]], optional): Sizes of hidden layers. Defaults to None.
        ps (float, optional): Dropout probability. Defaults to 0.5.
        kernel_size (int, optional): Kernel size for Conv2d. Defaults to 1.

    Returns:
        nn.Sequential: Fully convolutional head.
    """
    lin_ftrs = [num_features, 512, num_classes] if lin_ftrs is None else [num_features] + lin_ftrs + [num_classes]
    ps = [ps / 2] * (len(lin_ftrs) - 2) + [ps]
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = []
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_conv2d(ni, no, kernel_size=kernel_size, bn=True, p=p, actn=actn)
    return nn.Sequential(*layers)
