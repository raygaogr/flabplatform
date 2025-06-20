import math

import torch
import torch.nn as nn
from flabplatform.core.registry import MODELS
from torch.nn.modules.utils import _pair, _triple

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def obsolete_torch_version(torch_version, version_threshold) -> bool:
    return torch_version == 'parrots' or torch_version <= version_threshold


class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None


@MODELS.register_module('Conv', force=True)
class Conv2d(nn.Conv2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if obsolete_torch_version(TORCH_VERSION, (1, 4)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-2:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@MODELS.register_module('Conv3d', force=True)
class Conv3d(nn.Conv3d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if obsolete_torch_version(TORCH_VERSION, (1, 4)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-3:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@MODELS.register_module()
@MODELS.register_module('deconv')
class ConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if obsolete_torch_version(TORCH_VERSION, (1, 4)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@MODELS.register_module()
@MODELS.register_module('deconv3d')
class ConvTranspose3d(nn.ConvTranspose3d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if obsolete_torch_version(TORCH_VERSION, (1, 4)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-3:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


class MaxPool2d(nn.MaxPool2d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch 1.9 does not support empty tensor inference yet
        if obsolete_torch_version(TORCH_VERSION, (1, 9)) and x.numel() == 0:
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-2:], _pair(self.kernel_size),
                                     _pair(self.padding), _pair(self.stride),
                                     _pair(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)


class MaxPool3d(nn.MaxPool3d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch 1.9 does not support empty tensor inference yet
        if obsolete_torch_version(TORCH_VERSION, (1, 9)) and x.numel() == 0:
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-3:], _triple(self.kernel_size),
                                     _triple(self.padding),
                                     _triple(self.stride),
                                     _triple(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)

@MODELS.register_module()
class Linear(torch.nn.Linear):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # empty tensor forward of Linear layer is supported in Pytorch 1.6
        if obsolete_torch_version(TORCH_VERSION, (1, 5)) and x.numel() == 0:
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)
