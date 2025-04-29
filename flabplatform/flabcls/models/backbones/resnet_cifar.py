import torch.nn as nn
from flabplatform.flabcv.cnn import build_conv_layer, build_norm_layer
from flabplatform.flabcls.registry import MODELS
from .resnet import ResNet


@MODELS.register_module()
class ResNet_CIFAR(ResNet):
    """用于 CIFAR 数据集的 ResNet 骨干网络。

    参数:
        depth (int): 网络深度，可选值为 {18, 34, 50, 101, 152}。
        in_channels (int): 输入图像的通道数。默认值: 3。
        stem_channels (int): stem 层的输出通道数。默认值: 64。
        base_channels (int): 第一阶段的中间通道数。默认值: 64。
        num_stages (int): 网络的阶段数。默认值: 4。
        strides (Sequence[int]): 每个阶段第一个块的步幅。默认值: ``(1, 2, 2, 2)``。
        dilations (Sequence[int]): 每个阶段的膨胀率。默认值: ``(1, 1, 1, 1)``。
        out_indices (Sequence[int]): 指定输出来自哪些阶段。如果只指定一个阶段，则返回单个张量（特征图）；如果指定多个阶段，则返回张量的元组。默认值: ``(3, )``。
        deep_stem (bool): 该网络具有特定设计的 stem，因此必须为 False。
        avg_down (bool): 在 Bottleneck 中下采样时使用 AvgPool 而不是 stride 卷积。默认值: False。
        frozen_stages (int): 冻结的阶段（停止梯度更新并设置为 eval 模式）。-1 表示不冻结任何参数。默认值: -1。
        conv_cfg (dict | None): 卷积层的配置字典。默认值: None。
        norm_cfg (dict): 归一化层的配置字典。
        norm_eval (bool): 是否将归一化层设置为 eval 模式，即冻结运行时统计量（均值和方差）。注意: 仅对 Batch Norm 及其变体有效。默认值: False。
        with_cp (bool): 是否使用检查点。使用检查点会节省一些内存，但会降低训练速度。默认值: False。
        zero_init_residual (bool): 是否对残差块中的最后一个归一化层使用零初始化，以使其表现为恒等映射。默认值: True。
    """

    def __init__(self, depth, deep_stem=False, **kwargs):
        super(ResNet_CIFAR, self).__init__(
            depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'

    def _make_stem_layer(self, in_channels, base_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
