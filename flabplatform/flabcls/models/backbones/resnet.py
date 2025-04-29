import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from flabplatform.flabcv.cnn import build_conv_layer, build_norm_layer, build_activation_layer, ConvModule
from flabplatform.flabcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from flabplatform.flabcls.registry import MODELS
from .base_backbone import BaseBackbone

eps = 1.0e-5

class BasicBlock(BaseModule):
    """ResNet 的 BasicBlock。

    参数:
        in_channels (int): 此块的输入通道数。
        out_channels (int): 此块的输出通道数。
        expansion (int): ``out_channels/mid_channels`` 的比例，其中 ``mid_channels`` 是 conv1 的输出通道数。
            这是 BasicBlock 中的保留参数，始终为 1。默认值: 1。
        stride (int): 此块的步幅。默认值: 1。
        dilation (int): 卷积的膨胀率。默认值: 1。
        downsample (nn.Module, optional): 对恒等分支进行下采样的操作。默认值: None。
        with_cp (bool): 是否使用检查点。使用检查点会节省一些内存，但会降低训练速度。
        conv_cfg (dict, optional): 用于构建和配置卷积层的字典。默认值: None。
        norm_cfg (dict): 用于构建和配置归一化层的字典。默认值: dict(type='BN')。
        drop_path_rate (float): 随机深度率。默认值: 0。
        act_cfg (dict): 用于构建和配置激活函数的字典。默认值: dict(type='ReLU', inplace=True)。
        init_cfg (dict): 初始化配置字典。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, out_channels, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """ResNet 的 Bottleneck 块。

    参数:
        in_channels (int): 此块的输入通道数。
        out_channels (int): 此块的输出通道数。
        expansion (int): ``out_channels/mid_channels`` 的比例，其中 ``mid_channels`` 是 conv2 的输入/输出通道数。
            默认值: 4。
        stride (int): 此块的步幅。默认值: 1。
        dilation (int): 卷积的膨胀率。默认值: 1。
        downsample (nn.Module, optional): 对恒等分支进行下采样的操作。默认值: None。
            否则，stride=2 的层是第一个 1x1 卷积层。默认值: "pytorch"。
        with_cp (bool): 是否使用检查点。使用检查点会节省一些内存，但会降低训练速度。
        conv_cfg (dict, optional): 用于构建和配置卷积层的字典。默认值: None。
        norm_cfg (dict): 用于构建和配置归一化层的字典。默认值: dict(type='BN')。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 drop_path_rate=0.0,
                 init_cfg=None):
        super(Bottleneck, self).__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, self.mid_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            in_channels,
            self.mid_channels,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            self.mid_channels,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            self.mid_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample
        self.drop_path = DropPath(drop_prob=drop_path_rate
                                  ) if drop_path_rate > eps else nn.Identity()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.drop_path(out)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def get_expansion(block, expansion=None):
    """获取 ResNet 的层级 ID，以设置不同的学习率。

    ResNet 的阶段划分：
        50  :    [3, 4, 6, 3]
        101 :    [3, 4, 23, 3]
        152 :    [3, 8, 36, 3]
        200 :    [3, 24, 36, 3]
        eca269d: [3, 30, 48, 8]

    参数:
        param_name (str): 参数的名称。
        prefix (str): 参数的前缀。默认为空字符串。

    返回:
        Tuple[int, int]: 层级深度和层的总数量。
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion


class ResLayer(nn.Sequential):
    """用于构建 ResNet 风格骨干网络的 ResLayer。

    参数:
        block (nn.Module): 用于构建 ResLayer 的残差块。
        num_blocks (int): 块的数量。
        in_channels (int): 此块的输入通道数。
        out_channels (int): 此块的输出通道数。
        expansion (int, optional): BasicBlock/Bottleneck 的扩展比例。
            如果未指定，将首先通过 ``block.expansion`` 获取。如果块没有 "expansion" 属性，
            将使用以下默认值: BasicBlock 为 1，Bottleneck 为 4。默认值: None。
        stride (int): 第一个块的步幅。默认值: 1。
        avg_down (bool): 在 Bottleneck 中下采样时使用 AvgPool 而不是 stride 卷积。默认值: False。
        conv_cfg (dict, optional): 用于构建和配置卷积层的字典。默认值: None。
        norm_cfg (dict): 用于构建和配置归一化层的字典。默认值: dict(type='BN')。
        drop_path_rate (float or list): 随机深度率。默认值: 0。
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 drop_path_rate=0.0,
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * num_blocks

        assert len(drop_path_rate
                   ) == num_blocks, 'Please check the length of drop_path_rate'

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion=self.expansion,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=drop_path_rate[0],
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=drop_path_rate[i],
                    **kwargs))
        super(ResLayer, self).__init__(*layers)


@MODELS.register_module()
class ResNet(BaseBackbone):
    """ResNet 骨干网络。

    参数:
        depth (int): 网络深度，可选值为 {18, 34, 50, 101, 152}。
        in_channels (int): 输入图像的通道数。默认值: 3。
        stem_channels (int): stem 层的输出通道数。默认值: 64。
        base_channels (int): 第一阶段的中间通道数。默认值: 64。
        num_stages (int): 网络的阶段数。默认值: 4。
        strides (Sequence[int]): 每个阶段第一个块的步幅。默认值: ``(1, 2, 2, 2)``。
        dilations (Sequence[int]): 每个阶段的膨胀率。默认值: ``(1, 1, 1, 1)``。
        out_indices (Sequence[int]): 输出来自哪些阶段。默认值: ``(3, )``。
        deep_stem (bool): 用三个 3x3 卷积替换输入 stem 中的 7x7 卷积。默认值: False。
        avg_down (bool): 在 Bottleneck 中下采样时使用 AvgPool 而不是 stride 卷积。默认值: False。
        frozen_stages (int): 冻结的阶段（停止梯度更新并设置为 eval 模式）。
            -1 表示不冻结任何参数。默认值: -1。
        conv_cfg (dict | None): 卷积层的配置字典。默认值: None。
        norm_cfg (dict): 归一化层的配置字典。
        norm_eval (bool): 是否将归一化层设置为 eval 模式，即冻结运行时统计量（均值和方差）。
            注意: 仅对 Batch Norm 及其变体有效。默认值: False。
        with_cp (bool): 是否使用检查点。使用检查点会节省一些内存，但会降低训练速度。默认值: False。
        zero_init_residual (bool): 是否对残差块中的最后一个归一化层使用零初始化，以使其表现为恒等映射。默认值: True。
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 expansion=None,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3, ),
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(
                         type='Constant',
                         val=1,
                         layer=['_BatchNorm', 'GroupNorm'])
                 ],
                 drop_path_rate=0.0):
        super(ResNet, self).__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.expansion = get_expansion(self.block, expansion)

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = res_layer[-1].out_channels

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                ConvModule(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True),
                ConvModule(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ResNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """获取 ResNet 的层级 ID，以设置不同的学习率。

        ResNet 的阶段划分：
            50  :    [3, 4, 6, 3]
            101 :    [3, 4, 23, 3]
            152 :    [3, 8, 36, 3]
            200 :    [3, 24, 36, 3]
            eca269d: [3, 30, 48, 8]

        参数:
            param_name (str): 参数的名称。
            prefix (str): 参数的前缀。默认为空字符串。

        返回:
            Tuple[int, int]: 层级深度和层的总数量。
        """
        depths = self.stage_blocks
        if depths[1] == 4 and depths[2] == 6:
            blk2, blk3 = 2, 3
        elif depths[1] == 4 and depths[2] == 23:
            blk2, blk3 = 2, 3
        elif depths[1] == 8 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 24 and depths[2] == 36:
            blk2, blk3 = 4, 4
        elif depths[1] == 30 and depths[2] == 48:
            blk2, blk3 = 5, 6
        else:
            raise NotImplementedError

        N2, N3 = math.ceil(depths[1] / blk2 -
                           1e-5), math.ceil(depths[2] / blk3 - 1e-5)
        N = 2 + N2 + N3  # r50: 2 + 2 + 2 = 6
        max_layer_id = N + 1  # r50: 2 + 2 + 2 + 1(like head) = 7

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id, max_layer_id + 1

        if param_name.startswith('backbone.layer'):
            stage_id = int(param_name.split('.')[1][5:])
            block_id = int(param_name.split('.')[2])

            if stage_id == 1:
                layer_id = 1
            elif stage_id == 2:
                layer_id = 2 + block_id // blk2  # r50: 2, 3
            elif stage_id == 3:
                layer_id = 2 + N2 + block_id // blk3  # r50: 4, 5
            else:  # stage_id == 4
                layer_id = N  # r50: 6
            return layer_id, max_layer_id + 1

        else:
            return 0, max_layer_id + 1

