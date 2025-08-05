from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement


class BaseClassifier(BaseModel, metaclass=ABCMeta):
    """分类器的基类。

    参数:
        init_cfg (dict, optional): 初始化配置字典。默认为 None。
        data_preprocessor (dict, optional): 用于预处理输入数据的配置。如果为 None，将使用
            "BaseDataPreprocessor" 作为默认类型，详情请参考
            :class:`mmengine.model.BaseDataPreprocessor`。默认为 None。

    属性:
        init_cfg (dict): 初始化配置字典。
        data_preprocessor (:obj:`mmengine.model.BaseDataPreprocessor`): 一个额外的数据预处理模块，
            用于将从数据加载器中获取的数据处理为 :meth:`forward` 方法可接受的格式。
    """

    def __init__(self,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None):
        super(BaseClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor'):
        """训练和测试中统一的前向过程入口。

        该方法应支持以下三种模式："tensor"、"predict" 和 "loss"：

        - "tensor"：前向传播整个网络，返回张量或张量元组，不进行任何后处理，与普通的 nn.Module 类似。
        - "predict"：前向传播并返回预测结果，结果经过完整处理，返回一个 :obj:`BaseDataElement` 的列表。
        - "loss"：前向传播并根据输入和数据样本返回一个包含损失的字典。

        注意：此方法不处理反向传播和优化器更新，这些操作在 :meth:`train_step` 中完成。

        参数:
            inputs (torch.Tensor): 输入张量，通常形状为 (N, C, ...)。
            data_samples (List[BaseDataElement], optional): 每个样本的标注数据。如果 ``mode="loss"``，
                则此参数是必需的。默认为 None。
            mode (str): 指定返回值的类型。默认为 'tensor'。

        返回:
            返回值的类型取决于 ``mode`` 的值：

            - 如果 ``mode="tensor"``, 返回一个张量或张量元组。
            - 如果 ``mode="predict"``, 返回一个 :obj:`mmengine.BaseDataElement` 的列表。
            - 如果 ``mode="loss"``, 返回一个包含张量的字典。
        """
        pass

    def extract_feat(self, inputs: torch.Tensor):
        """从输入张量中提取特征，输入张量的形状为 (N, C, ...)。

        子类建议实现此方法以从主干网络（backbone）和颈部网络（neck）中提取特征。

        参数:
            inputs (torch.Tensor): 一批输入张量。其形状应为
                ``(num_samples, num_channels, *img_shape)``。
        """
        raise NotImplementedError

    # def extract_feats(self, multi_inputs: Sequence[torch.Tensor],
    #                   **kwargs) -> list:
    #     """Extract features from a sequence of input tensor.

    #     Args:
    #         multi_inputs (Sequence[torch.Tensor]): A sequence of input
    #             tensor. It can be used in augmented inference.
    #         **kwargs: Other keyword arguments accepted by :meth:`extract_feat`.

    #     Returns:
    #         list: Features of every input tensor.
    #     """
    #     assert isinstance(multi_inputs, Sequence), \
    #         '`extract_feats` is used for a sequence of inputs tensor. If you '\
    #         'want to extract on single inputs tensor, use `extract_feat`.'
    #     return [self.extract_feat(inputs, **kwargs) for inputs in multi_inputs]
