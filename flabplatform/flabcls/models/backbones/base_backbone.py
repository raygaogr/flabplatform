from abc import ABCMeta, abstractmethod

from mmengine.model import BaseModule


class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """基础骨干网络类。

    该类定义了骨干网络的基本功能。任何继承此类的骨干网络至少需要定义自己的 `forward` 方法。
    """
    def __init__(self, init_cfg=None):
        super(BaseBackbone, self).__init__(init_cfg)

    @abstractmethod
    def forward(self, x):
        """前向计算。

        参数:
            x (tensor | tuple[tensor]): 输入数据，可以是一个 Torch.tensor 或一个包含多个 Torch.tensor 的元组，
                用于前向计算。

        返回:
            具体返回值由子类实现决定。
        """
        pass

    def train(self, mode=True):
        """在前向计算之前设置模块状态。

        参数:
            mode (bool): 指定当前模式是训练模式还是测试模式。
        """
        super(BaseBackbone, self).train(mode)
