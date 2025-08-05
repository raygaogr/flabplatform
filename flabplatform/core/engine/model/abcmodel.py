
from abc import abstractmethod
import torch


class ABCModel(torch.nn.Module):
    """
    抽象基类，定义了所有模型必须实现的核心接口。

    该类提供了一个标准化的接口，确保所有派生模型类都实现必要的功能，
    如前向传播、预测、权重加载和损失计算。设计用于在 flabplatform
    框架中创建一致的模型行为。

    继承:
        torch.nn.Module: PyTorch 的基础模块类

    示例:
        class MyCustomModel(ABCModel):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self.classifier = nn.Linear(64 * 112 * 112, num_classes)
                self.criterion = None
                self.init_criterion()
                
            def forward(self, x, *args, **kwargs):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
            
            def predict(self, x):
                self.eval()  # 设置为评估模式
                with torch.no_grad():
                    output = self.forward(x)
                    predictions = torch.softmax(output, dim=1)
                return predictions
            
            def load(self, weights):
                if isinstance(weights, dict):
                    self.load_state_dict(weights)
                elif isinstance(weights, torch.nn.Module):
                    self.load_state_dict(weights.state_dict())
                else:
                    raise TypeError("weights must be a dict or torch.nn.Module")
                
            def loss(self, batch, preds=None):
                x, targets = batch
                if preds is None:
                    preds = self.forward(x)
                return self.criterion(preds, targets)
            
            def init_criterion(self):
                self.criterion = nn.CrossEntropyLoss()
                return self.criterion
    """
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """
        执行模型的前向传播。

        此方法定义了模型如何处理输入数据并生成输出。作为抽象方法，
        它必须由所有子类实现。

        参数:
            x (torch.Tensor): 输入张量
            *args: 额外的位置参数
            **kwargs: 额外的关键字参数

        返回:
            torch.Tensor: 模型的输出

        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            f"forward() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to perform a forward pass through your model."
        )

    @abstractmethod
    def predict(self, x):
        """
        使用模型执行预测。

        与 forward 方法不同，predict 专门用于推理阶段，可能包含后处理步骤，
        如概率计算、阈值应用等。

        参数:
            x (torch.Tensor): 用于预测的输入张量

        返回:
            torch.Tensor: 模型的预测结果

        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            f"predict() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to perform inference with your model."
        )

    @abstractmethod
    def load(self, weights):
        """
        将预训练权重加载到模型中。

        此方法用于将保存的权重加载到模型，支持从字典或另一个模型实例加载。

        参数:
            weights (dict | torch.nn.Module): 要加载的预训练权重

        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            f"load() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to load pre-trained weights into your model."
        )

    @abstractmethod
    def loss(self, batch, preds=None):
        """
        计算模型的损失。

        此方法接收一个数据批次并计算相应的损失值，用于模型训练。

        参数:
            batch (dict): 用于计算损失的数据批次
            preds (torch.Tensor | List[torch.Tensor], optional): 模型预测结果，
                如果为 None，则方法应先调用 forward 获取预测结果

        返回:
            torch.Tensor: 计算得到的损失值

        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            f"loss() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to compute the loss for your model."
        )
    
    @abstractmethod
    def init_criterion(self):
        """
        初始化模型的损失函数。
        
        此方法用于设置和初始化模型使用的损失函数或损失函数组合。
        
        返回:
            任意: 损失函数或损失函数组合
        
        异常:
            NotImplementedError: 如果子类未实现此方法
        """
        raise NotImplementedError(
            f"init_criterion() function not implemented in {self.__class__.__name__}. "
            "Please implement this method to initialize the loss function for your model."
        )
