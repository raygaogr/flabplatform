from abc import ABCMeta, abstractmethod

class ABCTrainer(metaclass=ABCMeta):
    """
    模型训练器的抽象基类，定义了标准训练流程的接口。
    
    此类作为所有训练器实现的基础，确保它们遵循统一的接口设计，
    包括数据加载、模型配置、训练循环、验证、检查点管理等功能。
    设计用于在 flabplatform 框架中实现可互换的训练组件。
    
    属性:
        model: 要训练的模型实例
        validator: 用于模型验证的验证器实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例
    
    示例：
    class SimpleTrainer(ABCTrainer):
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.config = config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.save_dir = Path(config['save_dir'])
            self.save_dir.mkdir(exist_ok=True, parents=True)
            self.epoch = 0
            self.best_accuracy = 0
        
        def train(self):
            # 准备训练组件
            train_loader = self.build_dataloader(self.config['train_data'])
            self.validator = self.build_validator()
            self.optimizer = self.build_optimizer(lr=self.config['learning_rate'])
            self.scheduler = self.build_scheduler()
            
            # 训练循环
            for epoch in range(self.config['epochs']):
                self.epoch = epoch
                self.model.train()
                
                # 训练一个epoch
                for batch_idx, batch in enumerate(train_loader):
                    batch = self.preprocess(batch)
                    inputs, targets = batch
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    
                    if batch_idx % 50 == 0:
                        print(f"Epoch: {epoch}/{self.config['epochs']}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
                
                # 验证模型
                metrics = self.validator(self.model)
                print(f"Validation metrics: {metrics}")
                
                # 更新学习率
                self.scheduler.step(metrics['loss'])
                
                # 保存模型和指标
                self.save_metrics(metrics)
                self.save_model()
            
            return {"best_accuracy": self.best_accuracy}
        
        def build_dataloader(self, dataset_path, batch_size=None, rank=0, mode="train"):
            if batch_size is None:
                batch_size = self.config['batch_size']
            
            # 简化的数据加载器创建
            from torch.utils.data import DataLoader
            dataset = torch.utils.data.TensorDataset(
                torch.randn(100, 3, 224, 224),  # 模拟图像
                torch.randint(0, 10, (100,))    # 模拟标签
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))
        
        def build_validator(self):
            val_loader = self.build_dataloader(self.config['val_data'], mode="val")
            return validator
        
        def build_optimizer(self, name="adam", lr=0.001, momentum=0.9, decay=1e-5):
            if name == "adam":
                return optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
            else:
                return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
        
        def build_scheduler(self):
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=5
            )
        
        def resume_training(self, ckpt):
            checkpoint = torch.load(ckpt)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch']
            self.best_accuracy = checkpoint['best_accuracy']
            return True
        
        def preprocess(self, batch):
            inputs, targets = batch
            return inputs.to(self.device), targets.to(self.device)
        
        def save_metrics(self, metrics):
            with open(self.save_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f)
        
        def save_model(self):
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'best_accuracy': self.best_accuracy
            }
            
            # 保存最新模型
            torch.save(checkpoint, self.save_dir / 'last.pt')
            
            # 如果是最佳模型，单独保存
            if self.best_accuracy == checkpoint['best_accuracy'] and self.epoch > 0:
                torch.save(checkpoint, self.save_dir / 'best.pt')
            
            return str(self.save_dir / 'last.pt')
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.validator = None
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def train(self):
        """
        执行模型训练流程。
        
        此方法应实现完整的训练循环，包括数据遍历、前向传播、
        损失计算、反向传播、优化器步骤、学习率调整、验证和检查点保存等。
        
        返回:
            训练结果，可能包括损失历史、指标等
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("train function not implemented in trainer")

    @abstractmethod
    def build_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        为给定数据集构建数据加载器。
        
        此方法应实现数据集加载和数据加载器配置，包括数据增强、
        批处理、多进程加载等设置。
        
        参数:
            dataset_path (str): 数据集路径
            batch_size (int, optional): 批大小，默认为16
            rank (int, optional): 分布式训练中的进程等级，默认为0
            mode (str, optional): 数据加载模式，如'train'、'val'，默认为'train'
        
        返回:
            数据加载器实例
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("get_dataloader function not implemented in trainer")
    
    @abstractmethod
    def build_validator(self):
        """
        构建模型验证器。
        
        此方法应实现验证器的创建和配置，用于评估模型在验证集上的性能。
        
        返回:
            验证器实例
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("get_validator function not implemented in trainer")
    
    @abstractmethod
    def build_optimizer(self, name="auto", lr=0.001, momentum=0.9, decay=1e-5):
        """
        构建优化器。
        
        此方法应实现优化器的创建和配置，支持不同的优化算法和参数设置。
        
        参数:
            name (str, optional): 优化器名称，如'SGD'、'Adam'或'auto'，默认为'auto'
            lr (float, optional): 学习率，默认为0.001
            momentum (float, optional): 动量参数，默认为0.9
            decay (float, optional): 权重衰减系数，默认为1e-5
        
        返回:
            优化器实例
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            "build_optimizer function not implemented in trainer. "
            "Please implement this method to build the optimizer for your model."
        )
    
    @abstractmethod
    def build_scheduler(self):
        """
        构建学习率调度器。
        
        此方法应实现学习率调度器的创建和配置，用于在训练过程中
        动态调整学习率。
        
        返回:
            学习率调度器实例
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("build_scheduler function not implemented in trainer")

    @abstractmethod
    def resume_training(self, ckpt):
        """
        从检查点恢复训练。
        
        此方法应实现从保存的检查点恢复训练状态的功能，包括模型权重、
        优化器状态、学习率调度器状态和训练迭代计数等。
        
        参数:
            ckpt: 检查点路径或检查点对象
        
        返回:
            恢复的训练状态或成功标志
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        return NotImplementedError(
            "resume_training function not implemented in trainer. "
            "Please implement this method to resume training from a checkpoint."
        )
    
    @abstractmethod
    def preprocess(self, batch):
        """
        对输入批次进行预处理。
        
        此方法应实现对模型输入和标签的预处理步骤，以满足模型
        的输入要求。
        
        参数:
            batch: 输入数据批次
        
        返回:
            预处理后的批次
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        return NotImplementedError(
            "preprocess_batch function not implemented in trainer. "
            "Please implement this method to preprocess the batch for your model."
        )
    
    @abstractmethod
    def save_metrics(self, metrics):
        """
        保存训练指标。
        
        此方法应实现将训练过程中收集的指标保存到文件的功能，
        通常是JSON格式。
        
        参数:
            metrics (dict): 要保存的指标字典
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("save_metrics function not implemented in trainer")

    @abstractmethod
    def save_model(self):
        """
        保存模型训练检查点和附加元数据。
        
        此方法应实现模型权重和训练状态的保存逻辑，包括：
        - 当前模型权重
        - 最佳模型权重
        - 优化器状态
        - 学习率调度器状态
        - 训练轮次信息
        - 训练指标历史
        - 其他有助于恢复训练的元数据
        
        实现应考虑不同的保存策略，如保存最近的检查点、最佳模型、
        定期保存等。还应考虑文件命名约定和版本控制。
        
        返回:
            保存的模型路径或包含各种保存路径的字典
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("save_model function not implemented in trainer")