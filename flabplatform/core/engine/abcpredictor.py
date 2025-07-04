

from abc import ABCMeta, abstractmethod

class ABCPredictor(metaclass=ABCMeta):
    """
    预测器组件的抽象基类，定义了标准预测流程的接口。
    
    此类作为所有预测器实现的基础，确保它们遵循统一的接口设计，
    包括数据预处理、模型推理、结果后处理和保存功能。设计用于
    在 flabplatform 框架中实现可互换的预测组件。
    
    属性:
        save_dir: 结果保存目录
        model: 用于预测的模型实例
        dataset: 用于预测的数据集

    示例：
    class CustomImagePredictor(ABCPredictor):
        def __init__(self, model=None, conf_threshold=0.5, save_dir='./results'):
            super().__init__()
            self.model = model
            self.conf_threshold = conf_threshold
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True, parents=True)
            self.results = []
            
        def preprocess(self, image):
            if isinstance(image, str):
                # 如果输入是路径，读取图像
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # 调整大小
            resized = cv2.resize(image, (640, 640))
            
            # 标准化
            normalized = resized / 255.0
            
            # 转换为tensor
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
            
            return tensor, image
        
        def postprocess(self, preds, img, orig_imgs):
            # 假设preds是bbox格式 [x1, y1, x2, y2, conf, class_id]
            processed_results = []
            
            for pred in preds:
                # 应用置信度阈值
                keep = pred[:, 4] > self.conf_threshold
                filtered_pred = pred[keep]
                
                # 转换坐标到原始图像尺寸
                if len(filtered_pred):
                    h, w = orig_imgs.shape[:2]
                    scale_w = w / img.shape[-1]
                    scale_h = h / img.shape[-2]
                    
                    # 调整坐标
                    filtered_pred[:, 0] *= scale_w
                    filtered_pred[:, 1] *= scale_h
                    filtered_pred[:, 2] *= scale_w
                    filtered_pred[:, 3] *= scale_h
                    
                    processed_results.append(filtered_pred)
            
            return processed_results
        
        def __call__(self, source=None, model=None):
            if model is not None:
                self.model = model
                
            if self.model is None:
                raise ValueError("No model provided for prediction")
                
            if isinstance(source, (str, list)):
                # 处理单个图像或图像列表
                results = []
                if isinstance(source, str):
                    source = [source]
                    
                for img_path in source:
                    # 预处理
                    tensor_img, orig_img = self.preprocess(img_path)
                    
                    # 模型推理
                    with torch.no_grad():
                        preds = self.model(tensor_img)
                    
                    # 后处理
                    processed = self.postprocess(preds, tensor_img, orig_img)
                    
                    # 保存结果
                    result = {
                        'path': img_path,
                        'predictions': processed,
                        'original_image': orig_img
                    }
                    results.append(result)
                    
                self.results = results
                return results
            else:
                raise ValueError("Source must be a path or list of paths")
    """
    def __init__(self):
        super().__init__()
        self.save_dir = None
        self.model = None
        self.dataset = None

    @abstractmethod
    def preprocess(self, image):
        """
        对输入图像进行预处理，准备用于模型推理。
        
        此方法应实现所有必要的预处理步骤，如调整大小、标准化、
        数据格式转换等，以确保输入符合模型的要求。
        
        参数:
            image: 原始输入图像，可以是各种格式（PIL图像、NumPy数组、文件路径等）
        
        返回:
            处理后的图像，通常为tensor或模型期望的格式
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Preprocess method not implemented in predictor")


    @abstractmethod
    def postprocess(self, preds):
        """
        对模型预测结果进行后处理。
        
        此方法应实现所有必要的后处理步骤，如置信度过滤、非极大值抑制、
        坐标转换等，将原始预测转换为最终可用的结果格式。
        
        参数:
            preds: 模型的原始预测输出
        
        返回:
            后处理完成的预测结果，格式应根据具体应用场景定义
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError(
            "Postprocess method not implemented in predictor. "
            "Please implement this method to process the model's predictions."
        )


    @abstractmethod
    def __call__(self, source):
        """
        执行预测过程。
        
        此方法是预测器的主要入口点，应实现完整的预测流程，包括
        数据加载、预处理、模型推理、后处理和结果收集。
        
        参数:
            source: 预测数据源，可以是图像路径、URL、数组或数据集
            model: 用于预测的模型，如果为None则使用预设模型
        
        返回:
            预测结果，格式应根据具体应用场景定义
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """       
        raise NotImplementedError(
            "The __call__ method is not implemented in the predictor. "
            "Please implement this method to run the predictor with the given source and model."
        )

