import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import yaml
import logging
logging.basicConfig(level=logging.INFO, filename='./classification_onnx_inference.log',
                    format='%(asctime)s[%(levelname)s]: %(message)s')
logger = logging.getLogger()


def softmax(logits):
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class ONNXModelInference:
    def __init__(self, model_path, labels, input_size=(224, 224), mean=None, std=None, provider="CUDAExecutionProvider"):
        self.model_path = model_path
        self.labels = labels
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32) if mean is not None else np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std, dtype=np.float32) if std is not None else np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.provider = provider
        
        self.session = ort.InferenceSession(self.model_path, providers=[self.provider])
        logger.info(f"ONNX Runtime Session Providers: {self.session.get_providers()}")

    def preprocess(self, img_paths):
        processed_images = []
        
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_size)

            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            
            processed_images.append(img)
        batch_images = np.vstack(processed_images)
        logger.info(f"Processed batch shape: {batch_images.shape}")
        return batch_images

    def predict(self, img_paths):
        batch_images = self.preprocess(img_paths)
        outputs = self.session.run(None, {"inputs": batch_images})
        logits = outputs[0]

        probs = softmax(logits)
        pred_indices  = np.argmax(probs, axis=1)
        
        # pred_labels = [self.labels[idx] for idx in pred_indices]

        return pred_indices, probs


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="ONNX Model Inference")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file")
    parser.add_argument('--image_paths', type=str, nargs='+', required=True, help="List of image paths for inference")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    labels = config.get('labels', [])
    if not labels:
        raise ValueError("Error: No labels found in the configuration file.")

    # 初始化模型推理类
    onnx_inference = ONNXModelInference(
        model_path=config['model_path'],
        labels=labels,
        input_size=tuple(config['input_size']),
        mean=config['mean'],
        std=config['std'],
        provider=config['provider']
    )
    
    # 进行批量推理并输出结果
    pred_classes, probs = onnx_inference.predict(args.image_paths)
    
    for i, pred_class in enumerate(pred_classes):
        print(f"Image {args.image_paths[i]} Predicted Class: {pred_class}")
        print(f"Softmax Probability: {probs[i][pred_class]}")

    # 输出推理结果
    # for i, (label, prob) in enumerate(zip(pred_labels, probs)):
    #     print(f"Image: {args.image_paths[i]} | Predicted Label: {label} | Confidence: {prob.max():.4f}")
    #     logger.info(f"Image: {args.image_paths[i]} | Predicted Label: {label} | Confidence: {prob.max():.4f}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=5 python onnx_inference.py --config classification_onnx_inference.yaml --image_paths /root/dataset/tiny-imagenet-200/val/n01443537/val_1551.JPEG
# CUDA_VISIBLE_DEVICES=5 python onnx_inference.py --config classification_onnx_inference.yaml --image_paths /root/dataset/tiny-imagenet-200/val/n01443537/val_1551.JPEG /root/dataset/tiny-imagenet-200/val/n01443537/val_1267.JPEG
