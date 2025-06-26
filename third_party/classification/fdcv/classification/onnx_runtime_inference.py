"""
ONNX Runtime inference module for various computer vision tasks.
"""
import tqdm

from enum import Enum
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from typing import Tuple, Union, List, Optional, Dict, Any
import os
from glob import glob
from .eval_category import ClassificationEvaluator, SegmentationEvaluator, DetectionEvaluator
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


__all__ = ['ONNXRuntimeInference']

class TaskType(Enum):
    """Supported task types for ONNX Runtime inference."""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    
class ONNXRuntimeInference:
    """Base class for ONNX Runtime inference across different computer vision tasks."""
    
    def __init__(self, model_path: str, 
                 task: Union[str, TaskType], 
                 input_size: Tuple[int, int] = (224, 224),
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 providers: Optional[List[str]] = None,
                 **kwargs: Any):
        """
        Initialize ONNX Runtime inference.
        
        Args:
            model_path: Path to ONNX model file
            input_size: Model input size (height, width)
            mean: Normalization mean values for RGB channels
            std: Normalization standard deviation values for RGB channels
            providers: List of execution providers. If None, uses all available providers
            **kwargs: Additional task-specific arguments
        """
        if isinstance(task, str):
            try:
                self.task = TaskType(task.lower())
            except ValueError:
                raise ValueError(f"Task must be one of {[t.value for t in TaskType]}")
        else:
            self.task = task
            
        if providers is None:
            providers = ort.get_available_providers()

        self.session = ort.InferenceSession(model_path, providers=providers)
        logger.info(f"Running on device: {self.session.get_providers()}")

        self.input_name = self.session.get_inputs()[0].name
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Add task-specific attributes
        if self.task == TaskType.CLASSIFICATION:
            self.num_classes = kwargs.get('num_classes')
            self.multi_label = kwargs.get('multi_label', False)
            if self.num_classes is None:
                raise ValueError("num_classes must be specified for classification task")

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """
        Apply softmax to logits.

        Args:
            logits: Input logits

        Returns:
            Softmax probabilities
        """
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def preprocess_image(self, image: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> np.ndarray:
        """
        Preprocess image or batch of images for inference.
        
        Args:
            image: Single image (path or numpy array in BGR format) or list of images
            
        Returns:
            Preprocessed image array with batch dimension
        """
        if isinstance(image, (str, np.ndarray)):
            images = [image]
        else:
            images = image

        processed_images = []
        for img in images:
            if isinstance(img, str):
                img_data = cv2.imread(img)
                if img_data is None:
                    raise ValueError(f"Image at path {img} could not be loaded.")
            else:
                img_data = img.copy()
                
            # Convert BGR to RGB
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            img_data = Image.fromarray(img_data)
            
            # Transform image
            img_tensor = self.transform(img_data)
            processed_images.append(img_tensor.numpy())
            
        return np.stack(processed_images, axis=0)

    def _inference_classification(self, image: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Run classification inference."""
        input_array = self.preprocess_image(image)
        outputs = self.session.run(None, {self.input_name: input_array})
        
        logits = outputs[0]
        if self.multi_label:
            probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
        else:
            probabilities = self.softmax(logits)
            
        pred_labels = probabilities.argmax(axis=-1)
        return pred_labels, probabilities

    def _inference_segmentation(self, image: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run segmentation inference."""
        raise NotImplementedError("Segmentation inference not yet implemented")

    def _inference_detection(self, image: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run detection inference."""
        raise NotImplementedError("Detection inference not yet implemented")

    def batch_inference(
        self,
        images: Union[str, List[Union[str, np.ndarray]]],
        batch_size: int = 16,
        image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        **kwargs: Any
    ) -> Union[Tuple[np.ndarray, List[str], np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Run batch inference on multiple images or all images in a folder.
        
        Args:
            images: Either a directory path containing images or a list of image paths/numpy arrays
            batch_size: Number of images to process in each batch
            image_extensions: Tuple of valid image file extensions (only used when images is a directory)
            **kwargs: Additional task-specific arguments
            
        Returns:
            For classification tasks:
                - labels: predicted class labels
                - image_paths: list of processed image paths (or None if input was numpy arrays)
                - probabilities: prediction probabilities
            For other tasks:
                - List of task-specific results
        """
        # Handle directory input
        if isinstance(images, str) and os.path.isdir(images):
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(glob(os.path.join(images, f'*{ext}')))
                image_paths.extend(glob(os.path.join(images, f'*{ext.upper()}')))
            
            if not image_paths:
                raise ValueError(f"No images found in {images} with extensions {image_extensions}")
                
            (f"Found {len(image_paths)} images in {images}")
            images = image_paths
        elif isinstance(images, str):
            raise ValueError("If a string is provided, it must be a directory path")
        
        # Convert single image to list
        if isinstance(images, np.ndarray):
            images = [images]
            
        num_images = len(images)
        results_labels = []
        results_probs = []
        
        for i in tqdm.tqdm(range(0, num_images, batch_size)):
            batch_images = images[i:min(i + batch_size, num_images)]
            input_batch = self.preprocess_image(batch_images)
            
            if self.task == TaskType.CLASSIFICATION:
                outputs = self.session.run(None, {self.input_name: input_batch})
                logits = outputs[0]
                
                if self.multi_label:
                    probabilities = 1 / (1 + np.exp(-logits))  # sigmoid
                else:
                    probabilities = self.softmax(logits)
                    
                pred_labels = probabilities.argmax(axis=-1)
                
                results_labels.append(pred_labels)
                results_probs.append(probabilities)
            else:
                raise NotImplementedError(f"Batch inference for {self.task.value} not yet implemented")
        
        if self.task == TaskType.CLASSIFICATION:
            labels = np.concatenate(results_labels)
            probs = np.concatenate(results_probs)
            # Return image paths only if input was a directory or list of paths
            paths = images if isinstance(images[0], str) else None
            return labels, paths, probs
        
        return []

    def __call__(
        self,
        image: Union[str, np.ndarray],
        **kwargs: Any
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
        """
        Run inference based on the task type.
        
        Args:
            image: Image path or numpy array in BGR format
            **kwargs: Additional task-specific arguments
            
        Returns:
            Task-specific inference results
        """
        if self.task == TaskType.CLASSIFICATION:
            return self._inference_classification(image, **kwargs)
        elif self.task == TaskType.SEGMENTATION:
            return self._inference_segmentation(image)
        elif self.task == TaskType.DETECTION:
            return self._inference_detection(image)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def evaluate(
        self,
        images: Union[str, List[Union[str, np.ndarray]]],
        labels: Optional[Union[List[int], np.ndarray]] = None,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model performance on a set of images.
        
        Args:
        images: Either:
            - Directory path containing images in class subdirectories
              (e.g., 'data/train' with subdirs 'cat/', 'dog/', etc.)
            - List of image paths
            - List of numpy arrays in BGR format
        labels: Ground truth labels (optional)
               Not required if images is a directory path, as labels will be
               inferred from the directory structure (subdirectory names)
        batch_size: Number of images to process at once
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
            
        Example directory structure:
            data/
            ├── cat/
            │   ├── cat1.jpg
            │   └── cat2.jpg
            ├── dog/
            │   ├── dog1.jpg
            │   └── dog2.jpg
            └── bird/
                ├── bird1.jpg
                └── bird2.jpg
        
        In this case, bird=0, cat=1, dog=2 (alphabetical order)
        """
        
        # Handle directory input
        if isinstance(images, str):
            if not os.path.isdir(images):
                raise ValueError(f"Path {images} is not a directory")
        
            subdirs = [d for d in sorted(os.listdir(images)) 
                    if os.path.isdir(os.path.join(images, d))]
            
            if not subdirs:
                raise ValueError(
                    f"No valid class subdirectories found in {images}. "
                    "Expected structure: root_dir/class_name/image_files"
                )
                
            logger.info(f"Found {len(subdirs)} classes: {', '.join(subdirs)}")
            classes_labels = subdirs
            # Get image paths and labels from directory structure
            image_paths = []
            labels = []
            for class_idx, class_name in enumerate(sorted(os.listdir(images))):
                class_dir = os.path.join(images, class_name)
                class_images = [f for f in glob(os.path.join(class_dir, "*"))
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
                if not class_images:
                    logger.warning(f"No images found in class directory: {class_name}")
                    continue
                    
                image_paths.extend(class_images)
                labels.extend([class_idx] * len(class_images))
                
                logger.info(f"Class {class_name} (index {class_idx}): "
                        f"found {len(class_images)} images")
                
            images = image_paths
            labels = np.array(labels)
        
        # Validate inputs
        if labels is None:
            raise ValueError("Labels must be provided when input is not a directory")
        if len(images) != len(labels):
            raise ValueError("Number of images and labels must match")
            
        # Process images in batches
        pred_labels, _, _ = self.batch_inference(images=images, batch_size=batch_size)
            
        y_pred = pred_labels
        y_true = np.array(labels)
        logger.info(f"classes_labels: {classes_labels}")
        # Select appropriate evaluator and compute metrics
        if self.task == TaskType.CLASSIFICATION:
            evaluator = ClassificationEvaluator(
                num_classes=self.num_classes,
                multi_label=self.multi_label,
                labels=classes_labels,
            )
        elif self.task == TaskType.SEGMENTATION:
            evaluator = SegmentationEvaluator(num_classes=self.num_classes)
        elif self.task == TaskType.DETECTION:
            evaluator = DetectionEvaluator()
            
        return evaluator._compute_metrics(y_true, y_pred)

