"""
Copyright (C) 2025 dsl.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import torchmetrics
from pathlib import Path
from torch import Tensor
import cv2
from PIL import Image
from torchvision import transforms
from .eval_category import ClassificationEvaluator

import os
import openvino as ov
import nncf

from torch.utils.data import DataLoader
# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ['Classifier']


class Classifier(LightningModule):
    """
    A Lightning Module for image classification tasks.
    
    This module provides a flexible and user-friendly interface for image classification
    with support for both single-label and multi-label classification. It includes
    features like model export, visualization tools, and comprehensive evaluation metrics.
    
    Args:
        num_classes (int): Number of output classes
        backbone (str): Name of the timm backbone model. Defaults to 'resnet18'
        img_c (int): Number of input image channels. Defaults to 3
        lr (float): Initial learning rate. Defaults to 0.001
        pretrained (bool): Whether to use pretrained weights. Defaults to True
        multi_label (bool): Whether to use multi-label classification. Defaults to False
        fcn_head (bool): Whether to use fully convolutional head. Defaults to False
        dropout (float): Dropout probability. Defaults to 0.5
        **kwargs: Additional arguments passed to parent class
    
    Example:
        >>> # Single-label classification with ResNet18
        >>> model = Classifier(num_classes=10, backbone='resnet18')
        >>> 
        >>> # Multi-label classification with EfficientNet and custom dropout
        >>> model = Classifier(
        ...     num_classes=5,
        ...     backbone='efficientnet_b0',
        ...     multi_label=True,
        ...     dropout=0.3
        ... )
    """
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet18',
        img_c: int = 3,
        lr: float = 0.001,
        pretrained: bool = True,
        multi_label: bool = False,
        fcn_head: bool = False,
        dropout: float = 0.5,
        **kwargs: Any
    ):
        super().__init__()

        # Validate inputs
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if img_c < 1:
            raise ValueError("img_c must be positive")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        self.hparams.update({
            'num_classes': num_classes, 
            'backbone': backbone, 
            'img_c': img_c, 
            'lr': lr, 
            'pretrained': pretrained, 
            'multi_label': multi_label, 
            'fcn_head': fcn_head,
            'dropout': dropout
        })
        logger.info(f"Classifier params: {self.hparams}")
        self.build_model(backbone, img_c, num_classes, fcn_head, dropout)
        
        # Initialize metrics
        task = "multilabel" if multi_label else "multiclass"
        metric_kwargs = {
            "task": task,
            "num_classes": num_classes,
            "average": 'micro' if multi_label else 'macro'
        }
        self.train_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.test_acc = torchmetrics.Accuracy(**metric_kwargs)
        
        # Initialize validation outputs list
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def build_model(
        self, 
        backbone: str, 
        img_c: int, 
        num_classes: int, 
        fcn_head: bool,
        dropout: float
    ) -> None:
        """
        Build the model architecture with specified backbone and head.
        
        Args:
            backbone (str): Name of the timm backbone model
            img_c (int): Number of input image channels
            num_classes (int): Number of output classes
            fcn_head (bool): Whether to use fully convolutional head
            dropout (float): Dropout probability
            
        Raises:
            Exception: If there is an error building the model components
        """
        from .backbones import TimmBackbone
        from .heads import num_features_model, create_base_head, create_fcn_head
        
        try:
            self.backbone = TimmBackbone(backbone, img_c, pretrained=self.hparams.pretrained)
            num_features = num_features_model(self.backbone)
            
            self.head = (create_fcn_head if fcn_head else create_base_head)(
                num_features=num_features,
                num_classes=num_classes,
                ps=dropout
            )
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Model predictions
        """

        # if self.training:
        #     self.backbone.train()
        # else:
        #     self.backbone.eval()

        features = self.backbone(x)
        return self.head(features)

    def loss_fn(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Calculate the loss based on model predictions and ground truth.
        
        Args:
            logits: Model predictions before activation
            labels: Ground truth labels (can be one-hot encoded for label smoothing)
            
        Returns:
            Tensor: Calculated loss value
        """
        if self.hparams.multi_label:
            # Multi-label case
            if labels.dim() == 1:
                labels = F.one_hot(labels, num_classes=self.hparams.num_classes).float()
            labels = labels.type(torch.float32)
            return F.binary_cross_entropy_with_logits(logits, labels)
        else:
            # Single-label case
            if labels.dim() == 2:  # 如果是平滑后的one-hot标签
                log_probs = F.log_softmax(logits, dim=-1)
                return -(labels * log_probs).sum(dim=-1).mean()
            return F.cross_entropy(logits, labels)

    def training_step(self, batch: Union[Tensor, Tuple[Tensor, ...]], batch_idx: int) -> Dict[str, Tensor]:
        """
        Training step for the model.
        
        Args:
            batch: Input batch, can be either:
                   - (images, labels) tuple for standard training
                   - (mixed_images, labels_a, labels_b, lam) tuple for mixup training
            batch_idx: Index of the current batch
            
        Returns:
            Dict containing loss and other metrics
        """
        # Handle mixup batch
        if len(batch) == 4:  # mixup batch
            x, y_a, y_b, lam = batch
            y_hat = self(x)
            
            loss = lam * self.loss_fn(y_hat, y_a) + (1 - lam) * self.loss_fn(y_hat, y_b)

            # For accuracy calculation, use a weighted combination of both labels
            if self.hparams.multi_label:
                preds = torch.sigmoid(y_hat) 
                mixed_labels = lam * y_a + (1 - lam) * y_b
                self.train_acc(preds, mixed_labels)
            else:
                preds = F.softmax(y_hat, dim=-1)
                # For single-label, we can't directly mix labels, so we calculate accuracy
                # for both and take weighted average
                acc_a = self.train_acc(preds, y_a)
                acc_b = self.train_acc(preds, y_b)
                acc = lam * acc_a + (1 - lam) * acc_b
                
                # For accuracy calculation, use the primary labels
                self.train_acc(F.softmax(y_hat, dim=-1), y_a)
        else:  # standard batch
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

            if self.hparams.multi_label:
                acc = self.train_acc(torch.sigmoid(y_hat), y)
            else:
                acc = self.train_acc(F.softmax(y_hat, dim=-1), y)

        # Log metrics
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        return {'loss': loss}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Validation step logic.
        
        Args:
            batch (Tuple[Tensor, Tensor]): Input batch containing (images, labels)
            batch_idx (int): Index of the current batch
            
        Returns:
            Dict[str, Tensor]: Dictionary containing the validation loss
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.softmax(logits, dim=1) if not self.hparams.multi_label else torch.sigmoid(logits)
            acc = self.val_acc(preds, y)
        
        # Log metrics
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, prog_bar=True)
        
        self.validation_step_outputs.append({'val_loss': loss})
        return {'val_loss': loss}

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation to compute epoch-level metrics.
        https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        """
        # Calculate mean validation loss
        val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # Get final accuracy
        val_acc = self.val_acc.compute()
        
        # Log epoch-level metrics
        self.log_dict({
            'val_loss_epoch': val_loss,
            'val_acc_epoch': val_acc
        }, prog_bar=True)
        
        # Reset metrics and clear outputs
        self.val_acc.reset()
        self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Test step logic.
        
        Args:
            batch (Tuple[Tensor, Tensor]): Input batch containing (images, labels)
            batch_idx (int): Index of the current batch
            
        Returns:
            Dict[str, Tensor]: Dictionary containing the test loss
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.softmax(logits, dim=1) if not self.hparams.multi_label else torch.sigmoid(logits)
            acc = self.test_acc(preds, y)
        
        # Log metrics
        self.log_dict({
            'test_loss': loss,
            'test_acc': acc
        }, prog_bar=True)
        
        return {'test_loss': loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer configuration
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        **kwargs: Any
    ) -> 'Classifier':
        """
        Load model from checkpoint file.
        
        Args:
            checkpoint_path (Union[str, Path]): Path to the checkpoint file
            map_location: PyTorch map_location argument for loading the checkpoint
            **kwargs: Additional arguments to override loaded hyperparameters
            
        Returns:
            Classifier: Loaded model instance
            
        Raises:
            FileNotFoundError: If checkpoint file does not exist
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        return super().load_from_checkpoint(checkpoint_path, map_location=map_location, pretrained=False, **kwargs)

    def export_onnx(self, onnx_path: Union[str, Path], input_shape: Tuple[int, ...] = (1, 3, 224, 224), **kwargs):
        """
        onnx.export only support onnx opset up to version 18. 
        torch: https://pytorch.org/docs/stable/onnx.html
        onnxruntime: https://onnxruntime.ai/docs/get-started/with-python.html
        
        Args:
            save_path (Union[str, Path]): Path to save the ONNX model
            input_shape (Tuple[int, ...]): Input tensor shape (batch_size, channels, height, width)
            
        Raises:
            RuntimeError: If export fails
        """
        try:
            self.eval()
            input_sample = torch.randn(input_shape).to(self.device)
            kwargs['opset_version'] = kwargs.get('opset_version', 18)
            kwargs['export_params'] = kwargs.get('export_params', True)
            kwargs['input_names'] = kwargs.get('input_names', ['inputs'])
            kwargs['output_names'] = kwargs.get('output_names', ['outputs'])
            kwargs['dynamic_axes'] = kwargs.get('dynamic_axes', {'inputs': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}})
            self.cpu()
            self.to_onnx(onnx_path, input_sample=input_sample, **kwargs)
            self.train()
            logger.info(f"Model exported to ONNX format: {onnx_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {str(e)}")
            raise

    def export_openvino(self,openvino_path: Union[str, Path], input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                        quantization_type:str='float32',data_loader:Union[None,DataLoader]=None, **kwargs):
        '''
        Args:
            save_path (Union[str, Path]): Path to save the ONNX model
            input_shape (Tuple[int, ...]): Input tensor shape (batch_size, channels, height, width)
            quantization_type:str='float32',['float32','float16','int8']
        Raises:
            RuntimeError: If export fails
        '''
        def transform_fn(data_item):
            images, _ = data_item
            return images
        assert quantization_type in ['float32','float16','int8'],f"quantization_type should in ['float32','float16','int8']."
        if quantization_type=='int8' and not isinstance(data_loader,DataLoader):
            raise ValueError(f'data_loader must be an instance of DataLoader')
        try:
            export_model = nn.Sequential(
                self.backbone,
                self.head,
            )
            export_model.eval()
            export_model.to('cpu')
            input_data = torch.rand(input_shape,device='cpu')
            ov_model = ov.convert_model(export_model, example_input=input_data,)
            if quantization_type=='int8':
                calibration_dataset = nncf.Dataset(data_loader, transform_fn)
                ov_model = nncf.quantize(ov_model, calibration_dataset)
            ov.save_model(ov_model, openvino_path,
                        compress_to_fp16=quantization_type=='float16')
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {str(e)}")
            raise

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        Make predictions on input data.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Model predictions after softmax/sigmoid activation
        """
        self.eval()
        x = x.to(self.device)
        logits = self.forward(x)
        
        if self.hparams.multi_label:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Get probability predictions for input data.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Probability predictions
        """
        return self.predict(x)

    @torch.no_grad()
    def predict_classes(self, x: Tensor) -> Tensor:
        """
        Get class predictions for input data.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tensor: Class predictions (indices)
        """
        probs = self.predict(x)
        if self.hparams.multi_label:
            return (probs > 0.5).float()
        return probs.argmax(dim=1)

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader, labels=None) -> Dict[str, float]:
        """Evaluate model performance using the Evaluator class."""
        evaluator = ClassificationEvaluator(
            num_classes=self.hparams.num_classes,
            multi_label=self.hparams.multi_label,
            labels=labels
        )
        return evaluator.evaluate(self, dataloader, device=self.device)

    @torch.no_grad()
    def single_inference(
        self,
        image: Union[str, np.ndarray],
        input_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        device: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform inference on a single image.
        
        Args:
            image: Image path or numpy array in BGR format
            input_size: Model input size (height, width)
            mean: Normalization mean values for RGB channels
            std: Normalization standard deviation values for RGB channels
            device: Device to run inference on ('cuda' or 'cpu'). If None, uses the model's current device
            
        Returns:
            Tuple of (predicted labels, prediction confidences)
        """
        if device is None:
            device = self.device
        
        self.to(device)
        self.eval()
        
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        # Preprocess image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Image at path {image} could not be loaded.")
        else:
            img = image.copy()
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Transform image
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
        # Run inference
        with torch.no_grad():
            outputs = self.forward(img_tensor)
        if self.hparams.multi_label:
            y_pred = torch.sigmoid(outputs)
        else:
            y_pred = torch.softmax(outputs, dim=-1)
            
        y_pred = y_pred.cpu().numpy()
        y_pred_labels = y_pred.argmax(axis=-1)
        return y_pred_labels, y_pred
