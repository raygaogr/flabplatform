"""
Copyright (C) 2025 dsl.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Optional, List
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import os
import os.path as osp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ['ClassificationEvaluator', 'SegmentationEvaluator', 'DetectionEvaluator']

class ModelInterface(Protocol):
    """Protocol defining the interface that models must implement for  evaluation."""
    def __call__(self, x: Any) -> Any:
        """Make predictions on input data."""
        ...

class BaseEvaluator(ABC):
    """Abstract base class for model evaluation."""

    @abstractmethod
    def evaluate(self, model: ModelInterface, dataloader: Any, device: Optional[str] = None) -> Dict[str, float]:
        """Evaluate model performance on a dataset."""
        pass

class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for classification tasks."""
    def __init__(self, num_classes: int, multi_label: bool = False, labels: Optional[List[str]] = None, save_path: Optional[str] = None):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.classes_labels = labels if labels is not None else [f'Class {i}' for i in range(num_classes)]
        self.save_path = save_path

    def evaluate(self, model: ModelInterface, dataloader: Any, device: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
        
        Args:
            model: Model that implements the ModelInterface
            dataloader: DataLoader containing the evaluation dataset
            device: Device to run evaluation on
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        logger.info("Evaluating model on validation dataset..., device: %s", device)
        all_preds = []
        all_labels = []
        # Collect predictions and labels
        for batch in dataloader:
            x, y = batch
            if hasattr(x, 'to') and device is not None:
                x = x.to(device)

            preds = model.predict(x)

            if isinstance(preds, torch.Tensor):
                if self.multi_label:
                    preds = (preds > 0.5).float()  # For multi-label classification
                else:
                    preds = preds.argmax(dim=1)  # For single-label classification
                preds = preds.cpu().numpy()
            elif isinstance(preds, np.ndarray):
                if self.multi_label:
                    preds = (preds > 0.5).astype(float)
                else:
                    preds = np.argmax(preds, axis=1)
                    
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y)

        # Concatenate all predictions and labels
        y_pred = np.array(all_preds)
        y_true = np.array(all_labels)

        return self._compute_metrics(y_true, y_pred)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics and display results."""
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Calculate hits and counts
        hits = np.diag(conf_matrix)  # Number of correct predictions per class
        pick_counts = np.sum(conf_matrix, axis=0)  # Number of predictions per class
        gt_counts = np.sum(conf_matrix, axis=1)  # Number of ground truth samples per class
        
        # Calculate overall metrics
        total_hits = np.sum(hits)
        total_samples = np.sum(conf_matrix)
        
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "Confusion Matrix": conf_matrix,
            "Total Hits": total_hits,
            "Total Samples": total_samples,
            "Per Class Hits": hits,
            "Per Class Predictions": pick_counts,
            "Per Class Ground Truth": gt_counts
        }

        # Display detailed metrics table
        self._display_metrics_table(
            metrics, 
            per_class_precision, 
            per_class_recall,
            per_class_f1,
            hits,
            pick_counts,
            gt_counts
        )
        
        # Plot confusion matrix
        # self._plot_confusion_matrix(conf_matrix)

        metrics.update({
            "per_class_precision": per_class_precision,
            "per_class_recall": per_class_recall,
            "per_class_f1": per_class_f1,
        })

        return metrics

    def _display_metrics_table(
        self,
        metrics: Dict[str, float],
        per_class_precision: np.ndarray,
        per_class_recall: np.ndarray,
        per_class_f1: np.ndarray,
        hits: np.ndarray,
        pick_counts: np.ndarray,
        gt_counts: np.ndarray
    ) -> None:
        """Display evaluation metrics in a formatted table."""
    
        # Create per-class metrics table
        data = {
            # 'Class': [f'Class {i}' for i in range(self.num_classes)] + ['Total'],
            'Class': [f'{self.classes_labels[i]}' for i in range(self.num_classes)] + ['Total'],
            'GT Count': [*gt_counts, np.sum(gt_counts)],
            'Predicted': [*pick_counts, np.sum(pick_counts)],
            'Hits': [*hits, np.sum(hits)],
            'Precision': [f"{v:.3f}" for v in per_class_precision] + [f"{metrics['Precision']:.3f}"],  
            'Recall': [f"{v:.3f}" for v in per_class_recall] + [f"{metrics['Recall']:.3f}"],   
            'F1 Score': [f"{v:.3f}" for v in per_class_f1] + [f"{metrics['F1 Score']:.3f}"],  
            'Hit Rate': [h / gt if gt else 0 for h, gt in zip(hits, gt_counts)] + [metrics['Accuracy']]  # 单个类别的acc: h / gt, 总acc: metrics['Accuracy']
        }
        df = pd.DataFrame(data)

        # Add hit rate (accuracy) for each class
        # df['Hit Rate'] = df['Hits'] / df['GT Count']
        # df['Hit Rate'] = df['Hit Rate'].fillna(0)  # Handle division by zero

        # print("\nDetailed Evaluation Metrics:")
        # print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False, floatfmt='.4f'))

        # 2. Create visual table using matplotlib
        fig, ax = plt.subplots(figsize=(10, len(df) * 0.5))
        
        # Hide axes
        ax.set_axis_off()
        
        # Create table
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f2f2f2'] * len(df.columns),  # Header color
            cellColours=[['#ffffff' if i % 2 == 0 else '#f9f9f9' for _ in range(len(df.columns))] 
                        for i in range(len(df))],  # Alternating row colors
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)  # Adjust table size
        # Add title
        plt.title('Evaluation Metrics', pad=20)
        # Adjust layout
        plt.tight_layout()

        # save png and csv
        if self.save_path:
            plt.savefig(osp.join(self.save_path, "Evaluation_metrics.png"), bbox_inches='tight', dpi=300)
            logger.info("Evaluation metrics saved to %s", self.save_path)
            df.to_csv(osp.join(self.save_path, 'Evaluation_metrics.csv'), index=False)
            logger.info("Evaluation metrics saved to %s", self.save_path)

        # Show plot
        plt.show()
        

    def _plot_confusion_matrix(self, conf_matrix: np.ndarray) -> None:
        """Plot confusion matrix with adaptive sizing."""
        fig_size = min(max(8, self.num_classes * 0.8), 20)
        plt.figure(figsize=(fig_size, fig_size))
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=self.classes_labels,
            yticklabels=self.classes_labels,
            # xticklabels=[f'Class {i}' for i in range(self.num_classes)],
            # yticklabels=[f'Class {i}' for i in range(self.num_classes)]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if self.save_path:
            plt.savefig(osp.join(self.save_path, 'confusion_matrix.png'), bbox_inches='tight', dpi=300)
            logger.info("Confusion matrix saved to %s", self.save_path)

        plt.show()

class SegmentationEvaluator(BaseEvaluator):
    """Evaluator for segmentation tasks."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def evaluate(self, model: ModelInterface, dataloader: Any, device: Optional[str] = None) -> Dict[str, float]:
        metrics = {}
        # TODO: Implement segmentation metrics (IoU, Dice, etc.)
        return metrics

class DetectionEvaluator(BaseEvaluator):
    """Evaluator for object detection tasks."""
    
    def evaluate(self, model: ModelInterface, dataloader: Any, device: Optional[str] = None) -> Dict[str, float]:
        metrics = {}
        # TODO: Implement detection metrics (mAP, etc.)
        return metrics 
