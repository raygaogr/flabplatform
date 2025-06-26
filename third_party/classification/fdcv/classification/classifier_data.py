"""
Copyright (C) 2025 dsl.
"""
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, default_collate
from torchvision import transforms, datasets
from pathlib import Path
import numpy as np
import logging
import torch.nn.functional as F
import json
from collections import defaultdict, Counter

import tqdm
import os
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

__all__ = ['ClassificationData', 'LabelmeClassificationData']

class Mixup:
    """Mixup data augmentation."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Mixup.
        
        Args:
            alpha: Alpha parameter for beta distribution
        """
        self.alpha = alpha
        
    def __call__(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to the batch.
        
        Args:
            batch: List of (image, label) tuples
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lambda)
        """
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.stack([torch.tensor(item[1]) for item in batch], dim=0)
        batch_size = images.size(0)
        
        # Generate mixup weights from beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Shuffle indices
        index = torch.randperm(batch_size)
        
        # Mix the images
        mixed_images = lam * images + (1 - lam) * images[index, :]
        
        return mixed_images, labels, labels[index], lam

class LabelSmoothing:
    """Label smoothing for classification tasks."""
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing.
        
        Args:
            smoothing: Smoothing factor (0-1). Defaults to 0.1
        """
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be between 0 and 1")
        self.smoothing = smoothing
        
    def __call__(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Apply label smoothing to the labels.
        
        Args:
            labels: Original labels (batch_size,)
            num_classes: Number of classes
            
        Returns:
            Smoothed labels (batch_size, num_classes)
        """
        # Convert to one-hot
        one_hot = F.one_hot(labels, num_classes).float()
        
        # Apply smoothing
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        return smoothed

class ClassificationData(LightningDataModule):
    """Lightning DataModule for image classification tasks."""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        split_ratio: float = 0.2,
        split_flag: bool = True,
        input_size: Tuple[int, int] = (224, 224),
        augmentation: bool = True,
        mixup: bool = False,
        mixup_alpha: float = 1.0,
        label_smoothing: float = 0.0,
        custom_transforms: Optional[Dict[str, Callable]] = None,
        **kwargs: Any
    ):
        """
        Initialize the data module.
        
        Args:
            data_dir: Directory containing the dataset
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses for data loading
            split_ratio: Ratio of validation set when splitting
            split_flag: Whether to split dataset or use predefined train/val dirs
            input_size: Model input size (height, width)
            augmentation: Whether to use data augmentation for training
            mixup: Whether to use mixup augmentation
            mixup_alpha: Alpha parameter for mixup
            label_smoothing: Smoothing factor for label smoothing
            custom_transforms: Custom transforms for different phases
            **kwargs: Additional arguments
        """
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.split_flag = split_flag
        self.input_size = input_size
        self.augmentation = augmentation
        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.custom_transforms = custom_transforms or {}

        self.num_classes = 0
        
        self._validate_inputs()
        self._setup_transforms()
        
        if self.mixup:
            self.mixup_fn = Mixup(alpha=mixup_alpha)
        if self.label_smoothing > 0:
            self.label_smoothing_fn = LabelSmoothing(smoothing=label_smoothing)
        
    def _validate_inputs(self):
        """Validate input parameters."""
        if self.split_ratio < 0 or self.split_ratio > 1:
            raise ValueError("split_ratio must be between 0 and 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
            
    def _setup_transforms(self):
        """Set up data transforms for different phases."""
        # Base transform for all phases
        base_transform = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Training transforms with augmentation
        train_transforms = []
        if self.augmentation:
            train_transforms.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0))
            ])
        train_transforms.extend(base_transform)
        
        # Set up transforms for each phase
        self.train_transform = (
            self.custom_transforms.get('train') or 
            transforms.Compose(train_transforms)
        )
        
        self.val_transform = (
            self.custom_transforms.get('val') or 
            transforms.Compose(base_transform)
        )
        
        self.test_transform = (
            self.custom_transforms.get('test') or 
            transforms.Compose(base_transform)
        )
    
    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if self.data_dir is None:
            logger.warning("No data directory provided. Skipping dataset setup.")
            return
            
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.check_dataset()    
        try:
            
            if self.split_flag:
                # Load and split single dataset
                self.dataset = datasets.ImageFolder(
                    root=str(self.data_dir),
                    transform=None  # Transform will be applied later
                )
                self.num_classes = len(self.dataset.classes)
                if len(self.dataset) == 0:
                    raise RuntimeError(f"No images found in {self.data_dir}")
                    
                # Split dataset
                dataset_len = len(self.dataset)
                train_len = int(dataset_len * (1 - self.split_ratio))
                val_len = dataset_len - train_len
                
                logger.info(
                    f"Dataset size: {dataset_len}, "
                    f"Split ratio: {self.split_ratio}, "
                    f"Train size: {train_len}, "
                    f"Val size: {val_len}"
                )
                
                # Create train/val datasets with appropriate transforms
                generator = torch.Generator().manual_seed(42)
                train_dataset, val_dataset = random_split(
                    self.dataset,
                    [train_len, val_len],
                    generator=generator
                )
                
                # 统计 `train_dataset` 和 `val_dataset` 中的类别样本数量
                train_labels = [self.dataset.targets[i] for i in train_dataset.indices]
                val_labels = [self.dataset.targets[i] for i in val_dataset.indices]

                self.train_class_counts = Counter(train_labels)
                self.val_class_counts = Counter(val_labels)

                logger.info(f"train_class_counts: {self.train_class_counts}")
                logger.info(f"val_class_counts: {self.val_class_counts}")

                # 按类别索引顺序排序
                self.train_class_counts_dict = {cls: self.train_class_counts[idx] for idx, cls in enumerate(self.dataset.classes)}
                self.val_class_counts_dict = {cls: self.val_class_counts[idx] for idx, cls in enumerate(self.dataset.classes)}

                # Wrap datasets to apply transforms
                self.train_dataset = TransformDataset(train_dataset, self.train_transform)
                self.val_dataset = TransformDataset(val_dataset, self.val_transform)
                
            else:
                # Load predefined train/val splits
                train_dir = self.data_dir / "train"
                val_dir = self.data_dir / "val"
                
                if not train_dir.exists():
                    raise FileNotFoundError(f"Training directory not found: {train_dir}")
                if not val_dir.exists():
                    raise FileNotFoundError(f"Validation directory not found: {val_dir}")
                    
                self.train_dataset = datasets.ImageFolder(
                    root=str(train_dir),
                    transform=self.train_transform
                )
                self.val_dataset = datasets.ImageFolder(
                    root=str(val_dir),
                    transform=self.val_transform
                )
                self.num_classes = len(self.train_dataset.classes)
                logger.info(
                    f"Train dataset size: {len(self.train_dataset)}, "
                    f"Val dataset size: {len(self.val_dataset)}"
                )

        except Exception as e:
            logger.error(f"Error setting up datasets: {str(e)}")
            raise
    
    def train_dataloader(self) -> DataLoader:
        """Create the training data loader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
            
        # 创建collate_fn来处理mixup和label smoothing
        def collate_fn(batch):
            if self.mixup:
                # 应用mixup
                mixed_batch = self.mixup_fn(batch)
                if self.label_smoothing > 0:
                    # 对两组标签都应用平滑
                    x, y_a, y_b, lam = mixed_batch
                    y_a = self.label_smoothing_fn(y_a, self.num_classes)
                    y_b = self.label_smoothing_fn(y_b, self.num_classes)
                    return x, y_a, y_b, lam
                return mixed_batch
            else:
                # 标准batch处理
                x, y = default_collate(batch)
                if self.label_smoothing > 0:
                    y = self.label_smoothing_fn(y, self.num_classes)
                return x, y

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Create the validation data loader.
        
        Returns:
            DataLoader: Validation data loader
            
        Raises:
            RuntimeError: If val_dataset is not set
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create the test data loader (uses validation dataset).
        
        Returns:
            DataLoader: Test data loader
            
        Raises:
            RuntimeError: If val_dataset is not set
        """
        return self.val_dataloader()
    
    def check_dataset(self):
        IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
        for root,_,file_name_list in os.walk(self.data_dir):
            for file_name in file_name_list:
                if os.path.splitext(file_name)[-1] in IMG_EXTENSIONS:
                    img_path = os.path.join(root,file_name)
                    try:
                        img = Image.open(img_path)
                    except Exception as e:
                        logger.info(e)
                        os.remove(img_path)

class TransformDataset:
    """Wrapper dataset that applies transforms on the fly."""
    
    def __init__(self, dataset: Dataset, transform: Optional[Callable] = None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label
        
    def __len__(self):
        return len(self.dataset)

class LabelmeClassificationDataset(Dataset):
    def __init__(self, metadata_path, root_dir, purpose="train", transform=None):
        self.root_dir = Path(root_dir)
        self.metadata_path = Path(metadata_path)
        self.purpose = purpose  # 'train' or 'eval'
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        self._load_metadata()

    def _load_metadata(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.classes = metadata.get("names", [])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        for dataset in metadata["datasets"]:
            if dataset.get("purpose") != self.purpose:
                logger.info(f"Skipping dataset {dataset['name']} for purpose {self.purpose}.")
                continue
            source_root = dataset["sourceRoot"]
            image_dir = Path(self.root_dir) / source_root / "image"
            label_dir = Path(self.root_dir) / source_root / "label"
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                logger.warning(f"Source root {source_root} does not exist in {self.root_dir}. Skipping dataset.")
                continue

            image_files = list(Path(image_dir).glob("*.jpg")) + \
                          list(Path(image_dir).glob("*.png")) + \
                          list(Path(image_dir).glob("*.jpeg")) + \
                          list(Path(image_dir).glob("*.bmp"))

            for image_file in image_files:
                label_path = Path(label_dir) / (image_file.stem + ".json") 
                if not label_path.exists():
                    logger.warning(f"Label file {label_path} does not exist for image {image_file}. Skipping.")
                    continue
                with open(label_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                shapes = data.get("shapes", [])
                if not shapes:
                    logger.warning(f"No shapes found in label file {label_path}. Skipping.")
                    continue
                if len(shapes) == 0:
                    logger.warning(f"No shapes found in label file {label_path}. Skipping.")
                else:
                    if "label" in shapes[0]:
                        label = shapes[0]["label"]
                    else:
                        logger.warning(f"No label found in first shape of {label_path}. Skipping.")
                        continue
                    if label not in self.class_to_idx:
                        logger.warning(f"Label {label} not in class_to_idx mapping. Skipping.")
                        continue
                    self.samples.append((str(image_file), self.class_to_idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class LabelmeClassificationData(LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: Tuple[int, int] = (224, 224),
        augmentation: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.metadata_path = Path(metadata_path)
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.augmentation = augmentation

        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = 0
        self.classes = []

    def _get_transforms(self):
        base_transform = [
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
        train_transform = []
        if self.augmentation:
            train_transform += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomResizedCrop(self.input_size, scale=(0.8, 1.0))
            ]
        train_transform += base_transform
        return transforms.Compose(train_transform), transforms.Compose(base_transform)

    def setup(self, stage: Optional[str] = None):
        train_tf, val_tf = self._get_transforms()

        train_dataset = LabelmeClassificationDataset(
            metadata_path=self.metadata_path,
            root_dir=self.root_dir,
            purpose="train",
            transform=train_tf
        )
        val_dataset = LabelmeClassificationDataset(
            metadata_path=self.metadata_path,
            root_dir=self.root_dir,
            purpose="eval",
            transform=val_tf
        )

        self.num_classes = len(train_dataset.classes)
        self.classes = train_dataset.classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
