"""
Data loading and preprocessing utilities for DermaLens
"""
import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2


class DermaLensDataset(Dataset):
    """Custom PyTorch Dataset for skin lesion images"""
    
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        image_size: int = 224,
        augment: bool = False,
        augmentation_config: Dict = None
    ):
        """
        Initialize dataset
        
        Args:
            dataset_root: Root directory containing dataset
            split: One of 'train', 'valid', or 'test'
            image_size: Size to resize images to
            augment: Whether to apply augmentation
            augmentation_config: Dictionary of augmentation parameters
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        
        # Get class directories and create mapping
        self.split_dir = self.dataset_root / split
        self.class_names = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Collect image paths
        self.image_paths = []
        self.labels = []
        self._load_image_paths()
        
        # Setup transforms
        self.transform = self._get_transforms()
    
    def _load_image_paths(self):
        """Load all image paths from dataset directory"""
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            image_files = [f for f in class_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            for image_path in image_files:
                self.image_paths.append(str(image_path))
                self.labels.append(self.class_to_idx[class_name])
    
    def _get_transforms(self):
        """Get image transformation pipeline"""
        if self.split == "train" and self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(self.augmentation_config.get('scale_min', 0.8),
                           self.augmentation_config.get('scale_max', 1.0)),
                    ratio=(self.augmentation_config.get('aspect_ratio_min', 0.9),
                           self.augmentation_config.get('aspect_ratio_max', 1.1))
                ),
                transforms.RandomHorizontalFlip(p=0.5) if self.augmentation_config.get('random_flip', True) else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=self.augmentation_config.get('random_rotation', 15)),
                transforms.ColorJitter(
                    brightness=self.augmentation_config.get('brightness', 0.2),
                    contrast=self.augmentation_config.get('contrast', 0.2),
                    saturation=self.augmentation_config.get('saturation', 0.2),
                    hue=self.augmentation_config.get('hue', 0.1)
                ) if self.augmentation_config.get('color_jitter', True) else transforms.Lambda(lambda x: x),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) if self.augmentation_config.get('gaussian_blur', True) else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get single sample"""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor, label, image_path
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index"""
        return self.class_names[label]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class"""
        distribution = {}
        for label in self.labels:
            class_name = self.class_names[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


def create_dataloaders(
    dataset_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_config: Dict = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for train, valid, and test sets
    
    Args:
        dataset_root: Root directory of dataset
        image_size: Size to resize images to
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        augmentation_config: Dictionary of augmentation parameters
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = DermaLensDataset(
        dataset_root,
        split="train",
        image_size=image_size,
        augment=True,
        augmentation_config=augmentation_config
    )
    
    valid_dataset = DermaLensDataset(
        dataset_root,
        split="valid",
        image_size=image_size,
        augment=False
    )
    
    test_dataset = DermaLensDataset(
        dataset_root,
        split="test",
        image_size=image_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


def get_dataset_stats(dataset_root: str) -> Dict[str, Dict]:
    """
    Get statistics about the dataset
    
    Args:
        dataset_root: Root directory of dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    
    for split in ["train", "valid", "test"]:
        dataset = DermaLensDataset(dataset_root, split=split)
        stats[split] = {
            "num_samples": len(dataset),
            "class_distribution": dataset.get_class_distribution(),
            "class_names": dataset.class_names
        }
    
    return stats
