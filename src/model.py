"""
ResNet-based model for skin lesion classification
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Optional


class DermaLensModel(nn.Module):
    """
    ResNet-based model with custom classification head for skin lesion classification
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        architecture: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Initialize model
        
        Args:
            num_classes: Number of output classes
            architecture: ResNet architecture ('resnet50', 'resnet101', etc.)
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone weights initially
        """
        super(DermaLensModel, self).__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        
        # Load pretrained model using weights parameter (PyTorch 2.9+)
        if architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif architecture == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
        elif architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Get input features for classification head
        in_features = self.backbone.fc.in_features
        
        # Remove original classification head
        self.backbone.fc = nn.Identity()
        
        # Create custom classification head
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_batch_norm(self):
        """Freeze batch normalization layers"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    
    def save_checkpoint(self, path: str, optimizer_state=None, epoch=None, metrics=None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            optimizer_state: Optimizer state dict
            epoch: Current epoch
            metrics: Training metrics
        """
        checkpoint = {
            'model_state': self.state_dict(),
            'architecture': self.architecture,
            'num_classes': self.num_classes,
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state'])
        return checkpoint
    
    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cpu'):
        """
        Create model from checkpoint
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved architecture
        model = cls(
            num_classes=checkpoint.get('num_classes', 6),
            architecture=checkpoint.get('architecture', 'resnet50'),
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        model.eval()
        
        return model


class MultiHeadModel(nn.Module):
    """
    ResNet-based model with multiple classification heads
    Useful for multi-task learning or ensemble approaches
    """
    
    def __init__(
        self,
        num_heads: int = 2,
        head_dimensions: List[int] = None,
        architecture: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        """
        Initialize multi-head model
        
        Args:
            num_heads: Number of classification heads
            head_dimensions: Output dimensions for each head
            architecture: ResNet architecture
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
        """
        super(MultiHeadModel, self).__init__()
        
        if head_dimensions is None:
            head_dimensions = [6] * num_heads
        
        # Load backbone using weights parameter (PyTorch 2.9+)
        if architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif architecture == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
        elif architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Create multiple heads
        self.heads = nn.ModuleList()
        for dim in head_dimensions:
            head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, dim)
            )
            self.heads.append(head)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through multiple heads
        
        Args:
            x: Input tensor
        
        Returns:
            List of output tensors from each head
        """
        features = self.backbone(x)
        outputs = [head(features) for head in self.heads]
        return outputs
