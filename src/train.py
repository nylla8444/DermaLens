"""
Training pipeline for DermaLens model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from src.model import DermaLensModel
from src.data_loader import create_dataloaders


class Trainer:
    """Training manager for DermaLens model"""
    
    def __init__(
        self,
        model: DermaLensModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: str = "cuda",
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs"
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training dataloader
            valid_loader: Validation dataloader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        
        # Create directories
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
            "valid_f1": [],
            "learning_rate": []
        }
        
        self.best_valid_acc = 0.0
        self.best_valid_f1 = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (avg_loss, accuracy, f1_score)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch + 1} [Valid]")
        
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.valid_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, accuracy, f1
    
    def fit(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        warmup_epochs: int = 5,
        unfreeze_backbone_epoch: int = None
    ):
        """
        Train model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            warmup_epochs: Number of warmup epochs with frozen backbone
            unfreeze_backbone_epoch: Epoch at which to unfreeze backbone
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Unfreeze backbone after warmup
            if unfreeze_backbone_epoch is None:
                unfreeze_backbone_epoch = warmup_epochs
            
            if epoch == unfreeze_backbone_epoch:
                print(f"Unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            valid_loss, valid_acc, valid_f1 = self.validate(epoch)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_acc"].append(valid_acc)
            self.history["valid_f1"].append(valid_f1)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/valid", valid_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/valid", valid_acc, epoch)
            self.writer.add_scalar("F1/valid", valid_f1, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | Valid F1: {valid_f1:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # GPU memory info
            if self.device == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_mem_max = torch.cuda.max_memory_allocated() / 1e9
                print(f"GPU Memory: {gpu_mem:.2f}GB / Max: {gpu_mem_max:.2f}GB")
            
            # Learning rate scheduling
            self.scheduler.step(valid_acc)
            
            # Save best model
            if valid_acc > self.best_valid_acc:
                self.best_valid_acc = valid_acc
                self.best_f1 = valid_f1
                self.best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = Path(self.checkpoint_dir) / "best_model.pth"
                self.model.save_checkpoint(
                    str(checkpoint_path),
                    optimizer_state=self.optimizer.state_dict(),
                    epoch=epoch,
                    metrics={
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "valid_loss": valid_loss,
                        "valid_acc": valid_acc,
                        "valid_f1": valid_f1
                    }
                )
                print(f"✓ Best model saved at epoch {epoch + 1}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)")
                break
        
        # Save training history
        self._save_history()
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\nTraining completed!")
        print(f"Best epoch: {self.best_epoch + 1}")
        print(f"Best validation accuracy: {self.best_valid_acc:.4f}")
        print(f"Best validation F1: {self.best_f1:.4f}")
    
    def _save_history(self):
        """Save training history to JSON"""
        history_path = Path(self.log_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history["train_loss"], label="Train")
        axes[0, 0].plot(self.history["valid_loss"], label="Valid")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history["train_acc"], label="Train")
        axes[0, 1].plot(self.history["valid_acc"], label="Valid")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.history["valid_f1"])
        axes[1, 0].set_title("Validation F1 Score")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(self.history["learning_rate"])
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale("log")
        
        plt.tight_layout()
        plot_path = Path(self.log_dir) / "training_history.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Training plot saved to {plot_path}")
        plt.close()


def train_model(
    dataset_root: str,
    config: Dict,
    device: str = "cuda"
):
    """
    Complete training pipeline
    
    Args:
        dataset_root: Path to dataset
        config: Configuration dictionary
        device: Device to train on
    """
    print("Loading data...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_root=dataset_root,
        image_size=config.get("dataset", {}).get("image_size", 224),
        batch_size=config.get("training", {}).get("batch_size", 32),
        augmentation_config=config.get("augmentation", {})
    )
    
    # Print dataset info
    train_dataset = train_loader.dataset
    print(f"Dataset loaded successfully!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {train_dataset.class_names}")
    print(f"Class distribution (train): {train_dataset.get_class_distribution()}")
    
    # Create model
    print("\nCreating model...")
    model = DermaLensModel(
        num_classes=config.get("dataset", {}).get("num_classes", 6),
        architecture=config.get("model", {}).get("architecture", "resnet50"),
        pretrained=config.get("model", {}).get("pretrained", True),
        dropout_rate=config.get("model", {}).get("dropout_rate", 0.3)
    )
    print(f"Model created: {model.architecture}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        lr=config.get("training", {}).get("learning_rate", 0.001),
        weight_decay=config.get("training", {}).get("weight_decay", 0.0001),
        checkpoint_dir=config.get("training", {}).get("model_save_dir", "./checkpoints"),
        log_dir=config.get("training", {}).get("log_dir", "./logs")
    )
    
    # Train model
    print("\nStarting training...")
    trainer.fit(
        num_epochs=config.get("training", {}).get("num_epochs", 100),
        early_stopping_patience=config.get("training", {}).get("early_stopping_patience", 15),
        warmup_epochs=config.get("training", {}).get("warmup_epochs", 5)
    )
    
    # Plot history
    trainer.plot_history()
    
    return trainer, train_loader, valid_loader, test_loader
