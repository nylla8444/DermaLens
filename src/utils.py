"""
Utility functions for DermaLens
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to use
        class_names: List of class names
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        # Get predictions and labels for this class
        class_mask = np.array(all_labels) == i
        class_labels = np.array(all_labels)[class_mask]
        class_preds = np.array(all_preds)[class_mask]
        
        if len(class_labels) > 0:
            class_acc = accuracy_score(class_labels, class_preds)
            
            # Binary metrics for this class
            binary_labels = (np.array(all_labels) == i).astype(int)
            binary_preds = (np.array(all_preds) == i).astype(int)
            
            per_class_metrics[class_name] = {
                "accuracy": class_acc,
                "precision": precision_score(binary_labels, binary_preds, zero_division=0),
                "recall": recall_score(binary_labels, binary_preds, zero_division=0),
                "f1": f1_score(binary_labels, binary_preds, zero_division=0)
            }
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_matrix(all_labels, all_preds)
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def get_device():
    """Get available device (cuda or cpu)"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_model_summary(model):
    """Print model architecture summary"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")
