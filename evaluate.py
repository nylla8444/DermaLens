"""
Model evaluation and testing script
Usage: python evaluate.py
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info logs

import torch
import yaml
from pathlib import Path
import json
from src.model import DermaLensModel
from src.data_loader import create_dataloaders
from src.utils import evaluate_model, plot_confusion_matrix, print_model_summary


def main():
    """Main evaluation function"""
    # Load config
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_root=config.get("dataset", {}).get("dataset_path", "./dataset"),
        image_size=config.get("dataset", {}).get("image_size", 224),
        batch_size=config.get("training", {}).get("batch_size", 32),
        augmentation_config=config.get("augmentation", {})
    )
    
    class_names = train_loader.dataset.class_names
    print(f"Classes: {class_names}")
    
    # Load model
    print("\nLoading model...")
    checkpoint_path = config.get("api", {}).get("model_checkpoint", "./checkpoints/best_model.pth")
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first: python train_model.py")
        return
    
    model = DermaLensModel.from_checkpoint(checkpoint_path, device=device)
    print_model_summary(model)
    
    # Evaluate on different datasets
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    results = {}
    
    # Training set evaluation
    print("\nEvaluating on training set...")
    train_metrics = evaluate_model(model, train_loader, device, class_names)
    results["train"] = {
        "accuracy": train_metrics["accuracy"],
        "precision": train_metrics["precision"],
        "recall": train_metrics["recall"],
        "f1": train_metrics["f1"]
    }
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    
    # Validation set evaluation
    print("\nEvaluating on validation set...")
    valid_metrics = evaluate_model(model, valid_loader, device, class_names)
    results["validation"] = {
        "accuracy": valid_metrics["accuracy"],
        "precision": valid_metrics["precision"],
        "recall": valid_metrics["recall"],
        "f1": valid_metrics["f1"]
    }
    print(f"  Accuracy:  {valid_metrics['accuracy']:.4f}")
    print(f"  Precision: {valid_metrics['precision']:.4f}")
    print(f"  Recall:    {valid_metrics['recall']:.4f}")
    print(f"  F1 Score:  {valid_metrics['f1']:.4f}")
    
    # Test set evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, class_names)
    results["test"] = {
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"]
    }
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    # Per-class metrics
    print("\nPer-class metrics (Test Set):")
    print("-" * 60)
    print(f"{'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    
    for class_name in class_names:
        if class_name in test_metrics["per_class"]:
            metrics = test_metrics["per_class"][class_name]
            print(f"{class_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print("-" * 60)
    
    # Save confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        save_path="logs/confusion_matrix.png"
    )
    
    # Save results to JSON
    print("\nSaving results...")
    results_path = Path("logs/evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
