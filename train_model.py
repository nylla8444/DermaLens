"""
Main training script for DermaLens
Usage: python train_model.py
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info logs

import torch
import yaml
from pathlib import Path
from src.train import train_model


def main():
    """Main training function"""
    # Load config
    config_path = Path("configs/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}\n")
    
    # Train model
    trainer, train_loader, valid_loader, test_loader = train_model(
        dataset_root=config.get("dataset", {}).get("dataset_path", "./dataset"),
        config=config,
        device=device
    )


if __name__ == "__main__":
    main()
