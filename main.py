"""
FastAPI inference server for DermaLens
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info logs

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import torch
from PIL import Image
import io
import numpy as np
from typing import Dict, List
import yaml
import logging

from src.model import DermaLensModel
from src.data_loader import DermaLensDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path("configs/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Global variables
model = None
device = None
class_names = None
class_info = {
    "Dermatitis": {
        "description": "Inflammatory skin condition often caused by allergies, irritants, or infections",
        "recommendation": "Consult a veterinarian. Keep affected area clean and dry. Avoid irritants.",
        "severity": "Medium"
    },
    "Demodicosis": {
        "description": "Caused by Demodex mites; can be localized or generalized",
        "recommendation": "Consult a veterinarian immediately. Requires prescription treatment.",
        "severity": "High"
    },
    "Fungal Infections": {
        "description": "Fungal infection (ringworm or other fungal species)",
        "recommendation": "Consult a veterinarian. Use antifungal treatments as prescribed.",
        "severity": "High"
    },
    "Healthy": {
        "description": "Skin appears healthy with no visible lesions or abnormalities",
        "recommendation": "Continue regular grooming and maintenance. Monitor for any changes.",
        "severity": "None"
    },
    "Hypersensitivity": {
        "description": "Allergic reaction or skin sensitivity (food, environmental, or contact allergy)",
        "recommendation": "Identify and avoid allergens. Consult vet if severe or persistent.",
        "severity": "Low-Medium"
    },
    "Ringworm": {
        "description": "Highly contagious fungal infection",
        "recommendation": "Isolate pet and consult veterinarian immediately. Treat all contact surfaces.",
        "severity": "High"
    }
}


def load_model():
    """Load trained model"""
    global model, device, class_names
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    checkpoint_path = config.get("api", {}).get("model_checkpoint", "./checkpoints/best_model.pth")
    
    if not Path(checkpoint_path).exists():
        logger.warning(f"Model checkpoint not found at {checkpoint_path}")
        logger.info("Model will be loaded when available")
        return False
    
    try:
        model = DermaLensModel.from_checkpoint(checkpoint_path, device=device)
        class_names = config.get("dataset", {}).get("class_names", [
            "Dermatitis", "Demodicosis", "Fungal Infections", 
            "Healthy", "Hypersensitivity", "Ringworm"
        ])
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    load_model()
    yield
    # Shutdown
    if model is not None:
        logger.info("Cleaning up model...")
        # Move model to CPU and clear CUDA memory
        if device == "cuda":
            model.cpu()
            torch.cuda.empty_cache()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DermaLens API",
    description="AI-powered dog skin lesion classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="DermaLens API",
    description="AI-powered dog skin lesion classification",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Return welcome message"""
    return {
        "message": "Welcome to DermaLens API",
        "description": "AI-powered dog skin lesion classification system",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "info": "/info",
            "classes": "/classes"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": device,
        "timestamp": str(np.datetime64('now'))
    }


@app.get("/info")
async def info():
    """Get API information"""
    return {
        "api_name": "DermaLens",
        "version": "1.0.0",
        "description": "AI-powered dog skin lesion classification",
        "model": {
            "architecture": config.get("model", {}).get("architecture", "resnet50"),
            "num_classes": config.get("dataset", {}).get("num_classes", 6),
            "device": device,
            "loaded": model is not None
        },
        "max_upload_size_mb": config.get("api", {}).get("max_upload_size", 10485760) / (1024 * 1024)
    }


@app.get("/classes")
async def classes():
    """Get available disease classes"""
    classes_info = {}
    for cls_name in class_names:
        classes_info[cls_name] = class_info.get(cls_name, {
            "description": "No description available",
            "recommendation": "Consult a veterinarian",
            "severity": "Unknown"
        })
    return {
        "classes": class_names,
        "details": classes_info
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin lesion class from uploaded image
    
    Parameters:
        file: Image file to classify
    
    Returns:
        Prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Check file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="File must be JPEG or PNG image")
    
    # Check file size
    max_size = config.get("api", {}).get("max_upload_size", 10485760)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    if file_size > max_size:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {max_size / (1024*1024):.1f}MB")
    
    try:
        # Read and process image
        file.file.seek(0)
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        
        # Get predictions
        confidence_scores = probabilities.cpu().numpy()
        predicted_class_idx = np.argmax(confidence_scores)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(confidence_scores[predicted_class_idx])
        
        # Prepare results
        results = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                class_names[i]: float(confidence_scores[i])
                for i in range(len(class_names))
            },
            "class_info": class_info.get(predicted_class, {
                "description": "No description available",
                "recommendation": "Consult a veterinarian",
                "severity": "Unknown"
            }),
            "device": device,
            "model_version": "1.0.0"
        }
        
        return JSONResponse(content=results)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/ui")
async def ui():
    """Serve web UI"""
    return FileResponse("templates/index.html")


if __name__ == "__main__":
    import uvicorn
    
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    reload = config.get("api", {}).get("reload", True)
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )
