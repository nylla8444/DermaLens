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
        "description": "Contact dermatitis is an inflammatory skin condition caused by direct contact with irritants or allergens such as certain plants, chemicals, cleaning products, or materials. It can cause redness, itching, scaling, and discomfort. Common in dogs with sensitive skin or exposure to environmental triggers.",
        "recommendation": "Identify and remove the source of irritation. Rinse the affected area with cool water. Avoid harsh chemicals or cleaners around your pet. Consider hypoallergenic bedding and bowls. Consult your veterinarian if symptoms persist or worsen, as prescription treatments may be needed."
    },
    "Demodicosis": {
        "description": "Demodicosis is caused by Demodex mites that naturally live in hair follicles but overpopulate when the immune system is compromised. Can be localized (small patches) or generalized (widespread). Common in puppies, elderly dogs, or immunocompromised pets. Causes hair loss, redness, scaling, and skin thickening.",
        "recommendation": "Schedule an immediate veterinary appointment for skin scraping diagnosis. Treatment requires prescription medication (topical or oral). Avoid stressful situations that weaken the immune system. Maintain clean bedding and living environment. Do not attempt home treatment as it requires veterinary supervision."
    },
    "Fungal Infections": {
        "description": "Fungal skin infections include ringworm and other fungal species that affect the skin, hair, and nails. These infections are highly contagious to other pets and humans. Symptoms include circular patches of hair loss, scaly or crusty skin, redness, and itching. Can spread rapidly if untreated.",
        "recommendation": "Isolate the affected pet from other animals and children immediately. Visit your veterinarian for fungal culture testing and treatment plan. Use antifungal medications as prescribed (typically 6-12 weeks). Thoroughly disinfect all bedding, toys, and living areas. Wash hands after handling the pet. All household pets should be examined."
    },
    "Healthy": {
        "description": "No visible signs of skin disease detected. The skin appears healthy with normal coloration, texture, and hair coat. No lesions, redness, scaling, or abnormalities observed. This indicates good overall skin health and proper care.",
        "recommendation": "Continue your current grooming and care routine. Maintain a balanced diet rich in omega fatty acids for skin health. Provide regular exercise and adequate hydration. Schedule routine veterinary check-ups for preventive care. Monitor for any changes in skin condition. Keep up with flea and tick prevention."
    },
    "Hypersensitivity": {
        "description": "Hypersensitivity reactions are allergic responses causing skin irritation, itching, redness, inflammation, and sometimes hair loss. Can be triggered by food allergens, environmental factors (pollen, dust mites), flea bites, or contact allergens. May be seasonal or year-round depending on the trigger.",
        "recommendation": "Identify and eliminate potential allergens through an elimination process. Consider a hypoallergenic diet trial (8-12 weeks). Use veterinarian-recommended flea prevention products year-round. Keep indoor environment clean and dust-free. Consult your vet about antihistamines, omega-3 supplements, or allergy testing for comprehensive management."
    },
    "Ringworm": {
        "description": "Ringworm is a highly contagious fungal infection (dermatophyte) that affects the skin, hair, and nails. Despite its name, it's not caused by worms. Appears as circular, scaly patches of hair loss with a ring-like appearance. Can spread to humans and other pets through direct contact or contaminated surfaces. Requires immediate attention.",
        "recommendation": "URGENT: Seek immediate veterinary treatment. Quarantine the infected pet in a separate room away from all animals and people. Thoroughly disinfect all surfaces, bedding, toys, and grooming tools with diluted bleach solution. Wear disposable gloves when handling the pet or cleaning. Complete the full course of antifungal treatment even if symptoms improve. All household members and pets should be examined by a doctor/veterinarian."
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
static_dir = Path(__file__).parent / "static"
logger.info(f"Static files directory: {static_dir}")
logger.info(f"Static directory exists: {static_dir.exists()}")
if static_dir.exists():
    logger.info(f"Static files: {list(static_dir.rglob('*'))}")

# Check if directory exists, create if needed
if not static_dir.exists():
    logger.warning(f"Static directory does not exist: {static_dir}")
else:
    app.mount("/static", StaticFiles(directory=str(static_dir), check_dir=True), name="static")


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
    """Get API information including model metrics"""
    model_accuracy = "Unknown"
    model_epoch = "Unknown"
    
    # Try to read accuracy from checkpoint
    checkpoint_path = Path("./checkpoints/best_model.pth")
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'metrics' in checkpoint:
                valid_acc = checkpoint['metrics'].get('valid_acc')
                if valid_acc:
                    model_accuracy = f"{valid_acc * 100:.2f}%"
                model_epoch = checkpoint.get('epoch', 'Unknown')
        except Exception as e:
            logger.warning(f"Could not read model metrics: {e}")
    
    return {
        "api_name": "DermaLens",
        "version": "1.0.0",
        "description": "AI-powered dog skin lesion classification",
        "model": {
            "architecture": config.get("model", {}).get("architecture", "resnet50"),
            "num_classes": config.get("dataset", {}).get("num_classes", 6),
            "device": device,
            "loaded": model is not None,
            "best_epoch": model_epoch,
            "validation_accuracy": model_accuracy
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
        
        # Calculate confidence level indicator
        def get_confidence_level(confidence_score, disease_name):
            """
            Determine confidence level indicator based on prediction confidence
            
            This is NOT a medical severity rating - it's a visual indicator
            of how confident the AI model is in its prediction.
            """
            # For healthy skin, inverse logic: high confidence = truly healthy
            if disease_name == "Healthy":
                if confidence_score >= 0.85:
                    return "Very High"
                elif confidence_score >= 0.70:
                    return "High"
                else:
                    return "Moderate"  # Low confidence on "healthy" means uncertain
            
            # For diseases: higher confidence = more certain diagnosis
            if confidence_score >= 0.85:
                return "Very High"
            elif confidence_score >= 0.70:
                return "High"
            elif confidence_score >= 0.55:
                return "Moderate"
            else:
                return "Low"  # Low confidence suggests uncertain diagnosis
        
        # Get base class info
        base_info = class_info.get(predicted_class, {
            "description": "No description available",
            "recommendation": "Consult a veterinarian"
        })
        
        # Calculate confidence level indicator
        confidence_level = get_confidence_level(confidence, predicted_class)
        
        # Create enhanced class info with confidence level
        enhanced_class_info = {
            "description": base_info["description"],
            "recommendation": base_info["recommendation"],
            "confidence_level": confidence_level
        }
        
        # Prepare results
        results = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                class_names[i]: float(confidence_scores[i])
                for i in range(len(class_names))
            },
            "class_info": enhanced_class_info,
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
    template_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(str(template_path))


@app.get("/static/js/app.js")
async def app_js():
    """Serve app.js directly as fallback"""
    app_js_path = Path(__file__).parent / "static" / "js" / "app.js"
    if app_js_path.exists():
        return FileResponse(str(app_js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


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
