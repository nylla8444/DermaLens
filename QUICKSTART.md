# QUICK START GUIDE for DermaLens

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Train the Model
```bash
python train_model.py
```
This will:
- Load the dataset from `dataset/train`, `dataset/valid`, `dataset/test`
- Train a ResNet50 model with transfer learning
- Save the best model to `checkpoints/best_model.pth`
- Generate training logs and plots in `logs/`

**Training Time**: 
- GPU (NVIDIA): ~2-3 hours
- CPU: ~12-24 hours

## 3. Start the API Server
```bash
python main.py
```

## 4. Access the Web Interface
Open your browser and go to:
```
http://localhost:8000/ui
```

## 5. Upload and Analyze
- Click the drop zone to upload a dog skin image
- Click "Analyze Image"
- View the AI-powered diagnosis and recommendations

## API Documentation
Auto-generated Swagger UI available at:
```
http://localhost:8000/docs
```

## Troubleshooting

**Model not found error**:
- Ensure you've run `python train_model.py` first
- Check that `checkpoints/best_model.pth` exists

**CUDA out of memory**:
- Reduce batch_size in `configs/config.yaml`
- Or use CPU by setting `device: "cpu"`

**Port 8000 already in use**:
- Change port in `configs/config.yaml`
- Or kill the process using the port

## Next Steps

1. Review `README.md` for detailed documentation
2. Check `configs/config.yaml` for hyperparameter tuning
3. Explore training metrics in `logs/training_history.png`
4. Review predictions in the web UI


