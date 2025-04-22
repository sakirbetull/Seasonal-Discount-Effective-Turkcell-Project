from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="Seasonal Discount Effect API",
    description="API for predicting the effect of seasonal discounts on sales",
    version="1.0.0"
)

# Define request model
class PredictionRequest(BaseModel):
    product_name: str
    city: str
    category_name: str
    yearquarter: str
    discount: float

# Define response model
class PredictionResponse(BaseModel):
    prediction: bool
    probability: float
    message: str

# Load model and encoders
def load_artifacts():
    """Load the trained model and necessary artifacts"""
    current_dir = Path(__file__).parent.parent.parent
    model_path = current_dir / "models" / "salesprediction_xgboost_model.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError("Model file not found")
    
    return joblib.load(model_path)

# Initialize model
try:
    model = load_artifacts()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Seasonal Discount Effect API",
        "status": "operational"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction based on input data"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'product_name': request.product_name,
            'city': request.city,
            'category_name': request.category_name,
            'yearquarter': request.yearquarter,
            'discount': request.discount
        }])
        
        # Preprocess input data
        input_data['city'] = input_data['city'].astype('category').cat.codes
        input_data['product_name'] = input_data['product_name'].astype('category').cat.codes
        input_data = pd.get_dummies(input_data, columns=['category_name', 'yearquarter'], drop_first=True)
        
        # Make prediction
        probability = model.predict_proba(input_data)[0][1]
        prediction = probability > 0.5
        
        # Prepare response
        message = "Discount is likely to be effective" if prediction else "Discount may not be effective"
        
        return PredictionResponse(
            prediction=bool(prediction),
            probability=float(probability),
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }
