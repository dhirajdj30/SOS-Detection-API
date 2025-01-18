from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import uvicorn
from typing import Optional


MODEL_PATH = "saved_models/sos_model.joblib"
THRESHOLDS_PATH = "saved_models/thresh.joblib"


app = FastAPI()

try:
    model = joblib.load(MODEL_PATH)
    thresholds = joblib.load(THRESHOLDS_PATH)
    print("Model and thresholds loaded successfully")
except Exception as e:
    print(f"Error loading model or thresholds: {str(e)}")
    model = None
    thresholds = None

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "SOS Detection API",
        "status": "active",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


class SensorData(BaseModel):
    acceleration: float = Field(..., description="Acceleration in m/s^2")
    rotation: float = Field(..., description="Rotation in rad/s")
    magnetic_field: float = Field(..., description="Magnetic field in μT")
    light: float = Field(..., description="Light in lux")

# Response model
class SOSResponse(BaseModel):
    sos_triggered: bool
    confidence: float
    threshold_exceeded: Optional[list[str]] = None

@app.post("/predict", response_model=SOSResponse)
async def predict_sos(data: SensorData):
    """
    Predict SOS triggering based on sensor data
    
    Args:
        data: SensorData object containing sensor readings
        
    Returns:
        SOSResponse object containing prediction results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Check which thresholds are exceeded
    exceeded_thresholds = []
    if data.acceleration > thresholds['acceleration']:
        exceeded_thresholds.append('acceleration')
    if data.rotation > thresholds['rotation']:
        exceeded_thresholds.append('rotation')
    if data.magnetic_field > thresholds['magnetic_field']:
        exceeded_thresholds.append('magnetic_field')
    if data.light > thresholds['light']:
        exceeded_thresholds.append('light')

    # Make prediction
    features = [[
        data.acceleration,
        data.rotation,
        data.magnetic_field,
        data.light
    ]]
    
    prediction = bool(model.predict(features)[0])
    
    # Get prediction probability if available
    try:
        confidence = float(max(model.predict_proba(features)[0]))
    except:
        confidence = 1.0 if prediction else 0.0

    return SOSResponse(
        sos_triggered=prediction,
        confidence=confidence,
        threshold_exceeded=exceeded_thresholds if exceeded_thresholds else None
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)