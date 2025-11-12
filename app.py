
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, joblib, json, requests
from typing import Optional

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
LE_PATH = os.path.join(os.path.dirname(__file__), "label_encoder.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")

app = FastAPI(title="Crop Recommender API")

# Load model + metadata on startup
model = None
label_encoder = None
metadata = None

@app.on_event("startup")
def load_model():
    global model, label_encoder, metadata
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(LE_PATH):
        label_encoder = joblib.load(LE_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)

class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    temperature_c: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    rainfall: Optional[float] = None
    soil_type: Optional[str] = None

def fetch_weather(lat, lon, api_key):
    # Use OpenWeatherMap - requires API key
    if not api_key: return None
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    r = requests.get(url, timeout=10)
    if r.status_code!=200: return None
    j = r.json()
    return {"temperature": j.get("main",{}).get("temp"), "humidity": j.get("main",{}).get("humidity")}

def fetch_soilgrids(lat, lon):
    # SoilGrids simple query (ISRIC) - public endpoint
    try:
        url = f"https://rest.soilgrids.org/query?lat={lat}&lon={lon}"
        r = requests.get(url, timeout=10)
        if r.status_code!=200:
            return None
        j = r.json()
        # Example: extract PHIHOX (pH in H2O) at 0-5cm if present
        ph_val = None
        props = j.get("properties",{})
        params = props.get("layers",{}) if props else {}
        # SoilGrids returns many layers; attempt to find 'ph' or 'PHIHOX'
        # We'll try properties.TAXOUSDA or analysis if available; fallback None
        # For safety return a structure
        return {"raw": j}
    except Exception as e:
        return None

@app.get("/metadata")
def get_metadata():
    return metadata or {"message":"no metadata found"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or label_encoder is None or metadata is None:
        raise HTTPException(status_code=500, detail="Model or metadata missing on server.")

    # If weather not provided, try fetching from OpenWeather using env key
    if req.temperature_c is None or req.humidity is None:
        ow = os.environ.get("OPENWEATHER_API_KEY")
        if ow:
            w = fetch_weather(req.latitude, req.longitude, ow)
            if w:
                if req.temperature_c is None:
                    req.temperature_c = w.get("temperature")
                if req.humidity is None:
                    req.humidity = w.get("humidity")

    # If ph not provided, try SoilGrids quick fetch (best-effort)
    if req.ph is None:
        sg = fetch_soilgrids(req.latitude, req.longitude)
        # This endpoint returns complex json; we include raw for debugging
        if sg:
            # No reliable simple ph extraction implemented here; leave None
            pass

    # Build feature vector using metadata feature_columns order
    feat_cols = metadata.get("feature_columns", [])
    feat = []
    for c in feat_cols:
        # map column names to request fields (simple mapping)
        val = None
        key = c.lower()
        if key in ("temperature","temperature_c","temp"):
            val = req.temperature_c
        elif key in ("humidity",):
            val = req.humidity
        elif key=="ph":
            val = req.ph
        elif key in ("n","npk_n"):
            val = req.N
        elif key in ("p","npk_p"):
            val = req.P
        elif key in ("k","npk_k"):
            val = req.K
        elif key=="rainfall":
            val = req.rainfall
        else:
            # For one-hot encoded soil types, try to map 'soil_{type}'
            if c.startswith("soil_") and req.soil_type:
                # if soil_type text matches suffix, set 1 else 0
                soil_suffix = c.replace("soil_","").lower()
                val = 1 if soil_suffix in req.soil_type.lower() else 0
            else:
                val = 0
        # fallback to 0 for None to keep vector length stable
        feat.append(float(val) if val is not None else 0.0)

    import numpy as np
    X = np.array([feat])
    pred = model.predict(X)[0]
    proba = None
    try:
        proba = float(model.predict_proba(X).max())
    except Exception:
        proba = None
    crop_name = label_encoder.inverse_transform([int(pred)])[0]
    return {"recommended_crop": crop_name, "confidence": proba, "features_used": feat_cols}

