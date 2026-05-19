from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")

app = FastAPI(title="CKD XGBoost API")

# --- Load XGBoost Model ---
try:
    # We only need the XGBoost model now
    with open("models/xgb_trackD.pkl", "rb") as f:
        xgb_model = pickle.load(f)
except Exception as e:
    print(f"Error loading XGBoost model: {e}")

# --- Feature List (ORDER IS CRITICAL) ---
FEATURE_COLS = [
    "has_nsaid", "LBXHGB", "LBXTC", "LBXTR", "has_diabetes_med", "URXUCR",
    "sex_male", "LBXSBU", "BPQ020", "BUN_Albumin_ratio", "DIET_DRXTP225",
    "DIQ010", "URXUMA", "LBXSUA", "LBXBPB", "RIDAGEYR", "LBXBCD", "BMXBMI",
    "DIET_DRXT_V_LEGUMES", "BPXSY1", "Waist_Height_ratio", "DIET_DRXT_G_WHOLE",
    "LBX4PA", "DIET_DRXT_PF_SOY", "LBXSAL", "LBXBCC", "BMXWAIST", "URXUCD",
    "LBXTHG", "BPXDI1", "DIET_DRXTATOA", "DIET_DRXTM221", "race_Mexican_American",
    "race_Other_Hispanic", "race_NH_White", "race_NH_Black", "Pulse_Pressure",
    "has_ace_arb", "DIET_DRXTP184", "LBXGLU", "LBXGH", "total_medications"
]

class PredictionRequest(BaseModel):
    height_cm: float
    data: dict 

@app.post("/predict")
async def predict_ckd(payload: PredictionRequest):
    try:
        p = payload.data.copy()
        
        # 1. Feature Engineering (must match training logic)
        p["BUN_Albumin_ratio"] = p["LBXSBU"] / p["LBXSAL"]
        p["Pulse_Pressure"] = p["BPXSY1"] - p["BPXDI1"]
        p["Waist_Height_ratio"] = p["BMXWAIST"] / payload.height_cm
        
        # 2. Convert dict to numpy array in the exact order required
        input_data = np.array([[p[f] for f in FEATURE_COLS]])
        
        # 3. Model Inference
        probabilities = xgb_model.predict_proba(input_data)[0]
        prob_ckd = float(probabilities[1])
        
        return {
            "status": "success",
            "ckd_probability": round(prob_ckd, 4),
            "prediction": "CKD" if prob_ckd >= 0.5 else "No CKD",
            "risk_level": "High" if prob_ckd > 0.7 else "Moderate" if prob_ckd > 0.3 else "Low"
        }
        
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides a PORT environment variable automatically
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)