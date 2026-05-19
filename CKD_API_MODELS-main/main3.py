from fastapi import FastAPI

from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import xgboost as xgb

# تحميل الموديل مرة واحدة فقط
model = joblib.load("models/logreg_trackD.pkl")
model2 = joblib.load("models/stacking_meta_model.pkl")
model3 = joblib.load("models/weighted_fusion_model.pkl")
model4 = joblib.load("models/xgb_trackD.pkl")

app = FastAPI()

# نفس features تاعك بالضبط
class PatientData(BaseModel):
    has_nsaid: int
    LBXHGB: float
    LBXTC: float
    LBXTR: float
    has_diabetes_med: int
    URXUCR: float
    RIAGENDR: int
    LBXSBU: float
    BPQ020: int
    BUN_Albumin_ratio: float
    DIET_DRXTP225: float
    DIQ010: int
    URXUMA: float
    MAP: float
    LBXSUA: float
    LBXBPB: float
    RIDAGEYR: int
    LBXBCD: float
    BMXBMI: float
    DIET_DRXT_V_LEGUMES: float
    BPXSY1: float
    Waist_Height_ratio: float
    DIET_DRXT_G_WHOLE: float
    LBX4PA: float
    DIET_DRXT_PF_SOY: float
    LBXSAL: float
    LBXBCC: float
    BMXWAIST: float
    URXUCD: float
    LBXTHG: float
    BPXDI1: float
    DIET_DRXTATOA: float
    DIET_DRXTM221: float
    RIDRETH1: int
    Pulse_Pressure: float
    has_ace_arb: int
    DIET_DRXTP184: float
    LBXGLU: float
    LBXGH: float
    total_medications: int

# endpoint
@app.post("/predict")
def predict(data: PatientData):

    # تحويل البيانات إلى array بنفس ترتيب التدريب
    features = np.array([
        data.has_nsaid,
        data.LBXHGB,
        data.LBXTC,
        data.LBXTR,
        data.has_diabetes_med,
        data.URXUCR,
        data.RIAGENDR,
        data.LBXSBU,
        data.BPQ020,
        data.BUN_Albumin_ratio,
        data.DIET_DRXTP225,
        data.DIQ010,
        data.URXUMA,
        data.MAP,
        data.LBXSUA,
        data.LBXBPB,
        data.RIDAGEYR,
        data.LBXBCD,
        data.BMXBMI,
        data.DIET_DRXT_V_LEGUMES,
        data.BPXSY1,
        data.Waist_Height_ratio,
        data.DIET_DRXT_G_WHOLE,
        data.LBX4PA,
        data.DIET_DRXT_PF_SOY,
        data.LBXSAL,
        data.LBXBCC,
        data.BMXWAIST,
        data.URXUCD,
        data.LBXTHG,
        data.BPXDI1,
        data.DIET_DRXTATOA,
        data.DIET_DRXTM221,
        data.RIDRETH1,
        data.Pulse_Pressure,
        data.has_ace_arb,
        data.DIET_DRXTP184,
        data.LBXGLU,
        data.LBXGH,
        data.total_medications
    ]).reshape(1, -1)

    # prediction
    prediction = model.predict(features)[0]

    # probability (if available)
    proba = model.predict_proba(features)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": proba
    }