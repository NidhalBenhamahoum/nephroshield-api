"""
CKD Model Test Script — xgb_trackD & logreg_trackD
====================================================
Both models expect 42 features in this exact order.
XGBoost  → raw values (scale-invariant, ready to use)
LogReg   → requires StandardScaler from training
"""

import pickle
import warnings
import numpy as np
import sys
import io

# --- Fix for Windows Encoding Issues ---
# This ensures that if any non-standard characters appear, the script won't crash.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings("ignore")

# ─── Load models ──────────────────────────────────────────────────────────────
XGB_PATH   = "models/xgb_trackD.pkl"
LR_PATH    = "models/logreg_trackD.pkl"
# SCALER_PATH = "models/scaler_trackD.pkl"   # Needed for LogReg

try:
    with open(XGB_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    with open(LR_PATH, "rb") as f:
        lr_model = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: Model file not found. {e}")
    sys.exit(1)

# ─── Feature list (42 features, ORDER IS STRICT) ──────────────────────────────
FEATURE_COLS = [
    "has_nsaid",            #  0  binary (0/1)
    "LBXHGB",               #  1  Hemoglobin (g/dL)
    "LBXTC",                #  2  Total cholesterol (mg/dL)
    "LBXTR",                #  3  Triglycerides (mg/dL)
    "has_diabetes_med",     #  4  binary (0/1)
    "URXUCR",               #  5  Urine creatinine (mg/dL)
    "sex_male",             #  6  binary (1=male, 0=female)
    "LBXSBU",               #  7  BUN (mg/dL)
    "BPQ020",               #  8  Hypertension dx: 1=yes, 2=no
    "BUN_Albumin_ratio",    #  9  ENGINEERED: LBXSBU / LBXSAL
    "DIET_DRXTP225",       # 10  Dietary EPA omega-3 (g)
    "DIQ010",               # 11  Diabetes dx: 1=yes, 2=no
    "URXUMA",               # 12  Urine albumin / microalbumin (mg/L)
    "LBXSUA",               # 13  Serum uric acid (mg/dL)
    "LBXBPB",               # 14  Blood lead (ug/dL)
    "RIDAGEYR",             # 15  Age (years)
    "LBXBCD",               # 16  Blood cadmium (ug/L)
    "BMXBMI",               # 17  BMI (kg/m²)
    "DIET_DRXT_V_LEGUMES", # 18  Legume intake (g)
    "BPXSY1",               # 19  Systolic BP (mmHg)
    "Waist_Height_ratio",   # 20  ENGINEERED: BMXWAIST / height_cm
    "DIET_DRXT_G_WHOLE",   # 21  Whole grain intake (g)
    "LBX4PA",               # 22  Blood polycyclic aromatic hydrocarbon
    "DIET_DRXT_PF_SOY",    # 23  Soy protein intake (g)
    "LBXSAL",               # 24  Serum albumin (g/dL)
    "LBXBCC",               # 25  Blood cobalt/chromium (ug/L)
    "BMXWAIST",             # 26  Waist circumference (cm)
    "URXUCD",               # 27  Urine cadmium (ug/L)
    "LBXTHG",               # 28  Blood total mercury (ug/L)
    "BPXDI1",               # 29  Diastolic BP (mmHg)
    "DIET_DRXTATOA",       # 30  Dietary vitamin E / alpha-tocopherol (mg)
    "DIET_DRXTM221",       # 31  Dietary fatty acid 22:1 (g)
    "race_Mexican_American",# 32 binary (0/1)
    "race_Other_Hispanic", # 33  binary (0/1)
    "race_NH_White",       # 34  binary (0/1)
    "race_NH_Black",       # 35  binary (0/1)
    "Pulse_Pressure",       # 36  ENGINEERED: BPXSY1 - BPXDI1
    "has_ace_arb",         # 37  binary (0/1)
    "DIET_DRXTP184",       # 38  Dietary fatty acid 18:4 (g)
    "LBXGLU",               # 39  Fasting glucose (mg/dL)
    "LBXGH",                # 40  HbA1c / glycohemoglobin (%)
    "total_medications",   # 41  Total medication count (integer)
]

# ─── Helper: build input vector from a patient dict ───────────────────────────
def build_input(patient: dict, height_cm: float) -> np.ndarray:
    p = patient.copy()
    p["BUN_Albumin_ratio"]  = p["LBXSBU"] / p["LBXSAL"]
    p["Pulse_Pressure"]     = p["BPXSY1"] - p["BPXDI1"]
    p["Waist_Height_ratio"] = p["BMXWAIST"] / height_cm
    return np.array([[p[f] for f in FEATURE_COLS]])

def predict(x: np.ndarray) -> dict:
    prob = xgb_model.predict_proba(x)[0]
    return {
        "prob_no_ckd": round(float(prob[0]), 4),
        "prob_ckd":    round(float(prob[1]), 4),
        "prediction":  "CKD" if prob[1] >= 0.5 else "No CKD",
    }

# ─── Test cases ───────────────────────────────────────────────────────────────
test_patients = [
    {
        "label": "High-Risk: 65yo NH-Black male, diabetic + hypertensive",
        "height_cm": 175,
        "data": {
            "has_nsaid": 1, "LBXHGB": 12.5, "LBXTC": 200, "LBXTR": 150,
            "has_diabetes_med": 1, "URXUCR": 120, "sex_male": 1, "LBXSBU": 25,
            "BPQ020": 1, "DIET_DRXTP225": 0, "DIQ010": 1, "URXUMA": 35,
            "LBXSUA": 7.2, "LBXBPB": 2.1, "RIDAGEYR": 65, "LBXBCD": 0.4,
            "BMXBMI": 30, "DIET_DRXT_V_LEGUMES": 0, "BPXSY1": 145,
            "DIET_DRXT_G_WHOLE": 0, "LBX4PA": 0.1, "DIET_DRXT_PF_SOY": 0,
            "LBXSAL": 3.8, "LBXBCC": 0.3, "BMXWAIST": 102, "URXUCD": 0.2,
            "LBXTHG": 1.2, "BPXDI1": 88, "DIET_DRXTATOA": 0, "DIET_DRXTM221": 0,
            "race_Mexican_American": 0, "race_Other_Hispanic": 0,
            "race_NH_White": 0, "race_NH_Black": 1,
            "has_ace_arb": 1, "DIET_DRXTP184": 0,
            "LBXGLU": 140, "LBXGH": 7.1, "total_medications": 4,
        }
    },
    {
        "label": "Moderate-Risk: 52yo NH-White female, hypertensive",
        "height_cm": 163,
        "data": {
            "has_nsaid": 0, "LBXHGB": 13.2, "LBXTC": 215, "LBXTR": 180,
            "has_diabetes_med": 0, "URXUCR": 90, "sex_male": 0, "LBXSBU": 18,
            "BPQ020": 1, "DIET_DRXTP225": 0, "DIQ010": 2, "URXUMA": 18,
            "LBXSUA": 5.8, "LBXBPB": 1.2, "RIDAGEYR": 52, "LBXBCD": 0.2,
            "BMXBMI": 27, "DIET_DRXT_V_LEGUMES": 0, "BPXSY1": 138,
            "DIET_DRXT_G_WHOLE": 0, "LBX4PA": 0.05, "DIET_DRXT_PF_SOY": 0,
            "LBXSAL": 4.2, "LBXBCC": 0.15, "BMXWAIST": 88, "URXUCD": 0.15,
            "LBXTHG": 0.8, "BPXDI1": 82, "DIET_DRXTATOA": 0, "DIET_DRXTM221": 0,
            "race_Mexican_American": 0, "race_Other_Hispanic": 0,
            "race_NH_White": 1, "race_NH_Black": 0,
            "has_ace_arb": 0, "DIET_DRXTP184": 0,
            "LBXGLU": 98, "LBXGH": 5.8, "total_medications": 1,
        }
    },
    {
        "label": "Low-Risk: 28yo healthy NH-White male, no meds",
        "height_cm": 170,
        "data": {
            "has_nsaid": 0, "LBXHGB": 15.5, "LBXTC": 180, "LBXTR": 95,
            "has_diabetes_med": 0, "URXUCR": 100, "sex_male": 1, "LBXSBU": 12,
            "BPQ020": 2, "DIET_DRXTP225": 0, "DIQ010": 2, "URXUMA": 5,
            "LBXSUA": 4.5, "LBXBPB": 0.8, "RIDAGEYR": 28, "LBXBCD": 0.1,
            "BMXBMI": 22, "DIET_DRXT_V_LEGUMES": 0, "BPXSY1": 115,
            "DIET_DRXT_G_WHOLE": 0, "LBX4PA": 0.02, "DIET_DRXT_PF_SOY": 0,
            "LBXSAL": 4.5, "LBXBCC": 0.1, "BMXWAIST": 78, "URXUCD": 0.1,
            "LBXTHG": 0.5, "BPXDI1": 75, "DIET_DRXTATOA": 0, "DIET_DRXTM221": 0,
            "race_Mexican_American": 0, "race_Other_Hispanic": 0,
            "race_NH_White": 1, "race_NH_Black": 0,
            "has_ace_arb": 0, "DIET_DRXTP184": 0,
            "LBXGLU": 85, "LBXGH": 5.1, "total_medications": 0,
        }
    },
]

# ─── Run tests ────────────────────────────────────────────────────────────────
print("=" * 65)
print("  CKD MODEL TEST - XGBoost Track D")
print("=" * 65)

for case in test_patients:
    x = build_input(case["data"], case["height_cm"])
    result = predict(x)
    
    # Using ASCII symbols for maximum compatibility
    icon = "[!]" if result["prediction"] == "CKD" else "[+]"
    
    print(f"\n{icon} {case['label']}")
    print(f"    P(CKD) = {result['prob_ckd']:.4f}   -->   {result['prediction']}")

print("\n" + "=" * 65)
print("  NOTE: LogReg (logreg_trackD)")
print("=" * 65)
print("""
  logreg_trackD was trained on STANDARDIZED (z-scored) features.
  The StandardScaler object is required for LogReg to work correctly.

  XGBoost is tree-based (scale-invariant) and works on these raw values.
""")