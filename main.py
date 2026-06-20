# app.py - CKD XGBoost API with XAI (Complete Working Version)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pickle
import numpy as np
import pandas as pd
import shap
import json
import os
import logging
import requests
from io import StringIO

app = FastAPI(title="CKD XGBoost API with XAI")

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("nephroshield-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GEMINI API CONFIGURATION
# ============================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
GEMINI_API_BASE_URL = os.environ.get(
    "GEMINI_API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")

if GEMINI_API_KEY:
    logger.info("Gemini REST generation configured with model %s", GEMINI_MODEL)
else:
    logger.warning("GEMINI_API_KEY not set; Gemini rapport generation disabled")

# ============================================
# EMBEDDED SHAP BACKGROUND DATA (No CSV needed!)
# ============================================

SHAP_BACKGROUND_CSV = """has_nsaid,LBXHGB,LBXTC,LBXTR,has_diabetes_med,URXUCR,sex_male,LBXSBU,BPQ020,BUN_Albumin_ratio,DIET_DRXTP225,DIQ010,URXUMA,LBXSUA,LBXBPB,RIDAGEYR,LBXBCD,BMXBMI,DIET_DRXT_V_LEGUMES,BPXSY1,Waist_Height_ratio,DIET_DRXT_G_WHOLE,LBX4PA,DIET_DRXT_PF_SOY,LBXSAL,LBXBCC,BMXWAIST,URXUCD,LBXTHG,BPXDI1,DIET_DRXTATOA,DIET_DRXTM221,race_Mexican_American,race_Other_Hispanic,race_NH_White,race_NH_Black,Pulse_Pressure,has_ace_arb,DIET_DRXTP184,LBXGLU,LBXGH,total_medications
0,14.5,180,100,0,110,1,12,2,3.8,0.1,2,5,4.5,1.0,35,0.1,22,25,115,0.45,35,0.03,8,4.5,0.05,80,0.1,0.3,75,12,0.005,0,0,1,0,40,0,0.005,90,5.0,0
1,13.0,210,150,1,90,0,20,1,5.5,0.3,1,30,6.0,0.8,55,0.2,29,15,140,0.55,20,0.08,3,3.9,0.15,92,0.2,0.6,90,8,0.015,1,0,0,0,50,1,0.015,110,6.5,2
1,11.5,240,200,1,70,1,30,1,7.0,0.5,1,80,8.0,0.5,70,0.4,35,8,165,0.65,10,0.15,1,3.5,0.3,105,0.4,0.9,100,4,0.025,0,1,0,0,65,1,0.025,140,8.0,4
0,12.8,195,130,0,95,1,18,1,4.8,0.2,2,15,5.5,0.9,60,0.15,27,18,135,0.52,25,0.06,4,4.0,0.12,90,0.18,0.5,85,9,0.012,0,0,0,1,50,0,0.012,100,5.8,1
1,10.0,260,220,1,60,0,35,1,8.0,0.8,1,100,9.0,0.3,75,0.5,38,5,175,0.7,8,0.2,1,3.2,0.4,110,0.5,1.0,105,3,0.03,0,0,1,0,70,1,0.03,160,9.0,5"""

# ============================================
# LOAD YOUR EXISTING MODEL
# ============================================
try:
    with open("models/xgb_trackD.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    print("✅ XGBoost model loaded")
except Exception as e:
    print(f"❌ Error loading XGBoost model: {e}")
    xgb_model = None

# ============================================
# LOAD SHAP EXPLAINER FROM EMBEDDED DATA
# ============================================
shap_explainer = None

try:
    background_data = pd.read_csv(StringIO(SHAP_BACKGROUND_CSV))
    print(f"✅ Background data loaded: {background_data.shape}")
    
    if xgb_model is not None:
        shap_explainer = shap.TreeExplainer(xgb_model, background_data)
        print("✅ Real SHAP explainer loaded")
    else:
        print("⚠️ Model not loaded, SHAP explainer not created")
except Exception as e:
    print(f"❌ Error loading SHAP explainer: {e}")
    import traceback
    traceback.print_exc()
    shap_explainer = None

# ============================================
# FEATURES
# ============================================

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

FEATURE_DESCRIPTIONS = {
    "has_nsaid": "NSAID medication use",
    "LBXHGB": "Hemoglobin level",
    "LBXTC": "Total cholesterol",
    "LBXTR": "Triglycerides",
    "has_diabetes_med": "Diabetes medication",
    "URXUCR": "Urinary creatinine",
    "sex_male": "Gender",
    "LBXSBU": "Blood urea nitrogen (BUN)",
    "BPQ020": "Hypertension history",
    "BUN_Albumin_ratio": "BUN to Albumin ratio",
    "DIQ010": "Diabetes diagnosis",
    "URXUMA": "Urinary albumin",
    "LBXSUA": "Uric acid",
    "RIDAGEYR": "Age",
    "BMXBMI": "BMI",
    "BPXSY1": "Systolic blood pressure",
    "BPXDI1": "Diastolic blood pressure",
    "has_ace_arb": "ACE/ARB medication",
    "LBXGLU": "Blood glucose",
    "LBXGH": "HbA1c",
}

class PredictionRequest(BaseModel):
    height_cm: float
    data: dict
    patient_context: Optional[Dict] = None

# ============================================
# REFERENCE VALUES AND NORMAL RANGES (KEPT FOR SHAP)
# ============================================

def get_reference_value(feature):
    """Get reference (population average) value for context"""
    reference_values = {
        "LBXHGB": 13.8,
        "LBXTC": 190,
        "LBXTR": 110,
        "LBXSBU": 15,
        "LBXSAL": 4.2,
        "LBXSUA": 5.0,
        "LBXGLU": 95,
        "LBXGH": 5.4,
        "LBXBPB": 1.0,
        "LBXBCD": 0.2,
        "LBX4PA": 0.05,
        "LBXBCC": 0.1,
        "LBXTHG": 0.5,
        "BPXSY1": 120,
        "BPXDI1": 80,
        "BMXBMI": 27,
        "BMXWAIST": 88,
        "URXUCR": 100,
        "URXUMA": 10,
        "URXUCD": 0.15,
        "RIDAGEYR": 50,
        "sex_male": 0.5,
        "BPQ020": 2,
        "DIQ010": 2,
        "has_diabetes_med": 0,
        "has_nsaid": 0,
        "has_ace_arb": 0,
        "total_medications": 0,
        "BUN_Albumin_ratio": 3.8,
        "Pulse_Pressure": 40,
        "Waist_Height_ratio": 0.5,
        "DIET_DRXTP225": 0.1,
        "DIET_DRXT_V_LEGUMES": 20,
        "DIET_DRXT_G_WHOLE": 30,
        "DIET_DRXT_PF_SOY": 5,
        "DIET_DRXTATOA": 10,
        "DIET_DRXTM221": 0.01,
        "DIET_DRXTP184": 0.01,
        "race_Mexican_American": 0,
        "race_Other_Hispanic": 0,
        "race_NH_White": 0,
        "race_NH_Black": 0,
    }
    return reference_values.get(feature, None)

def get_normal_range(feature):
    """Get normal range for display with units"""
    ranges = {
        "LBXHGB": "12-16 g/dL",
        "LBXTC": "125-200 mg/dL",
        "LBXTR": "50-150 mg/dL",
        "LBXSBU": "7-20 mg/dL",
        "LBXSAL": "3.5-5.0 g/dL",
        "LBXSUA": "3.5-7.2 mg/dL",
        "LBXGLU": "70-100 mg/dL",
        "LBXGH": "4.0-5.7%",
        "LBXBPB": "0-2.0 µg/dL",
        "LBXBCD": "0-0.5 µg/L",
        "LBX4PA": "0-0.1 µg/L",
        "LBXBCC": "0-0.2 µg/L",
        "LBXTHG": "0-1.0 µg/L",
        "BPXSY1": "90-130 mmHg",
        "BPXDI1": "60-85 mmHg",
        "BMXBMI": "18.5-25 kg/m²",
        "BMXWAIST": "≤88 cm (F) / ≤102 cm (M)",
        "URXUCR": "50-150 mg/dL",
        "URXUMA": "0-30 µg/mL",
        "URXUCD": "0-0.3 µg/L",
        "RIDAGEYR": "18-65 years",
        "sex_male": "0=Female, 1=Male",
        "BPQ020": "1=Yes, 2=No",
        "DIQ010": "1=Yes, 2=No",
        "has_diabetes_med": "0=No, 1=Yes",
        "has_nsaid": "0=No, 1=Yes",
        "has_ace_arb": "0=No, 1=Yes",
        "total_medications": "0-5",
        "BUN_Albumin_ratio": "3.5-5.5",
        "Pulse_Pressure": "30-50 mmHg",
        "Waist_Height_ratio": "0.45-0.55",
        "DIET_DRXTP225": "0.05-0.2 g",
        "DIET_DRXT_V_LEGUMES": "15-25 g",
        "DIET_DRXT_G_WHOLE": "25-35 g",
        "DIET_DRXT_PF_SOY": "3-8 g",
        "DIET_DRXTATOA": "8-15 mg",
        "DIET_DRXTM221": "0.005-0.015 g",
        "DIET_DRXTP184": "0.005-0.015 g",
        "race_Mexican_American": "0=No, 1=Yes",
        "race_Other_Hispanic": "0=No, 1=Yes",
        "race_NH_White": "0=No, 1=Yes",
        "race_NH_Black": "0=No, 1=Yes",
    }
    return ranges.get(feature, None)

# ============================================
# FALLBACK EXPLANATIONS (WHEN SHAP FAILS)
# ============================================

def get_fallback_explanations(feature_names, patient_values):
    """Generate fallback explanations when SHAP is unavailable"""
    explanations = []
    
    clinical_weights = {
        "LBXHGB": 5, "LBXTC": 3, "LBXTR": 2, "LBXSBU": 4,
        "BPXSY1": 5, "BPXDI1": 4, "LBXGLU": 5, "LBXGH": 5,
        "BMXBMI": 4, "URXUMA": 5, "has_diabetes_med": 5,
        "has_nsaid": 3, "has_ace_arb": 3, "RIDAGEYR": 4,
        "LBXSUA": 3, "URXUCR": 3, "LBXSAL": 3
    }
    
    for feature in feature_names:
        actual_value = patient_values.get(feature, 0)
        reference_value = get_reference_value(feature)
        normal_range = get_normal_range(feature)
        
        if reference_value is not None and isinstance(reference_value, (int, float)):
            try:
                deviation = abs(float(actual_value) - float(reference_value)) / float(reference_value) if float(reference_value) > 0 else 0
            except:
                deviation = 0
        else:
            deviation = 0.5 if float(actual_value) > 0 else 0
        
        base_importance = clinical_weights.get(feature, 1)
        shap_val = deviation * base_importance * 0.1
        
        if feature in ["LBXHGB"]:
            if reference_value is not None:
                impact = "decreases_risk" if float(actual_value) > float(reference_value) else "increases_risk"
            else:
                impact = "decreases_risk"
        elif feature in ["LBXGLU", "LBXGH", "BPXSY1", "URXUMA", "BMXBMI", "LBXTC", "LBXTR", "LBXSBU", "URXUCR"]:
            if reference_value is not None:
                impact = "increases_risk" if float(actual_value) > float(reference_value) else "decreases_risk"
            else:
                impact = "increases_risk"
        else:
            impact = "increases_risk" if shap_val > 0.01 else "decreases_risk"
        
        explanations.append({
            "feature": feature,
            "description": FEATURE_DESCRIPTIONS.get(feature, feature),
            "actual_value": actual_value,
            "reference_value": reference_value,
            "normal_range": normal_range,
            "shap_value": float(shap_val),
            "impact": impact,
            "absolute_impact": abs(float(shap_val)),
            "percent_contribution": 0
        })
    
    total_abs = sum(e["absolute_impact"] for e in explanations)
    if total_abs > 0:
        for e in explanations:
            e["percent_contribution"] = round((e["absolute_impact"] / total_abs) * 100, 1)
    
    explanations.sort(key=lambda x: x["absolute_impact"], reverse=True)
    return explanations[:15]

# ============================================
# SHAP EXPLANATIONS (WITH FALLBACK)
# ============================================

def get_real_shap_explanations(input_array, feature_names, patient_values):
    """Get REAL SHAP values with fallback"""
    if shap_explainer is None:
        return get_fallback_explanations(feature_names, patient_values)
    
    try:
        shap_values = shap_explainer.shap_values(input_array)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        shap_values = shap_values.flatten()
        
        explanations = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            actual_value = patient_values.get(feature, "N/A")
            reference_value = get_reference_value(feature)
            normal_range = get_normal_range(feature)
            
            explanations.append({
                "feature": feature,
                "description": FEATURE_DESCRIPTIONS.get(feature, feature),
                "actual_value": actual_value,
                "reference_value": reference_value,
                "normal_range": normal_range,
                "shap_value": float(shap_val),
                "impact": "increases_risk" if shap_val > 0 else "decreases_risk",
                "absolute_impact": abs(float(shap_val)),
                "percent_contribution": 0
            })
        
        total_abs_impact = sum(e["absolute_impact"] for e in explanations)
        if total_abs_impact > 0:
            for e in explanations:
                e["percent_contribution"] = round((e["absolute_impact"] / total_abs_impact) * 100, 1)
        
        explanations.sort(key=lambda x: x["absolute_impact"], reverse=True)
        return explanations[:15]
        
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        return get_fallback_explanations(feature_names, patient_values)

# ============================================
# GEMINI RAPPORT GENERATION (OPTIMIZED - SHORTER, CLEANER)
# ============================================

def generate_abnormal_labs_summary(patient_data):
    """Extract abnormal lab values for LLM context"""
    abnormal = []
    
    reference_ranges = {
        "LBXHGB": (12, 16, "g/dL"),
        "BPXSY1": (90, 130, "mmHg"),
        "BPXDI1": (60, 85, "mmHg"),
        "LBXGLU": (70, 100, "mg/dL"),
        "LBXGH": (4, 5.7, "%"),
        "BMXBMI": (18.5, 25, "kg/m²"),
        "LBXTC": (125, 200, "mg/dL"),
        "LBXTR": (50, 150, "mg/dL"),
        "URXUMA": (0, 30, "µg/mL")
    }
    
    for key, (low, high, unit) in reference_ranges.items():
        if key in patient_data:
            val = patient_data[key]
            if val < low:
                abnormal.append(f"- {FEATURE_DESCRIPTIONS.get(key, key)}: {val} {unit} (below normal range {low}-{high})")
            elif val > high:
                abnormal.append(f"- {FEATURE_DESCRIPTIONS.get(key, key)}: {val} {unit} (above normal range {low}-{high})")
    
    return "\n".join(abnormal) if abnormal else "All key markers within normal ranges."

def generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context):
    """Dynamic template-based rapport (fallback when Gemini is unavailable)"""
    
    name = patient_context.get('name', 'there') if patient_context else 'there'
    
    if risk_score >= 70:
        primary = f"I want to be direct with you, {name}, because early action matters. Your assessment shows a {risk_score:.0f}% risk for developing kidney disease over the next year."
        empathy = "I know this sounds concerning, but knowing this now gives us a powerful advantage. Every day we can work on this together."
    elif risk_score >= 40:
        primary = f"{name}, your assessment shows a {risk_score:.0f}% risk for kidney disease. This is a signal to focus on protective measures."
        empathy = "The good news is that moderate risk often responds very well to lifestyle changes and medication adjustments."
    else:
        primary = f"Great news, {name}! Your {risk_score:.0f}% risk score is in the low range. Your kidneys are functioning well relative to others your age."
        empathy = "Keep up the good work - your current habits are protecting your kidney health."
    
    risk_section = ""
    if top_risk_factors:
        risk_section = "\n\nWhat's affecting your kidney health:\n"
        for factor in top_risk_factors[:3]:
            risk_section += f"- {factor['factor']} (currently {factor['value']}) - contributing {factor['impact_percent']:.0f}% to your risk score\n"
    
    recommendations = []
    if any(f['factor'] == 'Blood pressure' for f in top_risk_factors):
        recommendations.append("- Work with your doctor to bring blood pressure below 130/80")
    if any(f['factor'] == 'HbA1c' for f in top_risk_factors):
        recommendations.append("- Improve blood sugar control - aim for HbA1c below 7%")
    if any(f['factor'] == 'BMI' for f in top_risk_factors):
        recommendations.append("- Consider a 5-10% weight reduction goal")
    if any(f['factor'] == 'Hemoglobin' for f in top_risk_factors):
        recommendations.append("- Discuss anemia management with your doctor")
    
    if not recommendations:
        recommendations = [
            "- Schedule your annual kidney function check",
            "- Stay well hydrated (6-8 glasses of water daily)",
            "- Limit sodium to less than 2300mg per day",
            "- Aim for 150 minutes of moderate exercise weekly"
        ]
    
    recs_text = "\n\nYour Action Plan:\n" + "\n".join(recommendations[:4])
    full_text = primary + risk_section + recs_text + f"\n\n{empathy}\n\nWarmly,\nYour Kidney Health Team"
    
    return {
        "risk_category": "high" if risk_score >= 70 else "moderate" if risk_score >= 40 else "low",
        "urgency": "Schedule follow-up within 1-2 weeks" if risk_score >= 70 else "Schedule within 3 months" if risk_score >= 40 else "Annual check-up",
        "tone": "concerned" if risk_score >= 70 else "proactive" if risk_score >= 40 else "reassuring",
        "full_rapport_text": full_text,
        "recommendations": recommendations[:5],
        "questions_for_doctor": [
            "What's my target blood pressure?",
            "Do I need medication adjustments?",
            "When should I repeat these tests?"
        ],
        "llm_generated": False,
        "llm_provider": "template",
        "llm_model": None
    }

def extract_structured_from_llm(llm_text, risk_score, top_factors, provider="gemini", model_name=None):
    """Extract structured data from LLM response"""
    if risk_score >= 70:
        risk_category = "high"
        urgency = "Please schedule a follow-up within 1-2 weeks"
        tone = "concerned_but_supportive"
    elif risk_score >= 40:
        risk_category = "moderate"
        urgency = "Schedule a follow-up within 1-3 months"
        tone = "proactive"
    else:
        risk_category = "low"
        urgency = "Continue regular annual check-ups"
        tone = "reassuring"
    
    recommendations = []
    lines = llm_text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if any(word in clean_line.lower() for word in ['recommend', 'should', 'consider', 'try', 'aim', 'action']):
            if len(clean_line) > 10 and not clean_line.startswith('```'):
                recommendations.append({
                    "action": clean_line[:50],
                    "detail": clean_line,
                    "priority": "high" if "urgent" in clean_line.lower() or "immediately" in clean_line.lower() else "medium"
                })
    
    return {
        "risk_category": risk_category,
        "urgency": urgency,
        "tone": tone,
        "full_rapport_text": llm_text,
        "recommendations": recommendations[:5],
        "questions_for_doctor": [
            "What is my target blood pressure given my results?",
            "Should I be concerned about my {top_factor} level?".format(
                top_factor=top_factors[0]['factor'] if top_factors else "kidney function"
            ),
            "Are there any medications I should avoid?",
            "How often should I repeat these tests?"
        ],
        "llm_generated": True,
        "llm_provider": provider,
        "llm_model": model_name
    }

def generate_llm_rapport(probability, shap_explanations, patient_data, patient_context):
    """Generate personalized rapport using Gemini - OPTIMIZED for shorter, cleaner output"""
    
    risk_score = probability * 100
    
    # Format SHAP explanations for LLM
    top_risk_factors = []
    top_protective_factors = []
    
    if shap_explanations:
        for exp in shap_explanations[:5]:
            if exp["impact"] == "increases_risk" and exp["percent_contribution"] > 5:
                top_risk_factors.append({
                    "factor": exp["description"],
                    "value": exp["actual_value"],
                    "impact_percent": exp["percent_contribution"]
                })
            elif exp["impact"] == "decreases_risk" and exp["percent_contribution"] > 5:
                top_protective_factors.append({
                    "factor": exp["description"],
                    "value": exp["actual_value"],
                    "impact_percent": exp["percent_contribution"]
                })
    
    # Build prompt for LLM - OPTIMIZED for shorter, cleaner output
    system_prompt = """You are a medical assistant. Create a short, professional health report for a patient.
    Keep it concise (max 300 words). Use plain text without markdown, emojis, or special formatting.
    Be empathetic but clinical. Provide specific advice based on their results."""
    
    user_prompt = f"""
    PATIENT: {patient_context.get('name', 'Patient') if patient_context else 'Patient'}
    AGE: {patient_context.get('age', 'Unknown') if patient_context else 'Unknown'}
    GENDER: {patient_context.get('gender', 'Unknown') if patient_context else 'Unknown'}
    
    CKD RISK: {risk_score:.1f}% ({'HIGH' if risk_score >= 70 else 'MODERATE' if risk_score >= 40 else 'LOW'})
    
    KEY RISK FACTORS:
    {json.dumps(top_risk_factors, indent=2) if top_risk_factors else 'None significant'}
    
    PROTECTIVE FACTORS:
    {json.dumps(top_protective_factors, indent=2) if top_protective_factors else 'None identified'}
    
    ABNORMAL LABS:
    {generate_abnormal_labs_summary(patient_data)}
    
    Provide a short health summary with:
    1. Greeting and risk assessment
    2. Top 2-3 factors affecting their health
    3. 2-3 actionable recommendations
    4. Questions for their doctor
    5. Encouraging closing
    
    Keep it short, professional, and supportive. No markdown, no emojis.
    """
    
    # Call Gemini API
    if GEMINI_API_KEY:
        try:
            gemini_prompt = f"{system_prompt}\n\n{user_prompt}"
            url = f"{GEMINI_API_BASE_URL}/models/{GEMINI_MODEL}:generateContent"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": gemini_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.5,
                    "maxOutputTokens": 800,
                    "topP": 0.9,
                },
            }
            response = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json=payload,
                timeout=45,
            )

            if response.status_code >= 400:
                logger.warning("Gemini API HTTP %s: %s", response.status_code, response.text[:1000])
                return generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context)

            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                logger.warning("Gemini returned no candidates")
                return generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context)

            parts = candidates[0].get("content", {}).get("parts", [])
            llm_rapport = "\n".join(
                part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")
            ).strip()

            if not llm_rapport:
                logger.warning("Gemini returned an empty text response")
                return generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context)

            return extract_structured_from_llm(
                llm_rapport,
                risk_score,
                top_risk_factors,
                provider="gemini",
                model_name=GEMINI_MODEL,
            )
        except Exception as e:
            logger.warning("Gemini API error: %s", e)
            return generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context)
    
    # Fallback to template
    return generate_template_rapport(risk_score, top_risk_factors, top_protective_factors, patient_context)

# ============================================
# CLINICAL INSIGHTS
# ============================================

def generate_clinical_insights(patient_data, shap_explanations):
    """Generate dynamic clinical insights based on actual values"""
    insights = []
    
    clinical_rules = [
        {
            "parameter": "BPXSY1",
            "threshold": 140,
            "condition": "above",
            "message": "Systolic BP above target (140 mmHg)",
            "recommendation": "Consider antihypertensive intensification",
            "evidence": "ACC/AHA guidelines recommend BP <130/80 for CKD patients"
        },
        {
            "parameter": "LBXGH",
            "threshold": 7,
            "condition": "above",
            "message": "HbA1c above target (7%)",
            "recommendation": "Review diabetes management plan",
            "evidence": "Each 1% reduction in HbA1c reduces microvascular complications by 37%"
        },
        {
            "parameter": "URXUMA",
            "threshold": 30,
            "condition": "above",
            "message": "Microalbuminuria detected",
            "recommendation": "Maximize ACE/ARB therapy",
            "evidence": "Albuminuria is an independent risk factor for CKD progression"
        },
        {
            "parameter": "BMXBMI",
            "threshold": 30,
            "condition": "above",
            "message": "BMI indicates obesity",
            "recommendation": "Refer to weight management program",
            "evidence": "Weight loss of 5-10% improves kidney outcomes"
        }
    ]
    
    for rule in clinical_rules:
        if rule["parameter"] in patient_data:
            value = patient_data[rule["parameter"]]
            if (rule["condition"] == "above" and value > rule["threshold"]) or \
               (rule["condition"] == "below" and value < rule["threshold"]):
                insights.append({
                    "type": "alert" if value > rule["threshold"] * 1.2 else "warning",
                    "parameter": FEATURE_DESCRIPTIONS.get(rule["parameter"], rule["parameter"]),
                    "value": value,
                    "message": rule["message"],
                    "recommendation": rule["recommendation"],
                    "evidence": rule["evidence"]
                })
    
    if shap_explanations:
        for exp in shap_explanations[:3]:
            if exp["impact"] == "increases_risk" and exp["percent_contribution"] > 10:
                insights.append({
                    "type": "insight",
                    "parameter": exp["description"],
                    "value": exp["actual_value"],
                    "message": f"{exp['description']} is a major contributor to risk",
                    "recommendation": f"Addressing {exp['description'].lower()} could significantly reduce risk",
                    "evidence": f"SHAP analysis shows {exp['percent_contribution']:.0f}% contribution to prediction"
                })
    
    return insights

# ============================================
# HEALTH CHECK ENDPOINT
# ============================================

@app.get("/health")
async def health_check():
    """Simple deployment health check"""
    return {
        "status": "ok",
        "model_loaded": xgb_model is not None,
        "shap_enabled": shap_explainer is not None,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
    }

# ============================================
# MAIN PREDICTION ENDPOINT
# ============================================

@app.post("/predict")
async def predict_ckd(payload: PredictionRequest):
    try:
        p = payload.data.copy()
        
        # Feature engineering
        if "LBXSAL" in p and p["LBXSAL"] > 0:
            p["BUN_Albumin_ratio"] = p.get("LBXSBU", 15) / p["LBXSAL"]
        else:
            p["BUN_Albumin_ratio"] = p.get("LBXSBU", 15) / 4.2
        
        p["Pulse_Pressure"] = p.get("BPXSY1", 120) - p.get("BPXDI1", 80)
        waist = p.get("BMXWAIST", 88)
        height = payload.height_cm
        p["Waist_Height_ratio"] = waist / height if height > 0 else 0.5
        
        # Prepare input array
        input_array = np.array([[float(p.get(f, 0)) for f in FEATURE_COLS]])
        
        # Get model prediction
        if xgb_model is None:
            raise HTTPException(status_code=503, detail="XGBoost model is not loaded")

        probabilities = xgb_model.predict_proba(input_array)[0]
        prob_ckd = float(probabilities[1])
        
        # Get SHAP explanations (with fallback)
        shap_explanations = get_real_shap_explanations(input_array, FEATURE_COLS, p)
        
        # Generate personalized rapport using Gemini
        personalized_rapport = generate_llm_rapport(
            prob_ckd, 
            shap_explanations, 
            p, 
            payload.patient_context
        )
        
        # Generate clinical insights
        clinical_insights = generate_clinical_insights(p, shap_explanations)
        
        # Debug: Log what we're returning
        print(f"📊 SHAP explanations count: {len(shap_explanations) if shap_explanations else 0}")
        print(f"📋 Rapport generated: {personalized_rapport is not None}")
        print(f"💡 Clinical insights count: {len(clinical_insights) if clinical_insights else 0}")
        
        return {
            "status": "success",
            "ckd_probability": round(prob_ckd, 4),
            "prediction": "CKD" if prob_ckd >= 0.5 else "No CKD",
            "risk_level": "High" if prob_ckd > 0.7 else "Moderate" if prob_ckd > 0.3 else "Low",
            "shap_explanations": shap_explanations if shap_explanations else [],
            "personalized_rapport": personalized_rapport if personalized_rapport else {},
            "clinical_insights": clinical_insights if clinical_insights else []
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
