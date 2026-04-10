import os
import joblib
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'features.pkl')

_model = None
_features = None

def load_artifacts():
    global _model, _features
    if _model is None or _features is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError("Model or Features file missing. Run train.py first.")
        _model = joblib.load(MODEL_PATH)
        _features = joblib.load(FEATURES_PATH)
    return _model, _features

# Massively expanded slang dictionary mapped to EXACT Kaggle columns
SYNONYM_MAP = {
    "pimple": "pus_filled_pimples", "pimples": "pus_filled_pimples", "breaking out": "pus_filled_pimples",
    "throw up": "vomiting", "throwing up": "vomiting", "puke": "vomiting", "barf": "vomiting",
    "tummy ache": "stomach_pain", "stomach ache": "stomach_pain", "belly ache": "belly_pain",
    "head hurts": "headache", "migraine": "headache", "head is pounding": "headache",
    "hot": "high_fever", "temperature": "high_fever", "feverish": "high_fever",
    "can't breathe": "breathlessness", "short of breath": "breathlessness", "gasping": "breathlessness",
    "dizzy": "dizziness", "spinning": "spinning_movements", "lightheaded": "dizziness",
    "rash": "skin_rash", "itch": "itching", "itchy": "itching", "scratching": "itching",
    "coughing": "cough", "hacked": "cough", "sneezing": "continuous_sneezing", "sneeze": "continuous_sneezing",
    "sad": "depression", "crying": "depression", "hopeless": "depression",
    "worried": "anxiety", "nervous": "anxiety", "panicking": "anxiety", "panic": "anxiety",
    "hurt": "pain", "ache": "pain", "aching": "joint_pain", "back pain": "back_pain",
    "drained": "fatigue", "exhausted": "fatigue", "tired": "fatigue", "sleepy": "lethargy",
    "runny nose": "runny_nose", "stuffy": "congestion"
}

def extract_features_from_text(user_text, features):
    """Reads free text and converts it into the 132-column binary array."""
    text = user_text.lower()
    
    # 1. Translate slang into Kaggle terms
    for slang, real_symptom in SYNONYM_MAP.items():
        # Using simple 'in' makes it catch "throwing up" much easier than strict regex
        if slang in text:
            text += f" {real_symptom.replace('_', ' ')} "

            
    extracted = []
    input_vector = np.zeros(len(features))
    
    # 2. Check which exact Kaggle features exist in the processed text
    for i, feature in enumerate(features):
        clean_feature = feature.replace('_', ' ')
        
        if clean_feature in text:
            input_vector[i] = 1
            if clean_feature.title() not in extracted:
                extracted.append(clean_feature.title())
                
    return input_vector, extracted

def predict_specialty(user_text):
    model, features = load_artifacts()
    
    input_vector, extracted_symptoms = extract_features_from_text(user_text, features)
    
    if len(extracted_symptoms) == 0:
        return {"error": "Could not identify specific medical symptoms from your text. Please be more descriptive."}
        
    # --- PRODUCTION SAFETY OVERRIDE ---
    # Since Kaggle data lacks mental health diseases, we force a manual override 
    # if extreme mental health keywords are the primary symptoms detected.
    mental_health_flags = ["Depression", "Anxiety"]
    if any(flag in extracted_symptoms for flag in mental_health_flags) and len(extracted_symptoms) <= 2:
        return {
            "predicted_specialty": "Psychiatrist",
            "confidence_score": 0.95,
            "top_3_predictions": [
                {"specialty": "Psychiatrist", "probability": 0.95},
                {"specialty": "General Physician", "probability": 0.04},
                {"specialty": "Neurologist", "probability": 0.01}
            ],
            "extracted_symptoms": extracted_symptoms
        }
    # ----------------------------------

    input_vector = input_vector.reshape(1, -1)
    
    probabilities = model.predict_proba(input_vector)[0]
    classes = model.classes_
    
    class_probs = list(zip(classes, probabilities))
    class_probs.sort(key=lambda x: x[1], reverse=True)
    
    top_specialty, top_confidence = class_probs[0]
    top_3 = [{"specialty": sp, "probability": prob} for sp, prob in class_probs[:3]]
    
    return {
        "predicted_specialty": top_specialty,
        "confidence_score": top_confidence,
        "top_3_predictions": top_3,
        "extracted_symptoms": extracted_symptoms
    }