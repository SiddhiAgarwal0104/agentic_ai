import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

def predict_resolution(intent, urgency):
    base_time = 24
    multiplier = 2.0 if urgency == "high" else 1.0
    
    intent_weights = {"pension": 1.5, "identity": 1.2, "education": 1.0}
    weight = intent_weights.get(intent.lower(), 1.0)
    
    predicted_hours = base_time * weight * multiplier
    return round(predicted_hours, 1)

def recommend_escalation(intent, urgency):
    if urgency.lower() == "high" and intent.lower() in ["pension", "legal"]:
        return "Escalate to Human Supervisor"
    
    return "Handled by AI"