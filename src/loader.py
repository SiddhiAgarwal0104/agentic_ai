import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_government_model")

def load_intent_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict_intent(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    conf_score, label_id = torch.max(probs, dim=-1)
    label = model.config.id2label[label_id.item()]


    urgency_words = ["urgent", "emergency", "asap", "jaldi", "serious", "kho gaya", "help"]
    is_urgent = any(word in text.lower() for word in urgency_words)

    return {
        "intent": label,
        "urgency": "high" if is_urgent else "low",
        "confidence_score": round(conf_score.item(), 4)
    }