import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models")


def load_intent_model():
    print(f" Loading Bharat Sahayak Intent Brain from: {MODEL_PATH}")
    
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model folder not found at {MODEL_PATH}. Check your directory structure!")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
  
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    

    model.eval()
    
    print("Model loaded successfully!")
    return tokenizer, model

def predict_intent(text, tokenizer, model):
 
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    

    with torch.no_grad(): 
        outputs = model(**inputs)
   
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
 
    conf_score, label_id = torch.max(probabilities, dim=-1)
    
 
    label = model.config.id2label[label_id.item()]
    
    return {
        "intent": label,
        "confidence_score": round(conf_score.item(), 4)
    }