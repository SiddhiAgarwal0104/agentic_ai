import re
import string

def preprocessing_node(state):
    text = state["user_input"]
    

    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    

    text = text.strip()
    

    hinglish_map = {"mera": "my", "naam": "name", "hai": "is"}
    words = text.split()
    normalized = [hinglish_map.get(w.lower(), w) for w in words]
    
    return {"clean_text": " ".join(normalized)}