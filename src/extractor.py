import spacy


nlp = spacy.load("en_core_web_sm")

def extract_entities(text):

    doc = nlp(text)
    

    entities = {
        "name": None,
        "location": None,
        "org": None,
        "misc": []
    }
    

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities["name"] = ent.text
        elif ent.label_ == "GPE": 
            entities["location"] = ent.text
        elif ent.label_ == "ORG":
            entities["org"] = ent.text
        else:
            entities["misc"].append({ent.text: ent.label_})
            
    return entities


test_text = "Mera naam Tanya Bora hai aur main Kanpur mein rehti hoon."
print(extract_entities(test_text))
