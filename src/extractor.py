import spacy
import re

nlp = spacy.load("xx_ent_wiki_sm")

def extract_entities(text):
    doc = nlp(text)
    entities = {"name": None, "location": None, "age": None}
    

    numbers = re.findall(r'\b\d{1,2}\b', text)
    if numbers:
        entities["age"] = int(numbers[0])

    for ent in doc.ents:
        if ent.label_ == "PER" or ent.label_ == "PERSON":
            entities["name"] = ent.text
        elif ent.label_ in ["LOC", "GPE"]:
            entities["location"] = ent.text
            
    return entities