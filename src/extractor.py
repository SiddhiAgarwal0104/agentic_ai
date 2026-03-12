import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extracts entities like name, location, organization from text
    using spaCy NER.
    """

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

        elif ent.label_ == "GPE":   # Geopolitical entity
            entities["location"] = ent.text

        elif ent.label_ == "ORG":
            entities["org"] = ent.text

        else:
            entities["misc"].append({ent.text: ent.label_})

    return entities


# Run test only when file is executed directly
if __name__ == "__main__":
    test_text = "Mera naam Tanya Bora hai aur main Kanpur mein rehti hoon."
    print(extract_entities(test_text))