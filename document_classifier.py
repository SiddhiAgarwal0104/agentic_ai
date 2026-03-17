def classify_document(text):

    if "Government of India" in text:
        return "Aadhar Card"

    elif "Permanent Account Number" in text:
        return "PAN Card"

    elif "Ration Card" in text:
        return "Ration Card"

    return "Unknown Document"