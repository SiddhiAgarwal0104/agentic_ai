import re

def extract_fields(text):

    name = re.findall(r"Name\s*:\s*(.*)", text)

    dob = re.findall(r"\d{2}/\d{2}/\d{4}", text)

    aadhar = re.findall(r"\d{4}\s\d{4}\s\d{4}", text)

    pan = re.findall(r"[A-Z]{5}[0-9]{4}[A-Z]", text)

    return {
        "name": name[0] if name else None,
        "dob": dob[0] if dob else None,
        "aadhar_number": aadhar[0] if aadhar else None,
        "pan_number": pan[0] if pan else None
    }