from ocr_extractor import extract_text
from document_classifier import classify_document
from field_extractor import extract_fields

def process_document(image_path):

    text = extract_text(image_path)

    doc_type = classify_document(text)

    fields = extract_fields(text)

    return {
        "document_type": doc_type,
        "extracted_fields": fields
    }