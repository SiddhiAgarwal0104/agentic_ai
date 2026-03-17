from fastapi import FastAPI
from pipeline import process_document

app = FastAPI()

@app.get("/process_document")
def process(image_path: str):

    return process_document(image_path)