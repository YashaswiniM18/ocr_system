from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os

from image_preprocessing import load_image
from ocr_engine import run_ocr
from doc_classifier import classify_document
from image_assets import extract_face, extract_signature
from field_extractors import aadhaar, pan, dl, marksheet

# ------------------ APP INIT ------------------

app = FastAPI(title="Real-Time OCR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ DIRECTORIES ------------------

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve output images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ------------------ UPLOAD API ------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load image
    image = load_image(file_path)

    # OCR
    texts = run_ocr(image)

    # DEBUG: print OCR text
    print("---- OCR TEXT ----")
    for t in texts:
        print(t)
    print("------------------")

    # Classify document
    doc_type = classify_document(texts)

    # Extract fields
    data = {}
    if doc_type == "Aadhaar":
        data = aadhaar.extract_aadhaar_fields(texts)
    elif doc_type == "PAN":
        data = pan.extract(texts)
    elif doc_type == "Driving License":
        data = dl.extract(texts)
    elif doc_type == "Marksheet":
        data = marksheet.extract(texts)

    # ------------------ FACE & SIGNATURE ------------------

    face_path = extract_face(
        file_path,
        os.path.join(OUTPUT_DIR, "face.jpg")
    )

    signature_path = extract_signature(
        file_path,
        os.path.join(OUTPUT_DIR, "signature.jpg")
    )

    # ------------------ RESPONSE ------------------

    return {
        "document_type": doc_type,
        "extracted_fields": data,
        "face_image": face_path,
        "signature_image": signature_path
    }

# ------------------ ROOT ------------------

@app.get("/")
def root():
    return {"message": "OCR API is running"}
