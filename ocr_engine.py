import cv2
import pytesseract
import os

from image_preprocessing import preprocess
from doc_classifier import classify
from vision_assets import extract_assets

from field_extractors import aadhaar, pan, driving_license, marksheet


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_text(text):
    return text.replace("O", "0").replace("I", "1")

def extract_fields(text, doc_type):
    if doc_type == "aadhaar":
        return aadhaar.extract(text)
    if doc_type == "pan":
        return pan.extract(text)
    if doc_type == "driving_license":
        return driving_license.extract(text)
    if doc_type == "marksheet":
        return marksheet.extract(text)
    return {}

def process_document(image_path):
    filename = os.path.basename(image_path)

    image = cv2.imread(image_path)
    original, processed = preprocess(image)

    raw_text = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
    raw_text = clean_text(raw_text)

    doc_type = classify(raw_text)
    fields = extract_fields(raw_text, doc_type)

    face_path, sign_path = extract_assets(original, filename)

    return {
        "document_type": doc_type,
        "raw_text": raw_text,
        "fields": fields,
        "face_image": face_path,
        "signature_image": sign_path
    }







