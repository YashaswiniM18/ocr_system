import cv2
import os

FACE_DIR = "outputs/faces"
SIGN_DIR = "outputs/signatures"
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(SIGN_DIR, exist_ok=True)

def extract_assets(image, filename):
    h, w, _ = image.shape

    face = image[int(h*0.12):int(h*0.42), int(w*0.05):int(w*0.35)]
    sign = image[int(h*0.72):int(h*0.92), int(w*0.35):int(w*0.85)]

    face_path = os.path.join(FACE_DIR, filename)
    sign_path = os.path.join(SIGN_DIR, filename)

    cv2.imwrite(face_path, face)
    cv2.imwrite(sign_path, sign)

    return face_path, sign_path
