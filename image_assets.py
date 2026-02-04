import cv2
import os
import numpy as np

# ---------------- FACE EXTRACTION ---------------- #

def extract_face(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # Take the largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # ✅ ADD PADDING
    pad = int(0.35 * w)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    face = img[y1:y2, x1:x2]

    cv2.imwrite(output_path, face)
    return output_path


# ---------------- SIGNATURE EXTRACTION ---------------- #

def extract_signature(image_path, output_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ✅ Expand ROI: lower half of the document
    roi_start = int(h * 0.5)
    roi = gray[roi_start:h, 0:w]

    # Improve contrast and remove noise
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter contours by aspect ratio (signatures are wide)
    signature_contours = []
    for c in contours:
        x, y, w1, h1 = cv2.boundingRect(c)
        if w1 / h1 > 3:  # width > 3x height
            signature_contours.append(c)

    if not signature_contours:
        return None

    # Pick the largest wide contour
    cnt = max(signature_contours, key=lambda c: cv2.boundingRect(c)[2])
    x, y, w1, h1 = cv2.boundingRect(cnt)

    # Adjust coordinates relative to original image
    y += roi_start

    # ✅ Add padding
    pad = 5
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w1 + pad, img.shape[1])
    y2 = min(y + h1 + pad, img.shape[0])

    signature = img[y1:y2, x1:x2]

    cv2.imwrite(output_path, signature)
    return output_path
