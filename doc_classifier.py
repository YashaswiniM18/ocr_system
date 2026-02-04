def classify_document(texts):
    text = " ".join(texts).upper()

    if "AADHAAR" in text:
        return "Aadhaar"
    if "PERMANENT ACCOUNT NUMBER" in text:
        return "PAN"
    if "DRIVING LICENCE" in text or "DL NO" in text:
        return "Driving License"
    if "UNIVERSITY" in text or "MARKS" in text:
        return "Marksheet"

    return "Unknown"





