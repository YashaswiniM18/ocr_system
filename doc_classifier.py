def classify(text):
    t = text.lower()

    if "aadhaar" in t or "uidai" in t:
        return "aadhaar"
    if "income tax" in t or "permanent account number" in t or "pan" in t:
        return "pan"
    if "driving licence" in t or "dl no" in t:
        return "driving_license"
    if "university" in t or "board of" in t or "marksheet" in t or "grade" in t:
        return "marksheet"

    return "unknown"


