def classify_document(texts):
    text = " ".join(texts).upper()

    # Stronger Heuristics for Aadhaar
    if any(k in text for k in ["AADHAAR", "AADHAR", "UNIQUE IDENTIFICATION", "UIDAI"]):
        return "Aadhaar"
    
    # Heuristic: "Government of India" + ("DOB" or "Year of Birth") + "Male/Female"
    if "GOVERNMENT OF INDIA" in text:
        if any(k in text for k in ["DOB", "YEAR OF BIRTH", "YOB"]) and \
           any(k in text for k in ["MALE", "FEMALE"]):
            return "Aadhaar"

    # Heuristic: VID (Virtual ID) is specific to Aadhaar
    if "VID :" in text or "VID:" in text:
        return "Aadhaar"

    # Heuristic: 12-digit number pattern (Aadhaar number)
    # Check if there is a sequence of 12 digits (possibly with spaces)
    import re
    if re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text):
        return "Aadhaar"

    if "PERMANENT ACCOUNT NUMBER" in text or "INCOME TAX DEPARTMENT" in text:
        return "PAN"
    if "DRIVING LICENCE" in text or "DL NO" in text or "DRIVING LICENSE" in text:
        return "Driving License"
        
    # Heuristic: 14-15 digit alphanumeric DL number pattern (e.g., MH0320080022135, KA01 20200016183)
    if re.search(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", text):
        return "Driving License"
    if "UNIVERSITY" in text or "MARKS" in text or "RESULT" in text:
        return "Marksheet"

    return "Unknown"





