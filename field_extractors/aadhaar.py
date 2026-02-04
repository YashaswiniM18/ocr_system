import re

def extract_name(text_lines):
    for line in text_lines:
        clean = line.strip()

        # Skip headers & noise
        if len(clean) < 3:
            continue
        if any(k in clean.lower() for k in [
            "government", "india", "unique", "identification",
            "authority", "aadhaar", "dob", "year", "male", "female"
        ]):
            continue

        # Name: alphabets + spaces only
        if re.fullmatch(r"[A-Za-z ]+", clean):
            words = clean.split()
            if 1 < len(words) <= 4:
                return clean.title()

    return None

def extract_dob(text_lines):
    # 1️⃣ Look for DOB near Gender (most reliable)
    for i, line in enumerate(text_lines):
        l = line.lower()
        if "female" in l or "male" in l:
            # Check nearby lines (above & below)
            for j in range(max(0, i - 2), min(len(text_lines), i + 3)):
                match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", text_lines[j])
                if match:
                    return match.group()

    # 2️⃣ Skip issue/generated dates explicitly
    for line in text_lines:
        l = line.lower()
        if any(k in l for k in ["issue", "generated", "gov", "india"]):
            continue
        match = re.search(r"\d{2}[/-]\d{2}[/-]\d{4}", line)
        if match:
            return match.group()

    return None



def extract_gender(text_lines):
    for line in text_lines:
        l = line.lower()
        if "female" in l:
            return "Female"
        if "male" in l:
            return "Male"
    return None


def extract_aadhaar_number(text_lines):
    for line in text_lines:
        # Aadhaar: 12 digits (with OR without spaces)
        match = re.search(r"\b\d{12}\b", line.replace(" ", ""))
        if match:
            return match.group()
    return None


def extract_aadhaar_fields(text_lines):
    return {
        "name": extract_name(text_lines),
        "date_of_birth": extract_dob(text_lines),
        "gender": extract_gender(text_lines),
        "aadhaar_number": extract_aadhaar_number(text_lines),
    }

