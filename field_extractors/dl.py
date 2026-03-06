import re
from image_assets import extract_dl_face, extract_dl_signature


# =====================================================
# DL NUMBER EXTRACTION
# =====================================================

def extract_dl_number(texts):

    full_text = " ".join(texts).upper()

    # Remove spaces & dashes
    clean = full_text.replace("-", "").replace(" ", "")

    # Indian DL pattern: State Code (2 chars) followed by 13-14 digits or alphanumeric formatting
    # More robust format e.g. MH0320080022135 or KA0120200016183
    pattern = r"[A-Z]{2}\d{13,14}|\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b"

    match = re.search(pattern, clean)
    if match:
        return match.group()
        
    # Second fallback on the uncleaned text to find exact matches
    match2 = re.search(r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b", full_text)
    if match2:
        return match2.group().replace("-", "").replace(" ", "")

    return None


# =====================================================
# DOB EXTRACTION
# =====================================================

def extract_dob(texts):

    full_text = " ".join(texts).upper()
    
    # 1. Targeted Regex for DOB label
    dob_match = re.search(r"D\.?O\.?B\.?\s*[:\-.]?\s*(\d{2}[/\-.]\d{2}[/\-.]\d{4})", full_text)
    if dob_match:
        return dob_match.group(1)

    # 2. General date pattern fallback
    pattern = r"\b\d{2}[\/\-.]\d{2}[\/\-.]\d{4}\b"
    matches = re.findall(pattern, full_text)

    if not matches:
        return None

    # DOB is usually the earliest year on the card (Issue Date and Expiry Date are later)
    dates = sorted(matches, key=lambda x: int(x[-4:]))
    return dates[0]


# =====================================================
# NAME EXTRACTION
# =====================================================

def extract_name(texts):

    full_text = " ".join(texts)
    
    # Generalized Stop Pattern for DL Artifacts
    # Added date pattern \d{2}/\d{2}/\d{4} to stop if DOB is on the same line without a label
    stop_pattern = r"(?:S/D|S/O|D/O|W/O|S/D/W|SON|DAUGHTER|WIFE|D\.O\.B|DOB|D\.0\.B|ADDRESS|ADD\s|ADD\.|PIN|SIGNATURE|B\.G\.|VALID|SEE RULE|RULE|\[S|COV|MCWG|LMV|DOI|\d{2}/\d{2}/\d{4})"
    
    # Try 1: Regex targeting NAME followed by anything up to a stopword
    regex = r"NAME\s*[:-]?\s*(.+?)(?=\s+" + stop_pattern + r"|$)"
    match = re.search(regex, full_text, flags=re.IGNORECASE)
    
    if match:
        name = match.group(1).lstrip(":-•= ").strip()
        # Post-process: Remove any trailing date if it somehow matched (e.g. if stop_pattern lookahead didn't catch it)
        name = re.split(r"[:\s-]*\d{2}/\d{2}/\d{4}", name)[0].strip()
        if len(name) >= 2:
            return name

    # Try 2: Line by line accumulation fallback if Regex fails
    for i, line in enumerate(texts):
        l = line.upper()
        if "NAME" in l:
            name_parts = []
            
            val_idx = l.find("NAME") + 4
            same_line_val = line[val_idx:].replace(":", "").strip()
            # Clean same line value from dates
            same_line_val = re.split(r"[:\s-]*\d{2}/\d{2}/\d{4}", same_line_val)[0].strip()

            if len(same_line_val) > 1:
                name_parts.append(same_line_val)
                
            j = i + 1
            while j < len(texts):
                part = texts[j].strip()
                upper_part = part.upper()
                
                if re.search(stop_pattern, upper_part, flags=re.IGNORECASE):
                    break
                    
                if len(part) > 1 or part.endswith('.'):
                    clean_part = part.lstrip(":-•= ")
                    # Clean part from dates
                    clean_part = re.split(r"[:\s-]*\d{2}/\d{2}/\d{4}", clean_part)[0].strip()
                    if clean_part:
                        name_parts.append(clean_part)
                j += 1
            if name_parts:
                return " ".join(name_parts)

    # fallback: longest meaningful line (but exclude dates)
    candidates = [t for t in texts if len(t.split()) >= 2 and not re.search(r"^\d{2}/\d{2}/\d{4}$", t.strip())]

    if candidates:
        best_candidate = max(candidates, key=len)
        # Even on longest line, strip trailing dates
        return re.split(r"[:\s-]*\d{2}/\d{2}/\d{4}", best_candidate)[0].strip()

    return None


# =====================================================
# MAIN FUNCTION REQUIRED BY FASTAPI
# =====================================================

def extract(texts, image):
    """
    IMPORTANT:
    Signature must match API:
    extract(texts, image)
    """

    try:

        if texts is None:
            texts = []

        dl_number = extract_dl_number(texts)
        dob = extract_dob(texts)
        name = extract_name(texts)


        return {
            "dl_number": dl_number,
            "name": name,
            "dob": dob
        }

    except Exception as e:
        print("DL Extraction Error:", e)

        return {
            "dl_number": None,
            "name": None,
            "dob": None
        }

