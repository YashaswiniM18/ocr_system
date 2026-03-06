import re
import math

def _recover_name_spaces(text):
    if not text or " " in text:
        return text
    # Heuristic: split names like "DMANIKANDAN" -> "D MANIKANDAN"
    # Only split if it starts with a consonant initial followed by another consonant,
    # and it's not a common name like "SHAIK" (S+H). 
    # But wait, SHAIK is actually a name. The problem is SHAIK MUZAMMIL is already spaced.
    # If the user says "S haik muzammil", it means the input was "SHAIK MUZAMMIL" 
    # and we split it because the first two were consonants.
    
    consonants = "BCDFGHJKLMNPQRSTVWXYZ"
    if len(text) > 4 and text[0] in consonants and text[1] in consonants:
        # Avoid splitting common name starts like SH, CH, TH, PH, GH
        if text[0:2] in ["SH", "CH", "TH", "PH", "GH"]:
            return text
        return text[0] + " " + text[1:]
    return text

def extract(texts, raw_data=None):
    # Normalize OCR lines
    lines = [t.strip() for t in texts if t.strip()]

    pan_number = None
    dob = None
    dob_index = -1
    dob_box = None

    # Step 1: Find PAN number and DOB
    for i, line in enumerate(lines):
        if not pan_number:
            pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", line)
            if pan_match:
                pan_number = pan_match.group()

        if not dob:
            dob_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", line)
            if dob_match:
                dob = dob_match.group()
                dob_index = i
                if raw_data and i < len(raw_data):
                    dob_box = raw_data[i].get("box")

    # Grouping logic: If raw_data exists, group blocks that are on the same horizontal line
    sorted_blocks = []
    if raw_data:
        # Sort by Y-coordinate of top-left corner
        for d in raw_data or []:
            box = d.get("box")
            if box:
                # Get average Y and min X
                avg_y = sum(p[1] for p in box) / 4
                avg_x = sum(p[0] for p in box) / 4
                min_x = min(p[0] for p in box)
                min_y = min(p[1] for p in box)
                # Rough width/height
                box_w = max(p[0] for p in box) - min(p[0] for p in box)
                box_h = max(p[1] for p in box) - min(p[1] for p in box)
                sorted_blocks.append({"text": d["text"], "y": avg_y, "x": avg_x, "box": box, "w": box_w, "h": box_h})

        # Grouping logic for names: PAN cards can be horizontal or vertical (rotated)
    if sorted_blocks:
        # Detect if text is mostly vertical (width < height)
        vertical_count = sum(1 for b in sorted_blocks if b["h"] > b["w"] * 1.5)
        is_rotated = vertical_count > len(sorted_blocks) * 0.3
        
        if is_rotated:
            # For rotated cards, sort by X. 
            # We need to know if X decreases from top to bottom or increases.
            # Heuristic: INCOME TAX DEPARTMENT is usually at the TOP.
            income_tax_x = None
            for b in sorted_blocks:
                if "INCOME" in b["text"].upper() or "ACCOUNT" in b["text"].upper():
                    income_tax_x = b["x"]
                    break
            
            # If Income Tax is at high X, then Top-to-Bottom is X Descending.
            # If Income Tax is at low X, then Top-to-Bottom is X Ascending.
            # Default to Descending as it's common for 90-deg CW rotation.
            reverse_x = True
            if income_tax_x is not None:
                # Compare to median X of all blocks
                all_x = sorted([b["x"] for b in sorted_blocks])
                median_x = all_x[len(all_x)//2]
                if income_tax_x < median_x:
                    reverse_x = False
            
            sorted_blocks.sort(key=lambda x: x["x"], reverse=reverse_x)
        else:
            sorted_blocks.sort(key=lambda x: x["y"])

    # Collect name candidates and label positions
    name_candidates = []
    labels = {} # type -> index in sorted_blocks
    
    prefixes = [
        r"^NAME[:\s]*", r"^FATHER['S]*\s*NAME[:\s]*", r"^S/O[:\s]*", r"^D/O[:\s]*", r"^W/O[:\s]*"
    ]
    label_patterns = {
        "name": [r"NAME\b", r"T\/NAME\b"],
        "father": [r"FATHER\b", r"FATHERS\s*NAME\b", r"faTa\b"]
    }

    candidate_list = [b["text"] for b in sorted_blocks] if sorted_blocks else lines

    for i, line in enumerate(candidate_list):
        clean_line = line.strip().upper()
        
        # Check for labels
        for l_type, p_list in label_patterns.items():
            for p in p_list:
                if re.search(p, clean_line, re.IGNORECASE):
                    labels[l_type] = i

        # Strip prefixes
        for p in prefixes:
            clean_line = re.sub(p, "", clean_line, flags=re.IGNORECASE).strip()
        
        # Candidate filtering
        if re.fullmatch(r"[A-Z0-9 \.]{2,}", clean_line):
            # Exclude headers and noise
            excluded = ["GOVERNMENT", "INCOME", "TAX", "DEPARTMENT", "SIGNATURE", 
                        "PERMANENT", "ACCOUNT", "NUMBER", "CARD", "INDIA", "HRROR", "ERROR", "HRD"]
            
            if any(k in clean_line for k in excluded):
                continue
            
            # Additional check for names: must contain at least one vowel (mostly)
            # or be a recognized initial pattern like "A K"
            if not re.search(r"[AEIOUY]", clean_line) and len(clean_line) > 3:
                continue

            # Skip small noise blocks that are likely misreads
            if len(clean_line) < 2:
                continue

            name_candidates.append({"text": clean_line, "index": i})

    # Step 3: Assign names
    name = None
    father_name = None

    # Strategy 1: Use label anchors if available
    name_val = None
    father_val = None

    if "name" in labels:
        # Name is usually the block *immediately before* or *immediately after* the label in sorted order
        # Depending on card layout. In the user's rotated case, Name is BEFORE the label in the sorted list (since X decreases).
        # Actually, let's look for nearest candidate.
        idx = labels["name"]
        nearest = sorted(name_candidates, key=lambda c: abs(c["index"] - idx))
        if nearest:
            name_val = nearest[0]["text"]
            # Remove from candidates so it doesn't get picked as father
            name_candidates = [c for c in name_candidates if c["text"] != name_val]

    if "father" in labels:
        idx = labels["father"]
        nearest = sorted(name_candidates, key=lambda c: abs(c["index"] - idx))
        if nearest:
            father_val = nearest[0]["text"]
            name_candidates = [c for c in name_candidates if c["text"] != father_val]

    if name_val and father_val:
        name = name_val.title()
        father_name = father_val.title()
    elif len(name_candidates) >= 2 or (name_val or father_val):
        # Fallback to order or partial anchors
        if not name_val:
            name_val = name_candidates[0]["text"] if name_candidates else None
            name_candidates = name_candidates[1:]
        if not father_val:
            father_val = name_candidates[0]["text"] if name_candidates else None
        
        name = name_val.title() if name_val else None
        father_name = father_val.title() if father_val else None
    
    # Final cleanup: recover spaces 
    if name: name = _recover_name_spaces(name.upper()).title()
    if father_name: father_name = _recover_name_spaces(father_name.upper()).title()

    return {
        "name": name,
        "father_name": father_name,
        "pan_number": pan_number,
        "date_of_birth": dob
    }

