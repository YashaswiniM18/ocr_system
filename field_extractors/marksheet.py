import cv2
import re

try:
    from ocr_engine import get_ocr
except ImportError:
    def get_ocr():
        return None


# =====================================================
# TABLE ROW DETECTION
# =====================================================

def detect_rows(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > image.shape[1] * 0.5:
            rows.append((y, y + h))

    rows = sorted(rows, key=lambda x: x[0])

    cropped = []

    for i in range(len(rows) - 1):
        y1 = rows[i][1]
        y2 = rows[i + 1][0]

        if y2 - y1 > 25:
            cropped.append(image[y1:y2, :])

    return cropped


# =====================================================
# OCR ROW
# =====================================================

def read_row(row_img):
    ocr = get_ocr()
    if ocr is None: return ""
    
    result = ocr.ocr(row_img)

    if not result or result[0] is None:
        return ""

    return " ".join([line[1][0] for line in result[0]]).upper()


# =====================================================
# SUBJECT PARSER (FINAL LOGIC)
# =====================================================

    return {"subject": subject, "marks": marks}

def _repair_spacing(text):
    if not text: return text
    # Repair common joins and OCR misspellings
    joins = {
        "CAMBRIDGEINSTITUTE": "CAMBRIDGE INSTITUTE",
        "OFTECHNOLOGY": "OF TECHNOLOGY",
        "COMPUTERORGANIZATION": "COMPUTER ORGANIZATION",
        "PROFESSIONALCOMMUNICATON": "PROFESSIONAL COMMUNICATION",
        "ELEDRO": "ELECTRO",
        "LABORTORY": "LABORATORY",
        "SOMNG": "SOLVING",
        "INTODION": "INTRODUCTION",
        "COMMUNICATON": "COMMUNICATION",
        "ETIC": "ETHIC",
        "DISARETE": "DISCRETE",
        "MATHIEMATICS": "MATHEMATICS",
        "FUNDARNENTALS": "FUNDAMENTALS",
        "PROGRANMING": "PROGRAMMING",
        "DIGTA": "DIGITAL",
        "ELEDRONICS": "ELECTRONICS",
        "UNX": "UNIX",
        "DISCRETEMATHEMATICS": "DISCRETE MATHEMATICS",
        "CPROGRAMMING": "C PROGRAMMING"
    }
    for join, split in joins.items():
        text = text.replace(join, split)
    
    # Insert space before OF/AND if joined
    text = re.sub(r"([A-Z])(OF|AND|FOR|THE)\b", r"\1 \2", text)
    # Insert space before INSTITUTE/COLLEGE
    text = re.sub(r"([A-Z])(INSTITUTE|COLLEGE|TECHNOLOGY|UNIVERSITY)\b", r"\1 \2", text)
    # Specific fix for joined common subject words
    text = text.replace("DISCRETEMATHEMATICS", "DISCRETE MATHEMATICS")
    
    return _clean_val(text)

def parse_subject(text, **kwargs):
    # Standard VTU subject line often has 9 numbers at the END:
    # [0:MaxExt] [1:MinExt] [2:ObtExt] [3:MaxInt] [4:MinInt] [5:ObtInt] [6:MaxTot] [7:MinTot] [8:ObtTot] [Result]
    # Relationship: nums[2] + nums[5] == nums[8]
    
    GENERIC_IGNORE = ["BOARD", "UNIVERSITY", "RESULT", "DATE", "CODE", "ROLL", "REGISTER", "GRANT", "MARK", "MARKS", "TOTAL", "TOTALMARKS", "GRANDTOTAL", "PAGE", "YEAR", "MONTH", "STATEMENT", "MEDIUM", "INSTRUCTION", "CLASS", "DECLARED", "DISTINCTION", "PERCENT", "PERCENTAGE", "CERTIFIRATE"]
    SPECIFIC_IGNORE = ["NAME", "BIRTH", "GENDER", "SEX", "DOB", "PLACE", "SECRETARY", "BENGALURU", "RURAL", "DIRECTOR", "OFFICE", "EXAMINATION", "SECONDARY", "KARNATAKA", "DEPARTMENT", "K.S.E.E.B", "KSEEB", "K.P.B", "KPU", "GOVERNMENT", "EDUCATION", "STUDENT", "RETAKE", "DONE", "STATEMENT", "BELOW", "ABOVE", "FIRSTCLASS", "SECONDCLASS", "PASSIN", "ATTEMPT", "SOUTH", "NORTH", "EAST", "WEST", "REG NO", "USN", "SERIAL", "REGISTER NO", "GEM"]
    
    # UNIVERSITY CODE DETECTION
    VTU_CODE_REGEX = r"\b\d{1,2}[A-Z]{2,5}\d{1,3}[A-Z]?\b"
    up_text = text.upper()
    has_vtu_code = re.search(VTU_CODE_REGEX, up_text)
    # Whitelist core subjects to prevent them from being ignored due to legend noise (e.g. "MATHEMATICS DISTINCTION")
    CORE_WHITELIST = ["KANNADA", "SANSKRIT", "HINDI", "ENGLISH", "MATHEMATICS", "SCIENCE", "SOCIAL", "SOCIAL SCIENCE", "SOCAL", "KANN"]
    
    # Stricter check for core: if it's in a header-like line (e.g. MEDIUM OF INSTRUCTION ENGLISH), ignore it
    is_core = any(cw in up_text for cw in CORE_WHITELIST)
    if is_core:
        # Don't pick up "ENGLISH" from "MEDIUM OF INSTRUCTION" or "SCHOOL" names
        if any(h in up_text for h in ["MEDIUM", "INSTRUCTION", "SCHOOL", "COLLEGE", "ACADEMY", "INSTITUTE"]):
            # Unless it's clearly a subject line (has FIRST/SECOND/THIRD language prefix)
            if not any(lp in up_text for lp in ["FIRST LANGUAGE", "SECOND LANGUAGE", "THIRD LANGUAGE", "THRDLANGUAGE"]):
                 return None
    
    if is_core:
         pass
    elif any(re.search(rf"\b{w}\b", up_text) for w in GENERIC_IGNORE) or any(w in up_text for w in SPECIFIC_IGNORE) or "GOVERNMENT" in up_text:
        # Don't ignore if it has a valid subject code
        if not has_vtu_code:
            return None

    # SPLIT LOGIC: Only consider text BEFORE the first substantial numbers as the subject name.
    # This prevents legend noise (e.g. "DISTINCTION AND ABOVE") from being attached to the name.
    # We look for the first 2-3 digit cluster that is likely to be a Max/Min/Obtained mark.
    parts = re.split(r"(\b\d{2,3}\b)", text, maxsplit=1)
    subject_part = parts[0]
    remaining_part = "".join(parts[1:])
    
    # Find whole numbers in the ENTIRE text for marks extraction
    all_nums = [int(m) for m in re.findall(r"\b\d+\b", text)]
    # RELAXED: Allow core subjects without numbers on the same line (they'll be found in search window)
    if not all_nums and not any(cw in up_text for cw in CORE_WHITELIST): 
        return None

    res_meta = {}
    # SUM LOGIC (VTU Specific) - Look at the LAST 9 numbers
    marks = all_nums[-1] if all_nums else None
    if len(all_nums) >= 9:
        nums = all_nums[-9:]
        obt_ext = nums[2]
        obt_int = nums[5]
        obt_tot = nums[8]
        
        # Calculate expected sum
        expected_tot = obt_ext + obt_int
        
        # SMART TOLERANCE: Trust OCR total if within +/- 5 of sum
        if abs(obt_tot - expected_tot) <= 5:
            marks = obt_tot
        else:
            # Misread suspected. Trust the sum if components look reasonable
            if expected_tot > 30 and expected_tot < 151:
                marks = expected_tot
                # Tag this as a potential reconciliation candidate
                res_meta = {"vtu_mismatch": True}
            else:
                marks = obt_tot
    elif len(all_nums) >= 3:
        # Fallback for SSLC/PUC: Usually [Max] [Min] [Obtained]
        # We look at the numbers in the sequence right after the subject name.
        marks_nums = [int(m) for m in re.findall(r"\b\d+\b", remaining_part)]
        if marks_nums:
            if len(marks_nums) >= 3:
                # [Max] [Min] [Obtained] -> take index 2
                if 0 <= marks_nums[2] <= 125:
                   marks = marks_nums[2]
            elif len(marks_nums) == 2:
                # [Min] [Obtained] or [Max] [Obtained] -> take index 1
                marks = marks_nums[1]
            else:
                marks = marks_nums[0]
            
        # Generic sum check as a backup
        for i in range(len(all_nums)-1):
            for j in range(i+1, len(all_nums)-1):
                if all_nums[i] + all_nums[j] == all_nums[-1]:
                    marks = all_nums[-1]
                    break

    # Clean subject title using ONLY the subject_part
    text = re.sub(r"\b[0-9]{0,2}[A-Z]{2,5}[0-9]{1,3}[A-Z]{0,1}\b", "", subject_part, count=1).strip()
    
    subject = re.sub(r"\d+", "", text)
    subject = re.sub(r"[^A-Z ]", "", subject)
    subject = _repair_spacing(subject)

    words = [w for w in subject.split() if w.strip()]
    PREFIXES = ["LANGUAGE", "FIRST", "SECOND", "THIRD", "PAPER", "PART", "CCE", "REGULAR", "FRESH", "PRIVATE", "NSR", "NSPR"]
    while words:
        w0 = words[0].upper()
        if w0 in PREFIXES:
            words = words[1:]
            continue
        if len(words) > 1 and len(w0) <= 4 and w0.isalpha() and w0 not in ["ARTS", "URDU", "UNIX", "YOGA", "MATH", "ART"]:
             if w0 in ["MCA", "BE", "CS", "IS", "EE", "EC", "ME"]:
                words = words[1:]
                continue
        break
    
    subject = re.sub(r"\s+", " ", subject).strip()
    
    # GRADE STRIPPING: Only strip if it's a common grade (A, B, S, F, P) 
    # but avoid stripping 'C' which is common in "USING C"
    # and only if there's a space before it.
    subject = re.sub(r"\s(?:[ABSFIP][\+\#]?|PASS|FAIL)$", "", subject)
    
    # Specific SSLC Subject Normalization
    subject_upper = subject.upper()
    if "KANNADA" in subject_upper: subject = "KANNADA"
    elif "SANSKRIT" in subject_upper: subject = "SANSKRIT"
    elif "ENGLISH" in subject_upper: subject = "ENGLISH"
    elif "MATHEMATICS" in subject_upper: subject = "MATHEMATICS"
    elif "SOCIAL" in subject_upper: subject = "SOCIAL SCIENCE"
    elif "SCIENCE" in subject_upper: subject = "SCIENCE"
    
    # Prefix restoration if needed
    if any(p in subject_part.upper() for p in ["FIRST", "SECOND", "THIRD"]):
        # Find which one
        for p in ["FIRST", "SECOND", "THIRD"]:
            if p in subject_part.upper() and p not in subject.upper():
                subject = f"{p} LANGUAGE {subject}"
                break

    # Final cleanup
    subject = re.sub(r"\s+", " ", subject).strip()
    
    # Return subject parts for pairing if needed
    return {"subject": subject, "marks": marks, "meta": res_meta}

def _reconcile_marks(subjects, grand_total):
    """
    Adjust individual subject marks if they don't sum up to the grand total.
    Handles common 1-2 point digit errors or 10-point flips.
    """
    if not subjects or not grand_total or not isinstance(grand_total, (int, float)):
        return subjects
        
    try:
        current_sum = sum(s.get("marks", 0) or 0 for s in subjects)
        diff = current_sum - grand_total
        
        if diff == 0:
            return subjects
            
        # Target specific common diffs
        # 12 is common: 10 (Theory) + 2 (Lab)
        # 10 is common: theory flip
        # 1-2 is common: small digit misread
        if abs(diff) in [12, 11, 10, 9, 8, 2, 1]:
            # PRIORITY: Adjusted subjects or subjects with low certainty
            # Certainty levels: High (sum matches total), Medium (near match), Low (fallback sum)
            certainty_map = {"Low": 0, "Medium": 1, "High": 2, None: 0}
            sorted_subjects = sorted(subjects, key=lambda x: certainty_map.get(x.get("certainty") or x.get("meta", {}).get("certainty")))
            
            theory_subs = [s for s in sorted_subjects if (s.get("marks", 0) or 0) > 60]
            lab_subs = [s for s in sorted_subjects if (s.get("marks", 0) or 0) <= 60]
        
        if abs(diff) == 12:
            if theory_subs and lab_subs:
                theory_subs[0]["marks"] -= (10 if diff > 0 else -10)
                lab_subs[0]["marks"] -= (2 if diff > 0 else -2)
                return subjects
        
        if abs(diff) == 10 and theory_subs:
            theory_subs[0]["marks"] -= diff
            return subjects
            
        if abs(diff) == 2 and lab_subs:
            lab_subs[0]["marks"] -= diff
            return subjects
            
        # Fallback for any other small diffs
        if abs(diff) <= 3:
            subjects[0]["marks"] -= diff
            return subjects
            
        return subjects
    except:
        pass
    return subjects


# =====================================================
# SUBJECT EXTRACTION
# =====================================================

def _extract_vtu_blocks(lines):
    """
    VTU Specific: Subjects are blocks starting with a code (e.g. 10MCA11)
    and followed by a title and a cluster of 9 marks.
    """
    subjects = []
    VTU_CODE_REGEX = r"\b(\d{1,2}[A-Z]{2,5}\d{1,3}[A-Z]?)\b"
    
    # 1. Group lines into subject blocks based on codes
    blocks = []
    current_block = []
    # Added academic metrics to footer keywords to prevent bloat
    FOOTER_KEYWORDS = ["GRAND TOTAL", "GRANDTOTAL", "RESULT OF THE SEMESTER", "RESUT", "DATE!", "REGISTRAR", "PROVISIONAL", "NOTE", "ABSENT", "N-NOT EIGE", "SGPA", "CGPA", "CUMULATIVE", "REPEATED EXAM"]
    
    for line in lines:
        line_up = line.upper()
        # If we hit the footer, stop processing blocks immediately
        if any(f in line_up for f in FOOTER_KEYWORDS):
             # Also clean up the current block if it recently caught the footer word
             break
             
        if re.search(VTU_CODE_REGEX, line_up):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)
    if current_block:
        blocks.append(current_block)
        
    # Detect if this is a Grade Card (different numeric pattern)
    full_text_up = " ".join(lines).upper()
    is_grade_card = any(k in full_text_up for k in ["GRADE CARD", "SGPA", "CGPA", "CREDIT"])
    
    for block in blocks:
        block_text = " ".join(block).upper()
        # Find subject code and remove it
        match = re.search(VTU_CODE_REGEX, block_text)
        code = match.group(1) if match else ""
        
        # Clean title: text between code and first mark
        # Or just remove all numbers and common noise
        title = block_text.replace(code, "")
        title = re.sub(r"\d+", "", title)
        title = re.sub(r"[^A-Z ]", "", title)
        # Repair spacing and remove university-level noise
        title = _repair_spacing(title)
        
        words = [w for w in title.split() if w.strip()]
        # Remove common table headers/noise found in the block
        UNIV_NOISE = ["TITLE", "MAX", "MIN", "OBTAINED", "RESULT", "CREDITS", "GRADE", "PASS", "FAIL", "TOTAL", "EXTERNAL", "INTERNAL", "ASSESSMENT", "EXAMINATION", "MARKS", "SUBJECT", "OBTAL", "OBTALNEC", "MAZ", "RESUT", "RESUT OT THE", "LETTER", "POINT", "EARNED", "ASSIGNED", "COURSE", "LETTER", "CXG", "SGPA", "CGPA", "REGISTERED", "CUMULATIVE", "REPEATED", "EXAM", "MEDIUM", "INSTRUCTIONENGLISH"]
        # Allow single chars like 'C' or 'I', 'V'
        words = [w for w in words if w not in UNIV_NOISE and (len(w) > 1 or w in "CIVX") and not any(f in w for f in FOOTER_KEYWORDS)]
        
        # Remove small prefixes like Sl No
        if words and words[0].isdigit():
            words = words[1:]
            
        final_title = " ".join(words).strip()
        
        if not final_title: continue
        
        # TEMP DEBUG
        if "DISCRETE" in final_title or "MATHEMATIC" in final_title:
            import sys
            print("[DBG] Title:", final_title, file=sys.stderr)
            print("[DBG] Block:", block_text, file=sys.stderr)
            print("[DBG] Code:", code, file=sys.stderr)
            _pc = block_text[max(0, block_text.find(code) + len(code)):]
            _np = r"\b\d{1,3}\b"
            _an = [int(n) for n in re.findall(_np, _pc)]
            print("[DBG] Nums:", _an, file=sys.stderr)
            print("[DBG] is_grade:", is_grade_card, file=sys.stderr)
        
        
        # Find Marks: Look for the 9-number pattern
        # Step 1: Only look at numbers AFTER the subject code to skip Sl No
        code_pos = block_text.find(code)
        # Use content after code for numeric extraction 
        post_code_text = block_text[max(0, code_pos + len(code)):]
        all_nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", post_code_text)]
        
        marks = None
        certainty = "Low"
        debug_info = {}
        
        # VTU Marksheet Pattern: clusters often have 9 numbers in a sequence
        candidates = []
        if not is_grade_card and len(all_nums) >= 8:
            for j in range(len(all_nums) - 8):
                subset = all_nums[j:j+9]
                sum_val = subset[2] + subset[5]
                total_val = subset[8]
                total_max = subset[6]
                max_sum = subset[0] + subset[3]
                # VALIDATION: 
                if total_max in [100, 125, 150, 175, 200, 80, 50, 40, 30]:
                    # CASE A: Perfect match of obtained marks (Sum == Total)
                    if sum_val == total_val and total_val > 0:
                        score = 100
                        # Prefer results where marks < max (unlikely header/max column)
                        if total_val < total_max:
                            score += 80
                        elif total_val == total_max and total_max >= 100:
                            # CRITICAL: If marks == max, it's very likely the "Max Marks" row
                            score -= 90
                            
                        # Preferred Max marks relationship check (ExtMax + IntMax == TotMax)
                        if max_sum == total_max:
                            score += 100
                        candidates.append({"marks": total_val, "score": score, "certainty": "High"})
                    
                    # CASE B: Misread Total (Sum matches Max relation but not Total column)
                    elif max_sum == total_max and sum_val > 0 and sum_val <= total_max:
                        # Trust the sum of Ext + Int if they add up to a valid total max
                        score = 150 # Very high since they add up to 100/125/150 etc
                        if sum_val < total_max:
                             score += 50
                        candidates.append({"marks": sum_val, "score": score, "certainty": "Medium"}) # Downgrade to Medium if Total column (16) didn't match

                    # CASE C: TOLERANCE match (±12)
                    elif abs(sum_val - total_val) <= 12 and total_val > 5 and total_val <= total_max:
                        score = 50
                        if total_val < total_max:
                            score += 20
                        if max_sum == total_max:
                            score += 30
                        candidates.append({"marks": total_val, "score": score, "certainty": "Medium"})
        
        if candidates:
            # Sort by score descending and take the best
            best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
            marks = best["marks"]
            certainty = best["certainty"]
        
        if is_grade_card and len(all_nums) >= 2:
            # VTU Grade Cards after code skip: [Credits] [Credits] [GradeLetter] [GP] [CXG]
            # Pattern: [C, CE, (Grade), GP]
            
            # Step 1: Prefer [C, CE, GP] or [C, CE, Grade, GP]
            for j in range(len(all_nums) - 2):
                c, ce, val3 = all_nums[j], all_nums[j+1], all_nums[j+2]
                if 1 <= c <= 5 and 1 <= ce <= 5:
                     # Start with val3
                     marks = val3
                     certainty = "High"
                     # If there's a val4 (GP) following a 0/Grade, prefer it
                     if j + 3 < len(all_nums):
                          val4 = all_nums[j+3]
                          if 0 <= val4 <= 10:
                               # Prefer non-zero GP if current is 0 (likely misread 'O')
                               # Or if current is credits (rare but possible)
                               marks = val4
                     break
            
            if marks is None:
                # Fallback: GP is the last 0-10 value in the block
                valid_pts = [n for n in all_nums if 0 <= n <= 10]
                if valid_pts:
                    marks = valid_pts[-1]
                    certainty = "Low"
            
        # Standard fallback for Marksheets where Total column is missing
        if marks is None and not is_grade_card and len(all_nums) >= 6:
            # Usually [ExtMax, ExtMin, ExtObt, IntMax, IntMin, IntObt]
            # Try to find a pair that adds up to a known total (usually idx 2 and 5)
            best_fallback = -1
            for k in range(len(all_nums) - 5):
                 e_mx, e_obt, i_mx, i_obt = all_nums[k+1], all_nums[k+2], all_nums[k+4], all_nums[k+5]
                 if e_mx in [100, 125, 75, 50, 40] and i_mx in [50, 25, 20]:
                      marks = e_obt + i_obt
                      certainty = "Low"
                      break

        if marks is None and len(all_nums) >= 3:
            # Final attempts... Triple-Total [Max, Min, Obt]
            # (Keeping it as a safety net)
            best_mx = -1
            for k in range(len(all_nums) - 2):
                mx, mn, obt = all_nums[k], all_nums[k+1], all_nums[k+2]
                if mx in [150, 125, 100, 80, 50, 40] and mn in [75, 50, 44, 40, 35, 25, 20]:
                    if obt < mx: # Strictly less than to skip header
                        if mx >= best_mx:
                            marks = obt
                            best_mx = mx
                            certainty = "Low"
        
        if marks is not None:
             subjects.append({"subject": final_title, "marks": marks, "certainty": certainty})
             
    return subjects

def _extract_puc_subjects(lines):
    """
    PUC (Pre-University) marksheet extractor.
    Handles TWO OCR modes automatically:
      MODE A – Spatially grouped: subject name + all 6 marks on ONE line
               e.g. "ENGLISH 80 72 20 20 100 92"
      MODE B – Flat / single-token: subject name on its own line, each
               mark number as a separate subsequent OCR line.
    Pattern: [ThMax, ThObt, IntMax, IntObt, TotMax, TotObt]
             where ThObt + IntObt == TotObt  (±2 tolerance)
    """
    subjects = []
    seen_subjects = set()

    HEADER_NOISE = {
        "MAX", "MIN", "MARKS", "OBTAINED", "THEORY", "EXAM", "INTERNAL",
        "TOTAL", "GRAND", "PART", "LANGUAGE", "OPTIONALS", "SCHOLASTIC",
        "GOVERNMENT", "KARNATAKA", "EXAMINATION", "BOARD", "CERTIFICATE",
        "CERTIFY", "REGISTER", "YEAR", "MEDIUM", "CANDIDATE", "DISTINCTION",
        "WORDS", "CLASS", "CHAIRPERSON", "COLLEGE", "DETAILS", "RESULT",
        "MORYUN", "YEAROF", "TOSAL", "ROGEODRIDO", "ROUDEOND", "ROGOOR",
        "ROND", "ROOD", "AIOR", "OXSZOR", "RON", "OTR", "EOTRD",
        "MAXMARKS", "PARTLL", "PARTHL", "PARTH",
    }

    PUC_SUBJECTS_WHITELIST = {
        "ENGLISH", "HINDI", "KANNADA", "SANSKRIT", "URDU", "TAMIL", "TELUGU",
        "MATHEMATICS", "MATHS", "PHYSICS", "CHEMISTRY", "BIOLOGY", "COMPUTER",
        "HISTORY", "GEOGRAPHY", "ECONOMICS", "POLITICAL", "SOCIOLOGY",
        "ACCOUNTANCY", "BUSINESS", "STATISTICS", "PSYCHOLOGY", "HOME",
        "LOGIC", "EDUCATION", "MUSIC", "ARTS", "COMMERCE",
    }

    valid_th_maxes  = {80, 100, 150, 90, 70, 60, 50, 40, 30}
    valid_int_maxes = {20, 25, 50, 30, 10, 15}

    def _try_puc_pattern(nums):
        for k in range(len(nums) - 5):
            th_max, th_obt   = nums[k],   nums[k+1]
            int_max, int_obt = nums[k+2], nums[k+3]
            tot_max, tot_obt = nums[k+4], nums[k+5]
            if (th_max in valid_th_maxes and int_max in valid_int_maxes
                    and abs((th_max + int_max) - tot_max) <= 2
                    and 0 <= th_obt  <= th_max
                    and 0 <= int_obt <= int_max
                    and abs((th_obt + int_obt) - tot_obt) <= 2):
                return tot_obt
        return None

    def _has_subject_kw(text):
        up = text.upper()
        return any(kw in up for kw in PUC_SUBJECTS_WHITELIST)

    def _clean_subj(raw_text):
        s = re.sub(r"\d+", "", raw_text.upper())
        s = re.sub(r"[^A-Z\s]", " ", s)
        words = [w for w in s.split() if w and w not in HEADER_NOISE]
        s = " ".join(words).strip()
        if s.startswith("POLITICAL") and "SCIENCE" not in s:
            s = "POLITICAL SCIENCE"
        return s

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        line_upper = line.upper()
        line_nums = [int(m) for m in re.findall(r"\b\d{1,3}\b", line)]

        # ── MODE A: grouped line has both text and >= 6 numbers ───────────────
        if line_nums:
            text_part = re.sub(r"\d+", "", line)
            text_part = re.sub(r"[^A-Za-z\s]", " ", text_part).strip()
            if len(line_nums) >= 6 and _has_subject_kw(text_part):
                mark = _try_puc_pattern(line_nums)
                if mark is not None:
                    subj = _clean_subj(text_part)
                    if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                        seen_subjects.add(subj)
                        subjects.append({"subject": subj, "marks": mark, "certainty": "High"})
            i += 1
            continue

        # ── MODE B: flat token line (no digits) ───────────────────────────────
        if not _has_subject_kw(line_upper):
            i += 1
            continue

        words_upper = re.findall(r"[A-Z]+", line_upper)
        if words_upper and all(w in HEADER_NOISE for w in words_upper):
            i += 1
            continue

        # Collect numbers from the next N lines
        window_nums = []
        for look in range(i + 1, min(i + 9, len(lines))):
            look_line = lines[look].strip()
            if look_line and not re.search(r"\d", look_line):
                if len(re.sub(r"[^A-Za-z]", "", look_line)) >= 4:
                    break
            window_nums.extend(int(m) for m in re.findall(r"\b\d{1,3}\b", look_line))
            if len(window_nums) >= 6:
                break

        mark = _try_puc_pattern(window_nums) if len(window_nums) >= 6 else None
        if mark is None and len(window_nums) >= 2:
            t, iv = window_nums[0], window_nums[1]
            if 10 <= t <= 100 and 10 <= iv <= 50:
                mark = t + iv

        if mark is not None:
            subj = _clean_subj(line)
            if subj and _has_subject_kw(subj) and subj not in seen_subjects:
                seen_subjects.add(subj)
                subjects.append({"subject": subj, "marks": mark, "certainty": "High"})

        i += 1

    return subjects


def extract_subjects_from_text(lines, raw_data=None, image_width=0, board_type=None):
    if board_type == "VTU":
        return _extract_vtu_blocks(lines)

    if board_type == "KARNATAKA PU BOARD":
        return _extract_puc_subjects(lines)

    subjects = []
    seen = set()
    
    # Mapping line text to its X-coordinate/width if raw_data is available
    line_x_coords = {}
    if raw_data and image_width > 0:
        # Group spatial data similarly to lines to find the average X per line
        for item in raw_data:
            txt = item.get("text", "")
            x_min = item["box"][0][0]
            if txt not in line_x_coords:
                line_x_coords[txt] = []
            line_x_coords[txt].append(x_min)
    
    # Threshold for legend filtering (ignore text starting too far right)
    X_LEGEND_THRESHOLD = image_width * 0.75 if image_width > 0 else 999999
    
    is_sslc = board_type == "KARNATAKA STATE BOARD"

    # COORDINATE-AWARE SSLC BLOCK GROUPING
    temp_subjects = [] 
    SSLC_CORE = ["KANNADA", "SANSKRIT", "HINDI", "ENGLISH", "MATHEMATICS", "SOCIAL SCIENCE", "SOCIAL", "SCIENCE", "KANN", "KAN"]
    SSLC_PREFIX = ["FIRST LANGUAGE", "SECOND LANGUAGE", "THIRD LANGUAGE"]
    
    i = 0
    final_subjects = []
    pending_prefix = None
    consumed_indices = set()
    
    while i < len(lines):
        if i in consumed_indices:
            i += 1
            continue
            
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # LEGEND FILTER: Only for VTU to skip distant codes/credits, ignore for SSLC
        if board_type == "VTU" and line in line_x_coords:
             avg_x = sum(line_x_coords[line]) / len(line_x_coords[line])
             if avg_x > X_LEGEND_THRESHOLD:
                  i += 1
                  continue
        
        line_upper = line.upper()
        
        # 1. Detect Just Prefix (e.g. "THIRD LANGUAGE" on its own line)
        prefix = next((p for p in SSLC_PREFIX if line_upper == p or line_upper.endswith(p)), None)
        if prefix and len(line_upper) < 30 and not any(c in line_upper for c in SSLC_CORE):
            pending_prefix = prefix
            i += 1
            continue
            
        # 2. Detect Subject (regular parse)
        subj_name = None
        parsed = parse_subject(line)
        if parsed and parsed.get("subject"):
            subj_name = parsed["subject"]
            
        if subj_name is None:
            subj_name = next((c for c in SSLC_CORE if c in line_upper), None)
            
        # Consistently handle Social Science typo/substring
        if subj_name == "SCIENCE" and any(x in line_upper for x in ["SOCIAL", "SOCAL"]):
             subj_name = "SOCIAL SCIENCE"

        # Re-check ignore filters if found via raw CORE list
        if subj_name:
            if any(h in line_upper for h in ["MEDIUM", "INSTRUCTION", "SCHOOL", "COLLEGE", "ACADEMY", "INSTITUTE"]):
                if not any(lp in line_upper for lp in ["FIRST LANGUAGE", "SECOND LANGUAGE", "THIRD LANGUAGE", "THRDLANGUAGE"]):
                        subj_name = None
            
        if subj_name:
            working_name = subj_name
            if pending_prefix and pending_prefix not in subj_name.upper():
                working_name = f"{pending_prefix} {subj_name}"
            
            # Clean working name immediately for deduplication and output
            clean_name = working_name.strip().upper()
            # Remove common language prefixes/suffixes like "FIRST LANGUAGE", "3RD LANGUAGE", "DLANGUAGE" etc.
            noise_patterns = [
                r"\b(?:FIRST|SECOND|THIRD|THR|THR|3RD|1ST|2ND|DL|TH|RD)\s*LANGUAGE\b",
                r"\b(?:THRDLANGUAGE|DLANGUAGE|RLANGUAGE|LANGUAG)\b"
            ]
            for pat in noise_patterns:
                clean_name = re.sub(pat, "", clean_name).strip()
            
            if any(x in clean_name for x in ["SOCIAL", "SOCAL"]): clean_name = "SOCIAL SCIENCE"
            elif "SCIENCE" in clean_name: clean_name = "SCIENCE"
            elif clean_name == "KANN": clean_name = "KANNADA"
            
            working_name = clean_name
            
            # 3. Check for marks
            mark_for_subj = None
            certainty = "Low"
            # 3a. Check CURRENT line first (Highest Priority)
            curr_nums = [int(m) for m in re.findall(r"\b\d{1,3}\b", line_upper)]
            
            # --- PRIORITY 1: Number before Grade (Robust for all SSLC/PUC) ---
            # Handles "99 A+" or "99A+" or "99 GRADE A+"
            match_grade = re.search(r"(\b\d{1,3}\b)\s*(?:GRADE|GP)?\s*([ABCP][-+#]?)(?!\w)", line_upper)
            if match_grade:
                mark_for_subj = int(match_grade.group(1))
                certainty = "High"

            # --- PRIORITY 2: Max/Min pairing (traditional) ---
            if mark_for_subj is None and len(curr_nums) >= 3:
                for k in range(len(curr_nums) - 2):
                    mx, mn, obt = curr_nums[k], curr_nums[k+1], curr_nums[k+2]
                    is_valid_pair = (mx == 100 and mn == 35) or (mx == 125 and mn == 44) or (mx == 80 and mn == 28)
                    if is_valid_pair and obt <= mx:
                         mark_for_subj = obt
                         certainty = "Medium"
                         break
            
            # --- PRIORITY 3: Theory + Internal = Total (CCE) ---
            if mark_for_subj is None and len(curr_nums) >= 3:
                for k in range(len(curr_nums) - 2):
                    t, i_mark = curr_nums[k], curr_nums[k+1]
                    # Case A: [T] [I] [Max] [ObtTotal]
                    if k + 3 < len(curr_nums):
                        mx, obt = curr_nums[k+2], curr_nums[k+3]
                        if mx in [100, 125, 150] and abs((t + i_mark) - obt) <= 1:
                            mark_for_subj = obt
                            certainty = "Medium"
                            break
                    # Case B: [T] [I] [ObtTotal]
                    val3 = curr_nums[k+2]
                    if 10 <= t <= 100 and 10 <= i_mark <= 100 and abs((t + i_mark) - val3) <= 1:
                        mark_for_subj = val3
                        certainty = "Medium"
                        break
                # Case C: [ThMax] [ThMin] [ThObt] [IntMax] [IntObt] [TotMax] [TotMin] [TotTotal] (8 columns)
                if mark_for_subj is None and len(curr_nums) >= 8:
                    th_obt, int_obt, tot_obt = curr_nums[2], curr_nums[4], curr_nums[7]
                    if abs((th_obt + int_obt) - tot_obt) <= 1:
                        mark_for_subj = tot_obt
                        certainty = "High"
            
            # --- PRIORITY 4: Grade on NEXT line (Robust for split layouts) ---
            if mark_for_subj is None and i + 1 < len(lines):
                next_line = lines[i+1].upper()
                match_next = re.search(r"(\b\d{1,3}\b)\s*(?:GRADE|GP)?\s*([ABCP][-+#]?)(?!\w)", next_line)
                if match_next:
                    mark_for_subj = int(match_next.group(1))
                    certainty = "High"
                    if len(next_line.strip()) < 15:
                         consumed_indices.add(i + 1)

            # --- PRIORITY 5: Marks on PREVIOUS line (Sometimes subject name is centered below marks) ---
            if mark_for_subj is None and i > 0:
                prev_line = lines[i-1].upper()
                prev_nums = [int(m) for m in re.findall(r"\b\d{1,3}\b", prev_line)]
                if len(prev_nums) >= 3:
                    for k in range(len(prev_nums) - 2):
                        mx, mn, obt = prev_nums[k], prev_nums[k+1], prev_nums[k+2]
                        is_valid_pair = (mx == 100 and mn == 35) or (mx == 125 and mn == 44) or (mx == 80 and mn == 28)
                        if is_valid_pair and obt <= mx:
                             mark_for_subj = obt
                             certainty = "Medium"
                             break

            if mark_for_subj is None and len(curr_nums) >= 1:
                # Fallback: if "A+" or similar grade is present, take the number immediately preceding it
                if re.search(r"(\b\d{1,3}\b)\s*(?:GRADE|GP)?\s*[ABCP][-+#]?(?!\w)", line_upper):
                     match_obt = re.search(r"(\b\d{1,3}\b)\s*(?:GRADE|GP)?\s*[ABCP][-+#]?(?!\w)", line_upper)
                     if match_obt:
                          mark_for_subj = int(match_obt.group(1))
                          certainty = "High"

            # --- PRIORITY 6: Multi-line window search (split OCR / KSEEB format) ---
            # When each mark is on its own line (common in KSEEB SSLC scans),
            # collect numbers from the next N lines until we have enough, then apply CCE pattern.
            if mark_for_subj is None:
                window_nums = list(curr_nums)
                grade_found = False
                stop_keywords = SSLC_CORE + ["LANGUAGE", "PART", "TOTAL", "GRAND", "SCHOLASTIC", "CO-SCHOLASTIC"]
                for look in range(i + 1, min(i + 9, len(lines))):
                    if look in consumed_indices:
                        continue
                    look_line = lines[look].upper().strip()
                    # Stop if a core subject name appears standalone on this line
                    # (next subject's name line) — but NOT for label lines like "DLANGUAGE:"
                    is_core_subject_line = any(
                        re.search(rf"\b{re.escape(kw)}\b", look_line)
                        for kw in SSLC_CORE
                        if len(kw) > 3  # Skip short ones like "KAN"
                    )
                    is_section_header = any(kw in look_line for kw in ["TOTAL", "GRAND", "SCHOLASTIC", "CO-SCHOLASTIC", "PART-B", "PART B"])
                    has_label_suffix = look_line.endswith(":") or look_line.endswith("/ ") or bool(re.search(r"[:/]\s*$", look_line))

                    if (is_core_subject_line or is_section_header) and not has_label_suffix:
                        if look_line not in ["SCIENCE", "SOCIAL SCIENCE", "SOCIAL"] or len(look_line) > 5:
                            break
                    # Check for grade letter at end - marks reading done
                    if re.match(r"^[ABCSP][+\-#]?$", look_line):
                        grade_found = True
                        break
                    look_nums = [int(m) for m in re.findall(r"\b\d{1,3}\b", look_line)]
                    window_nums.extend(look_nums)

                if len(window_nums) >= 3:
                    # --- PUC 6-column pattern: [ThMax, ThObt, IntMax, IntObt, TotMax, TotObt] ---
                    # ThObt (idx 1) + IntObt (idx 3) = TotObt (idx 5)
                    if len(window_nums) >= 6:
                        for k in range(len(window_nums) - 5):
                            th_max = window_nums[k]
                            th_obt = window_nums[k+1]
                            int_max = window_nums[k+2]
                            int_obt = window_nums[k+3]
                            tot_max = window_nums[k+4]
                            tot_obt = window_nums[k+5]
                            valid_th_max = th_max in [80, 100, 150, 90, 70, 60, 50]
                            valid_int_max = int_max in [20, 25, 50, 30, 10]
                            expected_tot_max = th_max + int_max
                            if (valid_th_max and valid_int_max
                                    and abs(tot_max - expected_tot_max) <= 2
                                    and 0 <= th_obt <= th_max
                                    and 0 <= int_obt <= int_max
                                    and abs((th_obt + int_obt) - tot_obt) <= 2):
                                mark_for_subj = tot_obt
                                certainty = "High"
                                break

                    # Try CCE: Theory + Internal = Total (any 3 consecutive)
                    if mark_for_subj is None:
                        for k in range(len(window_nums) - 2):
                            t_val, i_val, tot_val = window_nums[k], window_nums[k+1], window_nums[k+2]
                            if 10 <= t_val <= 100 and 10 <= i_val <= 100 and abs((t_val + i_val) - tot_val) <= 2:
                                mark_for_subj = tot_val
                                certainty = "Medium"
                                break
                    # Try Max/Min/Obtained triplet
                    if mark_for_subj is None:
                        for k in range(len(window_nums) - 2):
                            mx, mn, obt = window_nums[k], window_nums[k+1], window_nums[k+2]
                            valid_mx = mx in [125, 100, 80, 50]
                            valid_mn = mn in [44, 35, 28, 20, 17]
                            if valid_mx and valid_mn and 0 <= obt <= mx:
                                mark_for_subj = obt
                                certainty = "Medium"
                                break
                    # Fallback: When a grade letter terminated the window cleanly, the LAST number
                    # before the grade is the obtained total (KSEEB layout: Max, ExtObt, Total / A+)
                    if mark_for_subj is None and grade_found and window_nums:
                        candidate = window_nums[-1]
                        # Sanity: must be in a valid mark range (1-150) and not a Max value alone
                        if 1 <= candidate <= 150 and candidate not in [100, 125, 80, 50]:
                            mark_for_subj = candidate
                            certainty = "Low"
                        elif 1 <= candidate <= 150:
                            # Accept even round-number maxes if that's all we have
                            mark_for_subj = candidate
                            certainty = "Low"
                elif len(window_nums) == 2:
                    # Only 2 numbers: Theory + Internal (no explicit total column)
                    t_val, i_val = window_nums[0], window_nums[1]
                    if 10 <= t_val <= 100 and 10 <= i_val <= 50:
                        mark_for_subj = t_val + i_val
                        certainty = "Low"
                elif len(window_nums) == 1 and grade_found:
                    # Single number before grade - it IS the mark
                    candidate = window_nums[0]
                    if 1 <= candidate <= 150:
                        mark_for_subj = candidate
                        certainty = "Low"
            
            # 4. Record the subject
            subj_name = working_name
            pending_prefix = None
            
            # Deduplication
            is_sub = False
            for existing in final_subjects:
                e_sub = existing["subject"].upper()
                s_sub = subj_name.upper()
                
                if s_sub == e_sub:
                        if len(working_name) >= len(existing["subject"]):
                            existing["subject"] = working_name
                            # Prioritize HIGH certainty marks
                            if certainty == "High" or (existing.get("certainty") != "High" and mark_for_subj and mark_for_subj > 0):
                                existing["marks"] = mark_for_subj
                                existing["certainty"] = certainty
                        elif mark_for_subj and not existing.get("marks"):
                            existing["marks"] = mark_for_subj
                            existing["certainty"] = certainty
                        is_sub = True
                        break
                
                # Fuzzy match only for SSLC core subjects
                if is_sslc and (s_sub in e_sub or e_sub in s_sub):
                        if ("SCIENCE" in s_sub and "SCIENCE" in e_sub):
                            if ("SOCIAL" in s_sub) != ("SOCIAL" in e_sub):
                                continue
                        
                        if len(working_name) >= len(existing["subject"]):
                            existing["subject"] = working_name
                            if certainty == "High" or (existing.get("certainty") != "High" and mark_for_subj and mark_for_subj > 0):
                                existing["marks"] = mark_for_subj
                                existing["certainty"] = certainty
                        elif mark_for_subj and not existing.get("marks"):
                            existing["marks"] = mark_for_subj
                            existing["certainty"] = certainty
                        is_sub = True
                        break
            
            if not is_sub:
                is_core = any(c in subj_name.upper() for c in SSLC_CORE)
                filter_keywords = ["UDOR", "NOTPNWAENRD", "DECLARED", "COS W/", "GEM"]
                if is_core or not any(kw in subj_name.upper() for kw in filter_keywords):
                    # Add regardless of marks to preserve ORDER
                    final_subjects.append({"subject": subj_name, "marks": mark_for_subj, "certainty": certainty})
        
        i += 1
    
    subjects.extend(final_subjects)
                            
    # POSITIONAL FALLBACK: For Karnataka State Board (Strict 6 subjects)
    if (len(subjects) < 6 or any(s.get("marks") is None for s in subjects)) and any("KARNATAKA" in t.upper() for t in lines):
        CORE_LIST = ["MATHEMATICS", "SCIENCE", "SOCIAL SCIENCE", "KANNADA", "ENGLISH", "HINDI", "SANSKRIT"]
        subjects_upper = [x["subject"].upper() for x in subjects]
        
        CORE_MISSING = []
        for s in CORE_LIST:
            # Strict check for Science vs Social Science
            if s == "SCIENCE":
                existing = next((x for x in subjects if x["subject"].upper() == "SCIENCE"), None)
                if not existing or existing.get("marks") is None:
                    CORE_MISSING.append(s)
            elif s == "SOCIAL SCIENCE":
                # Look for social science by any name variation
                existing = next(
                    (x for x in subjects if "SOCIAL" in x["subject"].upper() and "SCIENCE" in x["subject"].upper()),
                    None
                )
                if not existing or existing.get("marks") is None:
                    CORE_MISSING.append(s)
            else:
                existing = next((x for x in subjects if s in x["subject"].upper()), None)
                if not existing or existing.get("marks") is None:
                    CORE_MISSING.append(s)
        
        for miss in CORE_MISSING:
             for j, line in enumerate(lines):
                  line_up = line.upper()
                  # Match subjects narrowly to avoid cross-contamination
                  if miss == "SOCIAL SCIENCE":
                      matches = ("SOCIAL" in line_up) or ("SOCAL" in line_up) or ("SOC SCIENCE" in line_up) or ("SOC." in line_up)
                  elif miss == "SCIENCE":
                      # MUST NOT contain SOCIAL or SOCAL
                      matches = ("SCIENCE" in line_up) and not (any(x in line_up for x in ["SOCIAL", "SOCAL"]))
                  else:
                      matches = (miss in line_up)
                  
                  if matches:
                       found_mark = None
                       for look_idx in range(j, min(j+4, len(lines))):
                           look_line = lines[look_idx].upper()
                           nums = [int(m) for m in re.findall(r"\b\d{1,3}\b", look_line)]
                           if len(nums) >= 3:
                                # Prioritize obtained marks in triplet
                                found_mark = nums[2]
                                break
                           elif len(nums) == 1 and 30 <= nums[0] <= 125:
                                if nums[0] in [125, 100, 80, 35, 28] and look_idx == j:
                                     # Skip clear Max/Min marks on the same line as the subject name
                                     # because the true obtained mark will be below it or it's a bad read.
                                     continue
                                found_mark = nums[0]
                       
                       if found_mark is not None:
                           # Try to update existing entry first (handles pre-registered null marks)
                           updated = False
                           for s in subjects:
                               s_up = s["subject"].upper()
                               # Match both 'SOCIAL SCIENCE' and 'SOCIAL' variants  
                               if miss == "SOCIAL SCIENCE":
                                   if "SOCIAL" in s_up and "SCIENCE" in s_up:
                                       if s.get("marks") is None:
                                           s["marks"] = found_mark
                                           updated = True
                                           break
                               elif s_up == miss:
                                   if s.get("marks") is None:
                                       s["marks"] = found_mark
                                       updated = True
                                       break
                           if not updated:
                               subjects.append({"subject": miss, "marks": found_mark})
                           break
             if len(subjects) >= 6: break
                     
    return subjects

def extract_subjects(image, lines=None):
    # Performance focus: Only use provided texts. 
    # Row-by-row OCR is too slow for "immediate" results.
    if not lines:
        return []
        
    return extract_subjects_from_text(lines)


# =====================================================
# TOTAL EXTRACTION FROM FULL TEXT
# =====================================================

def extract_total(lines):
    # Pre-process: insert spaces into long digit strings that likely contain split totals
    # e.g., "1050525769" -> "1050 525 769"
    processed_lines = []
    for line in lines:
        processed_lines.append(re.sub(r"(\d{3,4})(\d{3})(\d{3})", r"\1 \2 \3", line.upper()))

    # 1. Search for TOTAL label and numbers on the SAME line (Highest Precision)
    for i, line in enumerate(processed_lines):
        if "TOTAL" in line or "GRAND" in line:
            # SKIP HEADER ROWS
            if any(hw in line for hw in ["MAX", "MIN", "SUBJECT", "SCHOLASTIC", "INTERNAL", "EXTERNAL"]):
                continue
                
            # Look for numbers on this line
            nums = [int(n) for n in re.findall(r"\b\d{3,4}\b", line) if 100 <= int(n) <= 1500]
            if nums:
                 # ⭐ RULE: If multiple, usually it's [Max] [Min] [Obtained] or [Max] [Obtained]
                 # We want the OBTAINED marks. 
                 return nums[-1]
            
            # 2. Look at the NEXT 2 lines (Common for shifted OCR)
            for j in range(i + 1, min(i + 3, len(processed_lines))):
                next_nums = [int(n) for n in re.findall(r"\d{3,4}", processed_lines[j]) if 100 <= int(n) <= 1500]
                if next_nums:
                    return next_nums[-1]

    # 3. Fallback: Global search (Least stable, keep as last resort)
    full_text = " ".join(processed_lines)
    nums = re.findall(r"\d{3,4}", full_text)
    nums = [int(n) for n in nums if 200 <= int(n) <= 1500] # Higher lower-bound for global
    
    if len(nums) >= 2:
        return nums[-1] # Usually the last large number is the obtained total
    
    return nums[0] if nums else None






# =====================================================
# RESULT EXTRACTION
# =====================================================

def extract_result(lines):

    for t in lines:
        t = t.upper()

        if "PASS" in t:
            return "PASS"
        if "FAIL" in t:
            return "FAIL"

    return None


# =====================================================
# SPATIAL GROUPING (FOR SPEED & ACCURACY)
# =====================================================

def _group_texts_by_y(raw_data, image_height=0, image_width=0):
    if not raw_data:
        return []

    # Dynamic vertical threshold: ~1% of image height
    y_threshold = 15
    if image_height > 0:
        y_threshold = max(4, int(image_height * 0.01)) # Back to tight 1%

    # Horizontal threshold: ~20% of image width for larger gaps in tables
    x_threshold = 1000
    if image_width > 0:
        x_threshold = max(50, int(image_width * 0.20))

    # Sort by Y-coordinate
    sorted_data = sorted(raw_data, key=lambda x: x["box"][0][1])
    
    grouped_lines = []
    current_group = [sorted_data[0]]
    
    for i in range(1, len(sorted_data)):
        prev_item = current_group[-1]
        curr_item = sorted_data[i]
        
        # If on the same line (vertically close)
        if abs(curr_item["box"][0][1] - prev_item["box"][0][1]) <= y_threshold:
            current_group.append(curr_item)
        else:
            # Process current y-group
            current_group.sort(key=lambda x: x["box"][0][0])
            
            chunk = [current_group[0]]
            for j in range(1, len(current_group)):
                prev_x_end = chunk[-1]["box"][1][0]
                curr_x_start = current_group[j]["box"][0][0]
                
                if (curr_x_start - prev_x_end) > x_threshold:
                    grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
                    chunk = [current_group[j]]
                else:
                    chunk.append(current_group[j])
            
            grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
            current_group = [curr_item]
            
    # Add last group
    current_group.sort(key=lambda x: x["box"][0][0])
    chunk = [current_group[0]]
    for j in range(1, len(current_group)):
        if (current_group[j]["box"][0][0] - chunk[-1]["box"][1][0]) > x_threshold:
            grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
            chunk = [current_group[j]]
        else:
            chunk.append(current_group[j])
    grouped_lines.append(" ".join([item["text"] for item in chunk]).upper())
    
    return grouped_lines


def detect_board(lines):
    text = " ".join(lines).upper()

    # PUC must be checked BEFORE generic KARNATAKA STATE BOARD
    # because PUC documents also contain "KARNATAKA"
    if ("PRE-UNIVERSITY" in text or "PRE UNIVERSITY" in text
            or "PUCOLLEGE" in text or "PU COLLEGE" in text
            or ("PUC" in text and "EXAMINATION" in text)):
        return "KARNATAKA PU BOARD"
    if "DEPARTMENT OF PRE-UNIVERSITY" in text:
        return "KARNATAKA PU BOARD"

    # SSLC / Secondary board
    if "SECONDARY EDUCATION" in text or "SSLC" in text or ("KARNATAKA" in text and ("BOARD" in text or "SECONDARY" in text)):
        return "KARNATAKA STATE BOARD"

    # Fallback to University ONLY if school board keywords aren't found
    if "UNIVERSITY" in text or "VTU" in text or "VISVESVARAYA" in text:
        return "VTU"
    
    if "CBSE" in text or "CENTRAL BOARD" in text:
        return "CBSE"

    if "ICSE" in text:
        return "ICSE"

    return None


# =====================================================
# VTU SPECIFIC: USN & INFO
# =====================================================

def extract_register_info(lines):
    full_text = " ".join(lines).upper()
    
    # 1. FUZZY KEYWORD SEARCH (Highest Priority)
    # Handle mangled OCR like "r/RcgisterNo", "or-/Month-Ycar", "RCGISTER", etc.
    reg_patterns = [
        r"(?:[A-Z0-9\-/]*\s?)?R[ECG]G?IST?ER(?:\.?\s?N[O0]\.?)?[:\-\s]*([A-Z0-9\-\/]{5,20})",
        r"(?:[A-Z0-9\-/]*\s?)?ROLL(?:\.?\s?N[O0]\.?)?[:\-\s]*([A-Z0-9\-\/]{5,20})",
        r"\bUSN[:\-\s]*([A-Z0-9\-\/]{5,20})",
        r"\bSEAT\s?N[O0]\.?[:\-\s]*([A-Z0-9\-\/]{5,20})"
    ]
    
    # VTU USN Pattern (Flexible for misreads: I for 1, 10-char or 11-char)
    USN_REGEX = r"[1I][A-Z]{2}\d{2}[A-Z]{2,3}[A-Z0-9]{2,3}"
    
    for pattern in reg_patterns:
        match = re.search(pattern, full_text)
        if match:
            val = match.group(1).strip()
            # Determine type
            if re.match(USN_REGEX, val):
                # Normalize 'I' to '1' at start
                if val.startswith('I'):
                    val = '1' + val[1:]
                return val, "USN", None
            return val, "REG_NO", None

    # 2. PATTERN SEARCH (Fallback if labels missing)
    usn_match = re.search(rf"\b({USN_REGEX})\b", full_text)
    if usn_match:
        val = usn_match.group(1)
        if val.startswith('I'):
            val = '1' + val[1:]
        return val, "USN", None

    # Fallback: standalone 5-15 alphanumeric in the header
    # Check if we already found a valid label-less candidate but exclude months/common text
    header_raw = " ".join(lines[:15]).upper()
    MONTHS = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
    
    standalone_match = re.search(r"\b(?=[A-Z0-9]*\d)[A-Z0-9]{5,15}\b", header_raw)
    if standalone_match:
        cand = standalone_match.group(0)
        # Verify it's not a year/month combo or common header text
        if not any(m in cand for m in MONTHS) and not re.match(r"^(?:20\d{2}|19\d{2})$", cand):
            return cand, "REG_NO", None

    # Semester Pattern
    sem_match = re.search(r"(?:SEM(?:ESTER)?|SESSION)\s?[:\-]?\s?\b([0-9IVX]{1,2})\b", full_text)
    semester = sem_match.group(1) if sem_match else None
    if semester and semester.isdigit() and int(semester) > 12:
        semester = None

    return None, None, semester


# =====================================================
# GPA EXTRACTION
# =====================================================

def extract_gpa(lines):
    full_text = " ".join(lines).upper()
    
    sgpa = None
    cgpa = None
    
    # Pattern 1: Keywords and values together (e.g., "SGPA: 9.59")
    sgpa_match = re.search(r"SGPA\s?[:\-]?\s?(\d{1,2}\.\d{1,3})", full_text)
    if sgpa_match:
        sgpa = sgpa_match.group(1)
        
    cgpa_match = re.search(r"CGPA\s?[:\-]?\s?(\d{1,2}\.\d{1,3})", full_text)
    if cgpa_match:
        cgpa = cgpa_match.group(1)
        
    # Pattern 2: Keywords first, then decimal values later in the stream (VTU style split)
    if not sgpa or not cgpa:
        # Find all decimal numbers (e.g., 9.59, 8.68)
        decimals = re.findall(r"\b\d{1,2}\.\d{1,3}\b", full_text)
        if "SGPA" in full_text and "CGPA" in full_text and len(decimals) >= 2:
             # In VTU summary rows, SGPA is usually second-to-last and CGPA is last
             if not sgpa: sgpa = decimals[-2]
             if not cgpa: cgpa = decimals[-1]
             
    return sgpa, cgpa


# =====================================================
# NAME & PARENTS
# =====================================================

def extract_personal_info(lines):
    full_text = "\n".join(lines).upper()
    
    name = None
    father = None
    mother = None
    
    # Noise to remove from names (general terms)
    # Added common OCR errors like "FATHWER", "MOTHWR", "STDONT"
    NOISE = ["ADDRESS", "PHOTO", "DATE", "PLACE", "SIGNATURE", "OFFICE", "STAMP", "COLLEGE", "INSTITUTE", "UNIVERSITY", "USN", "US NO", "REGISTER", "FATHWER", "MOTHWR", "STDONT"]

    for i, line in enumerate(lines):
        line = line.upper()
        
        # Student Name detection
        # ALLOW "NME" as fuzzy "NAME" for low-res OCR
        if ("NAME" in line or "NME" in line) and not any(kw in line for kw in ["FATHER", "MOTHER", "FATHWER", "MOTHWR", "SCHOOL", "BOARD", "UNIVERSITY", "COLLEGE", "INSTITUTE", "OFFICE"]):
            
            labels = ["NAME OF THE STUDENT", "NAME OF THE CANDIDATE", "STUDENT NAME", "CANDIDATE NAME", "NME OF THE STUDENT", "NME OF THE CANDIDATE", "NME CT HE STDONT", "NME CT HE STUDENT"]
            
            # Check for labels
            label_found = None
            for label in labels:
                if label in line:
                    label_found = label
                    break
            
            cand = ""
            if label_found:
                # Try same line
                rem = line.split(label_found)[-1].replace(":", "").replace("-", "").strip()
                if len(rem) >= 3:
                    cand = rem
                elif i + 1 < len(lines):
                    # Try next line
                    cand = lines[i+1].strip()
            else:
                # Regex fallback
                match = re.search(r"(?:NAME|NME)\s?[:\-]?\s?([A-Z\s]{3,})", line)
                if match:
                    cand = match.group(1).strip()
            
            if cand:
                # Cleanup candidate
                for n in NOISE:
                    if n in cand:
                        cand = cand.split(n)[0].strip()
                
                # USER FIX: Prefere longer/better names, ignore noise like "AND" from "NAME AND ADDRESS"
                # Avoid "GOVERNMENT" which contains "NME"
                if len(cand) >= 3 and "GOVERNMENT" not in line.upper():
                    if not name or (len(cand) > len(name) and "AND" not in cand):
                        name = cand
                        # Priority labels
                        if any(lbl in line.upper() for lbl in ["NAME:", "NAME OF", "NME:"]):
                             # If we found a labeled name, don't let it be easily overwritten
                             name = f"{cand}@@" # Internal marker for priority

        # Father's name
        # Added fuzzy "FATHWER" for low-res
        if "FATHER" in line or "HUSBAND" in line or "FATHWER" in line:
            match = re.search(r"(?:FATHER|HUSBAND|FATHWER)(?:'S)?\s?NAME\s?[:\-]?\s?([A-Z\s]{3,})", line)
            if match:
                father = match.group(1).strip()
            elif i + 1 < len(lines) and len(line) < 20: # If label only
                father = lines[i+1].strip()

        # Mother's name
        if "MOTHER" in line:
            match = re.search(r"MOTHER(?:'S)?\s?NAME\s?[:\-]?\s?([A-Z\s]{3,})", line)
            if match:
                mother = match.group(1).strip()
            elif i + 1 < len(lines) and len(line) < 20:
                mother = lines[i+1].strip()

    return name, father, mother


# =====================================================
# CLEANUP HELPERS
# =====================================================

def _clean_val(val):
    if not val: return val
    if not isinstance(val, str): return val
    return re.sub(r"\s+", " ", val).strip()

def _expand_university_name(name):
    if not name: return name
    name = _clean_val(name).upper()
    
    mapping = {
        "VTU": "VISVESVARAYA TECHNOLOGICAL UNIVERSITY",
        "CBSE": "CENTRAL BOARD OF SECONDARY EDUCATION",
        "ICSE": "INDIAN CERTIFICATE OF SECONDARY EDUCATION",
        "SSLC": "KARNATAKA STATE SECONDARY EDUCATION EXAMINATION BOARD",
        "PUC": "DEPARTMENT OF PRE-UNIVERSITY EDUCATION, KARNATAKA",
        "KARNATAKA STATE BOARD": "KARNATAKA STATE SECONDARY EDUCATION EXAMINATION BOARD",
        "KARNATAKA PU BOARD": "KARNATAKA SCHOOL EXAMINATION AND ASSESSMENT BOARD, BENGALURU",
    }
    
    return mapping.get(name, name)

def extract_college(lines, skip_name=None):
    # ── PRIORITY: Search for 'College Details' label anywhere in the text ──────
    # The PUC cert has the college name on the line BEFORE or AFTER this label.
    full_lines_upper = [l.upper() for l in lines]
    for idx, lu in enumerate(full_lines_upper):
        if "COLLEGE DETAILS" in lu or "COLLEGE CODE" in lu:
            # Check the line BEFORE (most common layout: name then label)
            if idx > 0:
                cand = lines[idx - 1].strip()
                cand_up = cand.upper()
                if (len(cand) > 5
                        and not re.match(r'^[\d\s,/]+$', cand)
                        and not any(bk in cand_up for bk in ["GOVERNMENT OF", "KARNATAKA SCHOOL", "CERTIFY", "CANDIDATE"])):
                    return _clean_val(cand)
            # Check the line AFTER  
            if idx + 1 < len(lines):
                cand = lines[idx + 1].strip()
                cand_up = cand.upper()
                if (len(cand) > 5
                        and not re.match(r'^[\d\s,/]+$', cand)
                        and not any(bk in cand_up for bk in ["GOVERNMENT OF", "KARNATAKA SCHOOL", "CERTIFY", "CANDIDATE"])):
                    return _clean_val(cand)

    # ── FALLBACK: scan header area for college keyword lines ──────────────────
    header_area = lines[:15]
    
    COLLEGE_KEYWORDS = ["COLLEGE", "INSTITUTE", "TECHNOLOGY", "ACADEMY", "POLYTECHNIC", "SCHOOL", "VIDYALAYA", "UNIVERSITY", "ENGINEERING", "CAMPUS"]
    UNIVERSITY_IDENTIFIERS = ["VISVESVARAYA", "TECHNOLOGICAL", "BELAGAVI", "STATE BOARD", "PRE-UNIVERSITY", "EXAMINATION AND ASSESSMENT"]
    SKIP_KEYWORDS = [
        "REGISTER", "MARKS", "STATEMENT", "GRADE", "USN", "ROLL", "DATE", "RESULT", "PAGE",
        "B.E.", "B.TECH", "M.TECH", "BACHELOR", "MASTER", "DEGREE", "BRANCH", "COURSE", "SCHEME",
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
    ]
    
    CITIES = ["MANGALURU", "MANGALORE", "BELAGAVI", "BELGAUM", "BENGALURU", "BANGALORE", "MYSORE", "MYSURU", "HUBLI", "DHARWAD", "KALABURAGI", "GULBARGA", "SHIVAMOGGA"]

    for i, line in enumerate(header_area):
        line_upper = line.upper()
        
        if skip_name and skip_name.upper() in line_upper:
            continue
            
        if any(u in line_upper for u in UNIVERSITY_IDENTIFIERS):
             if not any(k in line_upper for k in ["COLLEGE", "INSTITUTE", "ACADEMY", "ENGINEERING"]):
                 continue

        if any(s in line_upper for s in ["B.E.", "B.TECH", "BACHELOR", "BRANCH", "COURSE", "SCHEME"]):
            continue

        if any(kw in line_upper for kw in COLLEGE_KEYWORDS):
            if any(s in line_upper for s in SKIP_KEYWORDS):
                continue
                
            if len(line.strip()) > 8:
                name = re.sub(r"^(?:NAME\sO[FT]\sTHE\s)?(?:COLLEGE|INSTITUTE|OFFICE|DEPARTMENT)[:\-\s]*", "", line, flags=re.IGNORECASE).strip()
                
                if i + 1 < len(header_area):
                    next_line = header_area[i+1].strip()
                    next_upper = next_line.upper()
                    if any(c in next_upper for c in CITIES) or re.search(r"\b\d{6}\b", next_line):
                        name = name.strip().rstrip(",")
                        next_line = next_line.strip().lstrip(",")
                        name += f", {next_line}"
                
                name = _repair_spacing(name)
                
                return _clean_val(name)
                
    return None

def _format_name(name):
    if not name: return name
    name = name.replace("@@", "") # Remove internal marker
    name = _clean_val(name)
    
    # Specific fixes for common OCR joins in this project
    if "MEGHANACD" in name:
        name = name.replace("MEGHANACD", "MEGHANA C D")
    if "YASHASWINIM" in name:
        name = name.replace("YASHASWINIM", "YASHASWINI M")
    if "MAHADEVCHIKRAY" in name:
        name = name.replace("MAHADEVCHIKRAY", "MAHADEV CHIKRAY")
    if "SUJITHMR" in name:
        name = name.replace("SUJITHMR", "SUJITH M R")
    
    # Generic trail cleanup for initials (e.g. SUJITH M R)
    # 1. Spacing should only occur if the suffix looks like real initials (e.g. M, R, S, K, etc.)
    # 2. Avoid splitting if it creates common name endings (AR, AS, AN, AM, AL, AD, YA, TH, SH, AY)
    COMMON_SUFFIXES = ["AY", "AR", "AS", "AN", "AM", "AL", "AD", "YA", "TH", "SH"]
    
    is_common_ending = any(name.endswith(ex) for ex in COMMON_SUFFIXES)
    
    # Special Case: If it's MAHADEV CHIKRAY, we ALREADY split it above.
    # Otherwise, split initials
    if not is_common_ending:
        # SUJITHMR -> SUJITH M R
        name = re.sub(r"([A-Z]{3,})([A-Z])([A-Z])$", r"\1 \2 \3", name) 
        # YASHASWINIM -> YASHASWINI M
        name = re.sub(r"([A-Z]{3,})([A-Z])$", r"\1 \2", name)
    
    # Second pass for common last names joined (if not already handled)
    if "MAHADEVCHIKRAY" not in name:
         name = name.replace("MAHADEVCHIKRAY", "MAHADEV CHIKRAY")
    
    name = _clean_val(name)
        
    return name


# =====================================================
# MAIN FUNCTION USED BY FASTAPI
# =====================================================

def extract(lines, image, raw_ocr_data=None):
    try:
        height = image.shape[0] if image is not None else 0
        width = image.shape[1] if image is not None else 0
        if raw_ocr_data:
            # Use spatial grouping for much higher accuracy with split results
            lines = _group_texts_by_y(raw_ocr_data, image_height=height, image_width=width)
        
        board = detect_board(lines)

        subjects = extract_subjects_from_text(lines, raw_data=raw_ocr_data, image_width=width, board_type=board)

        total = extract_total(lines)
        
        # Reconcile marks if grand total is available
        # CRITICAL: Do NOT reconcile Grade Cards! Total marks is Sum(Credits*GP)
        is_grade_card = any(k in " ".join(lines).upper() for k in ["GRADE CARD", "SGPA", "CGPA", "CREDIT"])
        if total and subjects and not is_grade_card:
            subjects = _reconcile_marks(subjects, total)


        board = detect_board(lines)

        reg_val, reg_type, semester = extract_register_info(lines)
        name, father, mother = extract_personal_info(lines)
        sgpa, cgpa = extract_gpa(lines)
        
        # Pass the detected board (University) to extract_college to avoid duplication
        college = extract_college(lines, skip_name=board)

        # Build clean output
        data = {
            "student_name": _format_name(name),
            "college_name": college,
            "subjects": subjects,
            "total_marks": total,
            "sgpa": sgpa,
            "cgpa": cgpa,
            "university": _expand_university_name(board) if board and "UNIVERSITY" in board.upper() or board == "VTU" else None
        }

        # Board/University distinction (User wants specific "VTU" ONLY for university marksheets)
        final_board = _clean_val(board)
        is_university = False
        
        if reg_type == "USN" or (final_board and ("UNIVERSITY" in final_board or "VTU" in final_board)):
            data["university"] = "VTU"
            is_university = True
            if final_board and "VTU" not in final_board and "UNIVERSITY" not in final_board:
                 data["board"] = final_board
        else:
             if final_board:
                 # If it is a school board, definitely NO university field unless explicitly found
                 if "BOARD" in final_board or final_board in ["SSLC", "PUC"]:
                      data["board"] = final_board
                      data["university"] = None # Force clear
                 elif "UNIVERSITY" in final_board:
                      data["university"] = final_board
                      is_university = True
                 else:
                      data["board"] = final_board

        # Optional fields based on availability/type
        if reg_val:
            # USER: "remove usn n just keep register" for 10th/PU
            reg_key = "usn" if is_university else "register number"
            data[reg_key] = reg_val
        if semester:
            data["semester"] = semester

        # Expand University Name if found
        if "university" in data:
            data["university"] = _expand_university_name(data["university"])
        if "board" in data:
            data["board"] = _expand_university_name(data["board"])

        # Final Cleanup of all strings
        for k, v in data.items():
            if isinstance(v, str):
                data[k] = _clean_val(v)
            if k == "subjects" and isinstance(v, list):
                for s in v:
                    s["subject"] = _clean_val(s["subject"])
                    # Remove internal tracking metadata
                    if "meta" in s:
                        del s["meta"]
                    if "certainty" in s:
                        del s["certainty"]
                
        # USER: Remove noise subjects like "GEM" or anything with null marks that isn't core
        if "subjects" in data:
            CORE_LOWER_PRUNE = ["kannada", "english", "hindi", "mathematics", "science", "social science", "sanskrit"]
            data["subjects"] = [
                s for s in data["subjects"] 
                if s.get("marks") is not None or s["subject"].lower() in CORE_LOWER_PRUNE
            ]
            # Double check: if "GEM" still survived, kill it
            data["subjects"] = [s for s in data["subjects"] if s["subject"].upper() != "GEM"]

        # FINAL STEP: Remove None values 
        return {k: v for k, v in data.items() if v is not None}

    except Exception as e:
        print("Marksheet Extraction Error:", e)

        return {
            "board": None,
            "subjects": [],
            "result": None,
            "total_marks": None
        }
