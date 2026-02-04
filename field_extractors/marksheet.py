def extract(texts):
    subjects = [t for t in texts if t.isalpha() and len(t) > 3]

    return {
        "subjects_detected": subjects[:10],
        "result": "PASS" if "PASS" in texts else "UNKNOWN"
    }

