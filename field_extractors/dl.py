import re

def extract(texts):
    text = " ".join(texts)

    dl = re.search(r"[A-Z]{2}[0-9]{13}", text)

    return {
        "driving_license_number": dl.group() if dl else None
    }
