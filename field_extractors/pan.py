import re

def extract(texts):
    text = " ".join(texts)

    pan = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text)
    dob = re.search(r"\d{2}/\d{2}/\d{4}", text)

    return {
        "pan_number": pan.group() if pan else None,
        "date_of_birth": dob.group() if dob else None
    }

