import re

def extract(text):
    fields = {}

    pan = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    dob = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
    name = re.search(r"Name\s*[:\-]?\s*(.*)", text, re.IGNORECASE)

    if pan:
        fields["document_number"] = pan.group()
    if dob:
        fields["dob"] = dob.group()
    if name:
        fields["name"] = name.group(1).strip()

    return fields
