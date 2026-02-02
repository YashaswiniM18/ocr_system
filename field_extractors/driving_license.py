import re

def extract(text):
    fields = {}

    dl = re.search(r"\b[A-Z]{2}\d{13}\b", text)
    dob = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
    name = re.search(r"Name\s*[:\-]?\s*(.*)", text, re.IGNORECASE)

    if dl:
        fields["document_number"] = dl.group()
    if dob:
        fields["dob"] = dob.group()
    if name:
        fields["name"] = name.group(1).strip()

    gender = re.search(r"\b(MALE|FEMALE|OTHER)\b", text.upper())
    if gender:
        fields["gender"] = gender.group().title()

    return fields
