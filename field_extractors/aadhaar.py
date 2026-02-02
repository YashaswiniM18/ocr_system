import re

def extract(text):
    fields = {}

    aadhaar = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
    dob = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text)
    gender = re.search(r"\b(MALE|FEMALE|OTHER)\b", text.upper())

    name = re.search(r"\n([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\n", text)

    if aadhaar:
        fields["document_number"] = aadhaar.group()
    if dob:
        fields["dob"] = dob.group()
    if gender:
        fields["gender"] = gender.group().title()
    if name:
        fields["name"] = name.group(1)

    return fields
