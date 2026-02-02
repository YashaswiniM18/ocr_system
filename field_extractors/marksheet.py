import re

def extract(text):
    fields = {}

    board = re.search(r"(Board of .*|University of .*|CBSE|ICSE)", text, re.IGNORECASE)
    result = re.search(r"(PASS|FAIL|DISTINCTION|FIRST CLASS)", text, re.IGNORECASE)

    subjects = re.findall(r"(Maths|Mathematics|Physics|Chemistry|Biology|English|History|Geography|Computer Science)", text, re.IGNORECASE)
    marks = re.findall(r"\b\d{2,3}\b", text)

    if board:
        fields["board_or_university"] = board.group()
    if result:
        fields["result"] = result.group().title()
    if subjects:
        fields["subjects"] = list(set(subjects))
    if marks:
        fields["marks"] = marks[:10]

    return fields
