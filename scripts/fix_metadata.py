"""
Repairs metadata.csv department_code / reference_number extraction using
the already-downloaded HTML text files, without re-hitting RBI's servers.

Bug being fixed: the original extraction assumed the circular's reference
number (e.g. 'RBI/2026-27/222') and its department code (e.g.
'DOR.AML.REC.192/14.06.001/2026-27') would appear on the same text line.
In practice RBI's markup sometimes splits them across separate tags, so
BeautifulSoup's get_text(separator="\n") inserted a newline between them,
and the single-line regex missed the department code. Fix: search a
window of characters AFTER the reference number, with newlines flattened
to spaces, instead of assuming same-line.
"""

import csv
import os
import re

METADATA_PATH = "data/raw/metadata.csv"
HTML_TEXT_DIR = "data/raw/html_text"


def extract_reference_and_department(body_text):
    # Reference formats seen so far:
    #   1. RBI/YYYY-YY/NNN              (dept code follows as separate token)
    #   2. RBI/DEPT/YYYY-YY/NNN         (dept code embedded in the reference)
    # Department codes are usually all-caps (DOR, DPSS) but not always
    # (DoS uses mixed case) — so we match case-insensitively and normalize
    # to uppercase afterward, rather than assuming a fixed casing.
    ref_match = re.search(r"RBI/(?:([A-Za-z]{2,6})/)?\d{4}-\d{2}/\d+", body_text)
    if not ref_match:
        return "", "UNKNOWN"

    window = body_text[ref_match.start(): ref_match.start() + 200]
    window = re.sub(r"\s+", " ", window.replace("\n", " ")).strip()

    if ref_match.group(1):
        department_code = ref_match.group(1).upper()
    else:
        after_ref = body_text[ref_match.end():].lstrip(" \t\n")
        dept_match = re.match(r"([A-Za-z]{2,6})(?=[.\s(])", after_ref)
        department_code = dept_match.group(1).upper() if dept_match else "UNKNOWN"

    return window, department_code

def main():
    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixed_count = 0
    for row in rows:
        text_filename = row["local_filename"].replace(".pdf", ".txt")
        text_path = os.path.join(HTML_TEXT_DIR, text_filename)

        if not os.path.exists(text_path):
            print(f"WARNING: no cached text file for {row['local_filename']}, skipping")
            continue

        with open(text_path, encoding="utf-8") as tf:
            body_text = tf.read()

        reference_number, department_code = extract_reference_and_department(body_text)
        row["reference_number"] = reference_number
        row["department_code"] = department_code
        fixed_count += 1

    fieldnames = ["id", "title", "date", "reference_number", "department_code",
                  "pdf_url", "local_filename"]
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Reprocessed {fixed_count} rows. metadata.csv updated in place.")


if __name__ == "__main__":
    main()
