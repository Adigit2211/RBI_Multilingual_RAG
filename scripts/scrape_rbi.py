"""
RBI Circular Scraper
---------------------
Walks backward through RBI's sequential notification IDs, extracts each
circular's title, date, department code, and PDF link, downloads the PDF,
and records everything in a metadata CSV.

Why this approach (not the month/year archive links):
RBI's archive navigation uses ASP.NET postback links (javascript:void(0)),
which a plain requests-based scraper can't trigger. But each circular's
detail page URL uses a plain sequential integer ID
(NotificationUser.aspx?Id=13664&Mode=0), so walking IDs backward from the
newest one is a reliable way to collect a batch without needing a browser.
"""

import os
import re
import csv
import time
import requests
from bs4 import BeautifulSoup

BASE_DETAIL_URL = "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id={id}&Mode=0"
MAIN_LIST_URL = "https://www.rbi.org.in/Scripts/NotificationUser.aspx"
RAW_DIR = "data/raw"
METADATA_PATH = "data/raw/metadata.csv"
TARGET_COUNT = 150          # how many circulars we want to successfully collect
MAX_ID_ATTEMPTS = 400       # safety cap so a bad run can't loop forever
REQUEST_DELAY_SECONDS = 1.0 # be polite to RBI's servers

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def get_latest_id():
    """Fetch the main notifications page and find the highest circular ID."""
    resp = requests.get(MAIN_LIST_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    ids = re.findall(r"NotificationUser\.aspx\?Id=(\d+)&Mode=0", resp.text)
    ids = [int(i) for i in ids]
    if not ids:
        raise RuntimeError("Could not find any notification IDs on the main page — "
                            "the site structure may have changed.")
    return max(ids)


def extract_department_code(reference_text):
    """
    RBI reference numbers look like: 'RBI/2026-27/222 DOR.AML.REC.192/14.06.001/2026-27'
    The department code is the letters right after the second '/', before the first '.'
    """
    match = re.search(r"RBI/\d{4}-\d{2}/\d+\s+([A-Z]+)\.", reference_text)
    if match:
        return match.group(1)
    return "UNKNOWN"


def parse_detail_page(html, circular_id):
    """Extract title, date, reference number, department code, PDF URL, and body text."""
    soup = BeautifulSoup(html, "html.parser")

    pdf_link_tag = soup.find("a", href=re.compile(r"rbidocs\.rbi\.org\.in.*\.PDF", re.IGNORECASE))
    if not pdf_link_tag:
        return None  # no PDF on this page — skip (could be a withdrawn/non-circular entry)
    pdf_url = pdf_link_tag["href"]

    title_tag = soup.find("b") or soup.find("strong")
    title = title_tag.get_text(strip=True) if title_tag else f"circular_{circular_id}"

    body_text = soup.get_text(separator="\n", strip=True)

    ref_match = re.search(r"RBI/\d{4}-\d{2}/\d+[^\n]*", body_text)
    reference_number = ref_match.group(0) if ref_match else ""

    date_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        body_text,
    )
    date_str = date_match.group(0) if date_match else ""

    department_code = extract_department_code(reference_number)

    return {
        "id": circular_id,
        "title": title,
        "date": date_str,
        "reference_number": reference_number,
        "department_code": department_code,
        "pdf_url": pdf_url,
        "html_text": body_text,
    }


def slugify(text, max_len=60):
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len]


def download_pdf(pdf_url, local_path):
    resp = requests.get(pdf_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Finding the latest circular ID...")
    latest_id = get_latest_id()
    print(f"Latest ID found: {latest_id}")

    collected = 0
    attempted = 0
    rows = []

    current_id = latest_id
    while collected < TARGET_COUNT and attempted < MAX_ID_ATTEMPTS:
        attempted += 1
        url = BASE_DETAIL_URL.format(id=current_id)
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                parsed = parse_detail_page(resp.text, current_id)
                if parsed:
                    date_part = parsed["date"].replace(",", "").replace(" ", "-") or "nodate"
                    filename = f"{date_part}_{slugify(parsed['title'])}.pdf"
                    local_path = os.path.join(RAW_DIR, filename)

                    download_pdf(parsed["pdf_url"], local_path)
                    parsed["local_filename"] = filename
                    rows.append(parsed)
                    collected += 1
                    print(f"[{collected}/{TARGET_COUNT}] Saved: {filename}")
                else:
                    print(f"ID {current_id}: no PDF found, skipping")
            else:
                print(f"ID {current_id}: HTTP {resp.status_code}, skipping")
        except requests.RequestException as e:
            print(f"ID {current_id}: request failed ({e}), skipping")

        current_id -= 1
        time.sleep(REQUEST_DELAY_SECONDS)

    if rows:
        fieldnames = ["id", "title", "date", "reference_number", "department_code",
                      "pdf_url", "local_filename"]
        with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row[k] for k in fieldnames})

        text_dir = os.path.join(RAW_DIR, "html_text")
        os.makedirs(text_dir, exist_ok=True)
        for row in rows:
            text_path = os.path.join(text_dir, row["local_filename"].replace(".pdf", ".txt"))
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(row["html_text"])

    print(f"\nDone. Collected {collected} circulars out of {attempted} IDs attempted.")
    print(f"Metadata written to {METADATA_PATH}")


if __name__ == "__main__":
    main()
