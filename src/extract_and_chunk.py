"""
PDF text extraction + structural chunking for RBI circulars.

Design decisions (why this isn't a naive fixed-size splitter):

1. RBI documents come in two structural shapes:
   - Regular circulars: flat numbered paragraphs (1., 2., 2.1)
   - Formal "Directions": Chapter I/II/III -> lettered sections (A., B., C.)
     -> numbered paragraphs within each section
   We track chapter/section state while scanning so every paragraph knows
   where it lives structurally (useful for precise citations later).

2. Every page carries repeated boilerplate: a bilingual (Hindi/English)
   letterhead with RBI's postal address, phone, and email, and sometimes a
   phishing-caution notice. This is stripped before chunking so it doesn't
   pollute embeddings or waste tokens repeated across every chunk.

3. Primary chunk boundary = numbered paragraph (the one structural unit
   present in both document shapes). A paragraph is only further split, with
   token-capped overlap, if it exceeds ~500 tokens on its own -- most RBI
   paragraphs are much shorter than that, so this rarely triggers.
"""

import csv
import json
import os
import re

import tiktoken
from pypdf import PdfReader

RAW_DIR = "data/raw"
METADATA_PATH = "data/raw/metadata.csv"
OUTPUT_PATH = "data/processed/chunks.jsonl"

MAX_TOKENS = 500
OVERLAP_TOKENS = 60

encoding = tiktoken.get_encoding("cl100k_base")

DEVANAGARI_LINE = re.compile(r"^[\u0900-\u097F\s.,:/–\-]+$")
BOILERPLATE_PATTERNS = [
    re.compile(r"Tel(ephone)?\s*(No)?[:.]?\s*[\d,\s-]+", re.IGNORECASE),
    re.compile(r"Fax\s*(No)?[:.]?\s*[\d,\s-]+", re.IGNORECASE),
    re.compile(r"E-?mail[:.]?\s*\S+@\S+", re.IGNORECASE),
    re.compile(r"Caution:\s*RBI never sends", re.IGNORECASE),
    re.compile(r"Department of \w+.*Reserve Bank of India.*Mumbai", re.IGNORECASE),
]

CHAPTER_PATTERN = re.compile(r"^Chapter\s+([IVXLC]+)\s*[-–]?\s*(.*)$")
SECTION_PATTERN = re.compile(r"^([A-Z])\.\s+(.+)$")
PARAGRAPH_PATTERN = re.compile(r"^(\d+)\.\s+(.*)$")
TOC_START = re.compile(r"Table of Contents", re.IGNORECASE)
TOC_ENTRY_PATTERN = re.compile(r"\.{2,}\s*\d+$")
REFERENCE_ANCHOR = re.compile(r"RBI\s*/\s*(?:[A-Za-z](?:\s?[A-Za-z]){0,5}\s*/\s*)?\d(?:\s?\d){3}\s*-\s*\d(?:\s?\d)?\s*/\s*\d+|Notification No\.\s*FEMA",
re.IGNORECASE)
PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}$")

def load_metadata():
    with open(METADATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["local_filename"]: row for row in rows}


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def clean_text(raw_text):
    # Discard everything before the circular's own reference number -- this
    # is where the bilingual letterhead, postal address, phone/email, and
    # phishing-caution notice all live. Anchoring on a structural marker we
    # know is reliably present eliminates this whole class of noise in one
    # step, instead of pattern-matching each noise variant individually.
    anchor_match = REFERENCE_ANCHOR.search(raw_text)
    if anchor_match:
        raw_text = raw_text[anchor_match.start():]

    lines = raw_text.split("\n")
    cleaned_lines = []
    in_toc = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if DEVANAGARI_LINE.match(stripped):
            continue

        if PAGE_NUMBER_LINE.match(stripped):
            continue

        if any(p.search(stripped) for p in BOILERPLATE_PATTERNS):
            continue

        if TOC_START.search(stripped):
            in_toc = True
            continue

        if in_toc:
            if TOC_ENTRY_PATTERN.search(stripped):
                continue
            if CHAPTER_PATTERN.match(stripped) or PARAGRAPH_PATTERN.match(stripped):
                in_toc = False
            else:
                continue

        cleaned_lines.append(stripped)

    return cleaned_lines


def split_into_structural_paragraphs(cleaned_lines):
    paragraphs = []
    current_chapter = None
    current_section = None
    current_para_num = None
    current_para_lines = []

    def flush():
        if current_para_lines:
            paragraphs.append({
                "chapter": current_chapter,
                "section": current_section,
                "paragraph_number": current_para_num,
                "text": " ".join(current_para_lines).strip(),
            })

    for line in cleaned_lines:
        chapter_match = CHAPTER_PATTERN.match(line)
        section_match = SECTION_PATTERN.match(line)
        para_match = PARAGRAPH_PATTERN.match(line)

        if chapter_match:
            flush()
            current_para_lines = []
            current_chapter = f"Chapter {chapter_match.group(1)} - {chapter_match.group(2)}".strip(" -")
            current_para_num = None
            continue

        if section_match and len(section_match.group(1)) == 1:
            if len(section_match.group(2)) < 80:
                flush()
                current_para_lines = []
                current_section = f"Section {section_match.group(1)} - {section_match.group(2)}".strip()
                current_para_num = None
                continue

        if para_match:
            flush()
            current_para_lines = [line]
            current_para_num = para_match.group(1)
            continue

        current_para_lines.append(line)

    flush()
    return [p for p in paragraphs if p["text"]]


def token_split_if_needed(paragraph_text):
    tokens = encoding.encode(paragraph_text)
    if len(tokens) <= MAX_TOKENS:
        return [paragraph_text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS
    return chunks


def process_pdf(local_filename, meta_row):
    pdf_path = os.path.join(RAW_DIR, local_filename)
    raw_text = extract_pdf_text(pdf_path)
    cleaned_lines = clean_text(raw_text)
    structural_paragraphs = split_into_structural_paragraphs(cleaned_lines)

    chunks = []
    for para in structural_paragraphs:
        sub_chunks = token_split_if_needed(para["text"])
        for i, sub_text in enumerate(sub_chunks):
            if len(sub_text.strip()) < 20:
                continue
            chunks.append({
                "text": sub_text,
                "chapter": para["chapter"],
                "section": para["section"],
                "paragraph_number": para["paragraph_number"],
                "sub_chunk_index": i if len(sub_chunks) > 1 else None,
                "source_pdf": local_filename,
                "circular_id": meta_row["id"],
                "title": meta_row["title"],
                "date": meta_row["date"],
                "department_code": meta_row["department_code"],
            })
    return chunks


def main():
    metadata = load_metadata()
    os.makedirs("data/processed", exist_ok=True)

    all_chunks = []
    skipped = 0

    for local_filename, meta_row in metadata.items():
        pdf_path = os.path.join(RAW_DIR, local_filename)
        if not os.path.exists(pdf_path):
            print(f"WARNING: PDF not found for {local_filename}, skipping")
            skipped += 1
            continue

        try:
            chunks = process_pdf(local_filename, meta_row)
            all_chunks.extend(chunks)
            print(f"{local_filename}: {len(chunks)} chunks")
        except Exception as e:
            print(f"ERROR processing {local_filename}: {e}")
            skipped += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(all_chunks):
            chunk["chunk_id"] = f"chunk_{i:05d}"
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_chunks)} total chunks from {len(metadata) - skipped} PDFs "
          f"({skipped} skipped). Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
