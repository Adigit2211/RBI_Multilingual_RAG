"""
Builds the labeled train/val dataset for the department-classification
LoRA fine-tuning task.

Design notes:
- Input text = each circular's FIRST chunk only (not the full document).
  Department is reliably inferable from the reference number, title, and
  opening paragraph, and DistilBERT's 512-token limit can't fit a full
  Directions document anyway.
- Excludes UNKNOWN-department circulars (no ground truth label available).
- Merges departments with only 1 example (DCM, FMRD, CO, DPSS) into an
  OTHER class -- a single example can't be learned or validated on, so
  pretending otherwise would just be memorization, not classification.
"""

import json
import random
from collections import Counter, defaultdict

CHUNKS_PATH = "data/processed/chunks.jsonl"
TRAIN_OUT = "finetune/train.jsonl"
VAL_OUT = "finetune/val.jsonl"
MIN_CLASS_SIZE = 5  # departments with fewer examples than this get merged into OTHER
VAL_FRACTION = 0.2
SEED = 42


def load_first_chunk_per_document():
    """For each source PDF, grab only its first chunk (chunk_id order preserved from extraction)."""
    seen_docs = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            pdf = chunk["source_pdf"]
            if pdf not in seen_docs:
                seen_docs[pdf] = chunk
    return list(seen_docs.values())


def main():
    random.seed(SEED)

    docs = load_first_chunk_per_document()
    print(f"Loaded {len(docs)} documents (one chunk each)")

    labeled = [d for d in docs if d["department_code"] and d["department_code"] != "UNKNOWN"]
    print(f"{len(labeled)} documents have a known department label")

    label_counts = Counter(d["department_code"] for d in labeled)
    print("Raw label distribution:", dict(label_counts))

    rare_labels = {label for label, count in label_counts.items() if count < MIN_CLASS_SIZE}
    print(f"Merging rare labels into OTHER: {rare_labels}")

    for d in labeled:
        if d["department_code"] in rare_labels:
            d["department_code"] = "OTHER"

    final_counts = Counter(d["department_code"] for d in labeled)
    print("Final label distribution:", dict(final_counts))

    # Stratified split: shuffle within each class, then split each class proportionally
    by_class = defaultdict(list)
    for d in labeled:
        by_class[d["department_code"]].append(d)

    train_docs, val_docs = [], []
    for label, items in by_class.items():
        random.shuffle(items)
        n_val = max(1, round(len(items) * VAL_FRACTION))
        val_docs.extend(items[:n_val])
        train_docs.extend(items[n_val:])

    random.shuffle(train_docs)
    random.shuffle(val_docs)

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        for d in train_docs:
            f.write(json.dumps({"text": d["text"], "label": d["department_code"]}, ensure_ascii=False) + "\n")

    with open(VAL_OUT, "w", encoding="utf-8") as f:
        for d in val_docs:
            f.write(json.dumps({"text": d["text"], "label": d["department_code"]}, ensure_ascii=False) + "\n")

    print(f"\nTrain set: {len(train_docs)} examples -> {TRAIN_OUT}")
    print(f"Val set: {len(val_docs)} examples -> {VAL_OUT}")


if __name__ == "__main__":
    main()

