"""
Embeds all chunks from chunks.jsonl using intfloat/multilingual-e5-small
and stores them in a persistent local Chroma collection.

Why e5's prefix convention matters: e5 models are trained to expect a
"passage: " prefix on indexed documents and a "query: " prefix on search
queries. These produce different (asymmetric) representations by design --
skipping or swapping the prefix silently degrades retrieval quality rather
than raising an error, so it's easy to miss if you don't know to look for it.
This script handles the indexing side ("passage: "); the retrieval script
(next step) handles the query side ("query: ").
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_DIR = "data/processed/chroma_db"
COLLECTION_NAME = "rbi_circulars"
MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 64


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def clean_metadata(chunk):
    """Chroma metadata values must be str/int/float/bool, not None."""
    return {
        "source_pdf": chunk["source_pdf"] or "",
        "circular_id": str(chunk["circular_id"]) or "",
        "title": chunk["title"] or "",
        "date": chunk["date"] or "",
        "department_code": chunk["department_code"] or "",
        "chapter": chunk["chapter"] or "",
        "section": chunk["section"] or "",
        "paragraph_number": chunk["paragraph_number"] or "",
    }


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading embedding model ({MODEL_NAME})... this may take a minute on first run")
    model = SentenceTransformer(MODEL_NAME)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh start each run -- if the collection already exists, drop it,
    # so re-running this script after a chunking change doesn't leave stale
    # or duplicate vectors behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    print(f"Embedding and inserting {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts_with_prefix = ["passage: " + c["text"] for c in batch]
        embeddings = model.encode(texts_with_prefix, show_progress_bar=False).tolist()

        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]  # store raw text (no prefix) for display
        metadatas = [clean_metadata(c) for c in batch]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        done = min(start + BATCH_SIZE, len(chunks))
        print(f"  {done}/{len(chunks)} chunks embedded")

    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} vectors.")
    print(f"Stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
