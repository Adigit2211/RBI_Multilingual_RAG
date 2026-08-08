"""
Gradio chat interface for the RBI Multilingual RAG Assistant.
Deployed on Hugging Face Spaces.

Startup behavior: the Chroma vector store is built fresh from the
committed chunks.jsonl on first launch, rather than persisting a binary
DB in git or a separate blob store. This trades a short (1-2 min)
cold-start delay for a simpler, fully reproducible deployment -- anyone
who forks this Space gets a working index automatically, with no extra
infrastructure to set up.
"""

import os
import sys

import chromadb
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from language_utils import detect_language

CHUNKS_PATH = "data/processed/chunks.jsonl"
CHROMA_DIR = "data/processed/chroma_db"
COLLECTION_NAME = "rbi_circulars"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
GENERATION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

TOP_K = 5
CONFIDENCE_DISTANCE_THRESHOLD = 0.35

SYSTEM_PROMPT = """You are a compliance assistant answering questions about RBI (Reserve Bank of India) circulars and Directions.

Rules:
- Answer ONLY using the numbered source excerpts provided below. Do not use any outside knowledge.
- If the provided sources do not contain enough information to answer, say so clearly instead of guessing.
- When you use information from a source, reference it by its number, e.g. "[Source 2]".
- Be precise and concise. This is used for regulatory compliance, so accuracy matters more than fluency.
- Respond in the same language the user asked the question in. If the question is in Hindi, answer in Hindi. If it's code-mixed Hindi-English, you may respond in a similarly natural code-mixed style."""


def ensure_vector_store_built(embed_model):
    """Build the Chroma collection from chunks.jsonl if it doesn't already exist."""
    import json

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing_collections = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing_collections:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            print(f"Vector store already built ({collection.count()} vectors). Skipping rebuild.")
            return collection

    print("Building vector store from chunks.jsonl (first launch)...")
    collection = client.get_or_create_collection(COLLECTION_NAME)

    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    batch_size = 64
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = ["passage: " + c["text"] for c in batch]
        embeddings = embed_model.encode(texts, show_progress_bar=False).tolist()
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [{
            "source_pdf": c["source_pdf"] or "",
            "circular_id": str(c["circular_id"]) or "",
            "title": c["title"] or "",
            "date": c["date"] or "",
            "department_code": c["department_code"] or "",
            "chapter": c["chapter"] or "",
            "section": c["section"] or "",
            "paragraph_number": c["paragraph_number"] or "",
        } for c in batch]
        collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"Vector store built: {collection.count()} vectors.")
    return collection


print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
collection = ensure_vector_store_built(embed_model)
hf_client = InferenceClient(token=os.environ["HF_TOKEN"])
print("Ready.")


def retrieve(query, top_k=TOP_K):
    query_embedding = embed_model.encode(["query: " + query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


def build_prompt(query, retrieved):
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(retrieved["documents"], retrieved["metadatas"])):
        location = meta["title"]
        if meta.get("section"):
            location += f" ({meta['section']})"
        context_blocks.append(f"[Source {i+1}] ({location})\n{doc}")
    context_text = "\n\n".join(context_blocks)
    return f"""Context sources:

{context_text}

Question: {query}

Answer the question using only the sources above, citing them by number."""


def respond(message, history):
    lang_info = detect_language(message)
    retrieved = retrieve(message)
    best_distance = retrieved["distances"][0]
    low_confidence = best_distance > CONFIDENCE_DISTANCE_THRESHOLD

    user_message = build_prompt(message, retrieved)

    response = hf_client.chat_completion(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=500,
        temperature=0.2,
    )
    answer_text = response.choices[0].message.content

    output = ""
    output += f"*Detected language: {lang_info['language']} (via {lang_info['method']})*\n\n"
    if low_confidence:
        output += (f"⚠️ *Best retrieval match was weak (distance={best_distance:.3f}). "
                   f"This answer may not be well-grounded in the available circulars.*\n\n")

    output += answer_text + "\n\n---\n**Sources:**\n"
    for i, meta in enumerate(retrieved["metadatas"]):
        section = f", {meta['section']}" if meta.get("section") else ""
        output += f"\n[{i+1}] {meta['title']} ({meta['date']}, Dept: {meta['department_code']}{section})"

    return output


demo = gr.ChatInterface(
    fn=respond,
    title="RBI Circular Assistant",
    description=(
        "Ask questions about RBI circulars and Directions in English, Hindi, or code-mixed "
        "Hindi-English. Answers are grounded in retrieved source text with citations, and "
        "low-confidence retrievals are flagged rather than answered speculatively."
    ),
    examples=[
        "What are the cybersecurity requirements for banks?",
        "बैंकों के लिए साइबर सुरक्षा आवश्यकताएं क्या हैं?",
        "What is the Kisan Credit Card scheme?",
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
