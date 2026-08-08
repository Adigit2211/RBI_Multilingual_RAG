"""
Retrieval + generation pipeline for the RBI multilingual RAG assistant.

Key design choices:

1. Citations come from retrieval metadata, not from the LLM. We never ask
   the model to recall or repeat source names -- it only sees numbered
   context blocks ([Source 1], [Source 2]...) and we map those numbers
   back to real circular titles/dates/sections ourselves afterward. This
   avoids the common failure mode of LLMs fabricating plausible-looking
   but incorrect citations.

2. Low-confidence flagging uses Chroma's L2 distance on the top match.
   Below CONFIDENCE_DISTANCE_THRESHOLD we still attempt an answer, but
   warn the user the grounding is weak, rather than silently returning a
   confident-sounding answer built on a poor retrieval match.

3. The prompt explicitly instructs the model to answer only from the
   provided context and say so if the answer isn't present -- the core
   anti-hallucination mechanism for RAG.
"""

import os

import chromadb
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from language_utils import detect_language

load_dotenv()

CHROMA_DIR = "data/processed/chroma_db"
COLLECTION_NAME = "rbi_circulars"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
GENERATION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

TOP_K = 5
CONFIDENCE_DISTANCE_THRESHOLD = 0.35  # empirically: >0.35 = weak match, based on Step 4's test queries

SYSTEM_PROMPT = """You are a compliance assistant answering questions about RBI (Reserve Bank of India) circulars and Directions.

Rules:
- Answer ONLY using the numbered source excerpts provided below. Do not use any outside knowledge.
- If the provided sources do not contain enough information to answer, say so clearly instead of guessing.
- When you use information from a source, reference it by its number, e.g. "[Source 2]".
- Be precise and concise. This is used for regulatory compliance, so accuracy matters more than fluency.
- Respond in the same language the user asked the question in. If the question is in Hindi, answer in Hindi. If it's code-mixed Hindi-English, you may respond in a similarly natural code-mixed style."""

def load_models():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    hf_client = InferenceClient(token=os.environ["HF_TOKEN"])
    return embed_model, collection, hf_client


def retrieve(query, embed_model, collection, top_k=TOP_K):
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

    user_message = f"""Context sources:

{context_text}

Question: {query}

Answer the question using only the sources above, citing them by number."""

    return user_message


def format_citations(retrieved):
    citations = []
    for i, meta in enumerate(retrieved["metadatas"]):
        citations.append({
            "source_number": i + 1,
            "title": meta["title"],
            "date": meta["date"],
            "department": meta["department_code"],
            "section": meta.get("section") or None,
            "circular_id": meta["circular_id"],
        })
    return citations


def answer_question(query, embed_model, collection, hf_client):
    lang_info = detect_language(query)
    retrieved = retrieve(query, embed_model, collection)
    best_distance = retrieved["distances"][0]

    low_confidence = best_distance > CONFIDENCE_DISTANCE_THRESHOLD

    user_message = build_prompt(query, retrieved)

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
    citations = format_citations(retrieved)

    return {
        "answer": answer_text,
        "citations": citations,
        "best_match_distance": best_distance,
        "low_confidence": low_confidence,
        "detected_language": lang_info,
      }


def main():
    print("Loading models and vector store...")
    embed_model, collection, hf_client = load_models()
    print("Ready.\n")

    while True:
        query = input("Ask a question about RBI circulars (or 'quit'): ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        result = answer_question(query, embed_model, collection, hf_client)

        print()
        print(f"[Detected language: {result['detected_language']['language']} "
              f"(via {result['detected_language']['method']})]")

        if result["low_confidence"]:
            print(f"[Note: best retrieval match was weak (distance={result['best_match_distance']:.3f}). "
                  f"This answer may not be well-grounded in the available circulars.]")

        print()
        print("Answer:")
        print(result["answer"])
        print()
        print("Sources:")
        for c in result["citations"]:
            section = f", {c['section']}" if c["section"] else ""
            print(f"  [{c['source_number']}] {c['title']} ({c['date']}, Dept: {c['department']}{section})")
        print()


if __name__ == "__main__":
    main()
