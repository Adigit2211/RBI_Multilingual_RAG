# RBI Multilingual RAG Assistant

An end-to-end multilingual Retrieval-Augmented Generation (RAG) system over
RBI (Reserve Bank of India) regulatory circulars, with a LoRA fine-tuned
classifier as a complementary component — built to demonstrate the design
judgment (RAG vs. fine-tuning, chunking strategy, groundedness, multilingual
handling) relevant to conversational-AI and enterprise-document NLP roles.

## What this does

Ask a question about RBI circulars — in English, Hindi, or code-mixed
Hindi-English — and get a grounded answer with citations back to the actual
source circulars, or an explicit statement that the available circulars
don't contain the answer, rather than a hallucinated response.

Separately, a LoRA fine-tuned classifier automatically tags a circular's
issuing department (DOR, DOS, FIDD, etc.) — a narrow, high-volume,
closed-set task deliberately handled by fine-tuning rather than by the
RAG/LLM pipeline (see below for why).

## Architecture

RBI website (150 circulars)
│ scrape (sequential notification ID walk)
▼
Raw PDFs + metadata.csv (title, date, reference number, department)
│ pypdf extraction + structural chunking
▼
chunks.jsonl (6,531 chunks, tagged with chapter/section/paragraph + metadata)
│ embed (intfloat/multilingual-e5-small)
▼
Chroma vector store (persistent, local)
│ retrieve top-k + confidence check
▼
Prompt (numbered context blocks) → LLM (Llama-3.3-70B via HF Inference)
│
▼
Grounded answer + citations (mapped from retrieval metadata, not LLM recall)

A separate, parallel path: `chunks.jsonl` → first-chunk-per-document →

## Why RAG for Q&A, fine-tuning for classification

This is the central design decision of the project, and the one worth
being able to explain in depth:

**RAG is the right tool when the answer space is open-ended and must be
grounded in retrieved evidence.** "What does this circular require" has
no fixed answer set — the system needs to find the relevant text and
generate a response conditioned on it, with citations, so a human can
verify the claim against the source.

**Fine-tuning is the right tool when the task is a fixed, closed-set,
high-volume decision.** Classifying a circular into one of a handful of
departments is exactly this: the label space is bounded, the task repeats
every time a new document arrives, and you want fast, cheap, deterministic
inference — not a full LLM call with non-deterministic phrasing every
time. A LoRA-adapted DistilBERT classifies a document in milliseconds for
near-zero marginal cost; using an LLM prompt for the same task would work,
but would be slower, non-deterministic, and far more expensive at scale
for no accuracy benefit.

## Key design decisions

**Chunking (structural, not fixed-size):** RBI documents come in two
structural shapes — flat numbered-paragraph circulars, and formal
"Directions" documents with a Chapter → lettered Section → numbered
paragraph hierarchy. Chunk boundaries are set at numbered paragraphs (the
structural unit present in both shapes), with a 500-token/60-token-overlap
cap applied only when an individual paragraph is unusually long. This
preserves clause-level integrity — a paragraph referencing "as per para 3
above" isn't severed from paragraph 3 by an arbitrary fixed-size window.

**Embeddings:** `intfloat/multilingual-e5-small`, chosen specifically for
its cross-lingual embedding space — a Hindi query and its English
translation land close together in vector space, enabling retrieval
across languages without a separate translation step. Verified directly:
an English and a Hindi phrasing of the same cybersecurity question both
retrieve the same relevant source chunks.

**Groundedness:** every answer is generated from a prompt containing only
retrieved chunks, with an explicit system instruction to answer only from
provided context and say so if the context doesn't contain the answer.
Citations shown to the user come from retrieval metadata, not from asking
the LLM to recall source names — avoiding fabricated citations. A distance
threshold on the top retrieval match flags low-confidence answers before
generation, rather than letting the LLM improvise on a poor match.

**Multilingual query handling:** two-layer language detection —
Devanagari/Latin script counting first (deterministic, catches code-mixing
unambiguously), `langdetect` as fallback for pure-Latin-script queries to
distinguish English from romanized Hindi. Known limitation: `langdetect`
performs poorly on short romanized-Hindi phrases with no script cues to
anchor on (documented with a concrete failing example in
`src/language_utils.py`) — a production system would use an Indic-specific
LID model instead.

## Known limitations (stated explicitly)

- **Department classifier:** trained on only 136 labeled circulars across
  4 classes after merging 4 single-example departments into `OTHER`. DOR
  and DOS (93% of labeled data) classify strongly (93–95% F1); the merged
  `OTHER` class fails entirely (0 F1) due to having only 3 training
  examples spanning 4 unrelated departments. This is a corpus-size
  limitation, not a modeling error — full results in `finetune/RESULTS.md`.
- **9% of scraped circulars (14/150)** could not be department-classified
  by regex extraction: some (FEMA notifications) use a genuinely different
  reference-numbering convention issued by RBI's Foreign Exchange
  Department; others were "Directions" index pages that didn't expose
  full reference metadata in their HTML at scrape time.
- **Romanized Hindi language detection** is unreliable on short phrases
  (see above).
- **Live deployment:** attempted on Hugging Face Spaces (free-tier Gradio
  hosting was restricted to GPU-oriented ZeroGPU-only access partway
  through this project, not available on this account), then Render.com
  free tier (512MB memory limit insufficient for the PyTorch +
  sentence-transformers + embedding model footprint — an inherent
  constraint of transformer-based ML on free compute, not an application
  bug), then Google Cloud Run (in progress). The full application is
  verified working end-to-end locally (see screenshots/testing evidence
  in commit history) — see "Running locally" below to run it yourself.

## Project structure

## Project structure

```
scripts/            RBI scraper, metadata repair script
src/                 PDF extraction/chunking, embeddings, RAG pipeline, language detection
finetune/            Fine-tuning dataset prep, results
notebooks/           Colab LoRA fine-tuning notebook
app/                 Gradio chat interface
data/raw/            Scraped PDFs + metadata (PDFs gitignored, regenerable via scraper)
data/processed/      Chunked text (committed) + Chroma vector store (gitignored, regenerable)
```

## Running locally

```bash
git clone https://github.com/Adigit2211/RBI_Multilingual_RAG.git
cd RBI_Multilingual_RAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# .env file with your Hugging Face token (Inference permission)
echo "HF_TOKEN=your_token_here" > .env

# Rebuild the pipeline from scratch, or skip to the last step if
# data/processed/chunks.jsonl is already present (it's committed)
python3 scripts/scrape_rbi.py            # ~5 min, hits RBI's servers
python3 scripts/fix_metadata.py
python3 src/extract_and_chunk.py
python3 src/build_vector_store.py        # builds data/processed/chroma_db/

# CLI chat
python3 src/rag_pipeline.py

# Or the Gradio UI
python3 app/app.py   # then open http://127.0.0.1:7860
```

## Tech stack

Scraping: `requests`, `beautifulsoup4` · PDF extraction: `pypdf` ·
Chunking: `tiktoken` · Embeddings: `sentence-transformers`
(`intfloat/multilingual-e5-small`) · Vector store: `chromadb` ·
Generation: `meta-llama/Llama-3.3-70B-Instruct` via Hugging Face Inference
Providers · Fine-tuning: `transformers` + `peft` (LoRA) on
`distilbert-base-multilingual-cased`, trained on Colab free T4 GPU ·
Language detection: `langdetect` + custom script-based logic · UI:
`gradio`

## License

MIT — see [LICENSE](LICENSE).
