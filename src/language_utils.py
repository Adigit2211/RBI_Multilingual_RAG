"""
Lightweight language detection/routing for the RAG assistant.

Detection strategy (two layers, in order of reliability):
1. Script-based check: count Devanagari vs Latin characters directly in
   the query. If both scripts appear, that's unambiguous evidence of
   code-mixing -- cheap, deterministic, no model needed.
2. langdetect fallback: for queries that are entirely Latin script, used
   to distinguish English from romanized Hindi ("Hinglish written in
   Latin letters"), which script-counting alone cannot separate since
   both are 100% Latin characters.

Known limitations (stated explicitly, not hidden):
- langdetect is unreliable on short queries (a few words) and does not
  reliably detect code-mixing on its own -- that's why script-based
  detection runs first and takes priority when it finds mixed scripts.
- Romanized Hindi that closely resembles English words can be
  misclassified. A production system would use a model specifically
  trained on Indic code-mixed text (e.g. a fine-tuned LID model) instead
  of a general-purpose library like langdetect.
"""

import re

from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # makes langdetect's output deterministic run-to-run

DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
LATIN_PATTERN = re.compile(r"[a-zA-Z]")


def detect_language(query):
    """
    Returns a dict: {"language": "en" | "hi" | "hi-en-mixed", "method": str}
    """
    devanagari_chars = len(DEVANAGARI_PATTERN.findall(query))
    latin_chars = len(LATIN_PATTERN.findall(query))

    if devanagari_chars > 0 and latin_chars > 0:
        return {"language": "hi-en-mixed", "method": "script_detection"}

    if devanagari_chars > 0 and latin_chars == 0:
        return {"language": "hi", "method": "script_detection"}

    # Entirely Latin script -- could be English or romanized Hindi.
    # Fall back to langdetect, but don't trust it blindly on very short input.
    try:
        detected = detect(query)
        if detected == "hi":
            return {"language": "hi-romanized", "method": "langdetect"}
        elif detected == "en":
            return {"language": "en", "method": "langdetect"}
        else:
            # langdetect guessed some other language -- for this project's
            # scope (English/Hindi only), treat as English with low
            # confidence rather than claiming certainty about a language
            # we don't otherwise support.
            return {"language": "en", "method": f"langdetect_uncertain({detected})"}
    except LangDetectException:
        # langdetect fails outright on very short or ambiguous strings
        return {"language": "en", "method": "langdetect_failed_default_en"}


if __name__ == "__main__":
    test_queries = [
        "What are the cybersecurity requirements for banks?",
        "बैंकों के लिए साइबर सुरक्षा आवश्यकताएं क्या हैं?",
        "बैंकों के लिए cybersecurity requirements क्या हैं?",
        "bank ke liye KYC requirements kya hain",
    ]
    for q in test_queries:
        result = detect_language(q)
        print(f"{result} <- {q}")
