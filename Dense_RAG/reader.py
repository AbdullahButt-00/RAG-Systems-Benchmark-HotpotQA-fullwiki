"""
Ollama LLM reader: takes a question and retrieved passages, returns a concise answer.
"""

import requests
import time

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SEC

_SYSTEM = (
    "You are a precise question-answering assistant. "
    "Answer using only the provided passages. "
    "Give a short, factual answer — a phrase or a few words, never a full sentence unless required."
)


def _build_prompt(question: str, passages: list[dict]) -> str:
    context_lines = [
        f"[{i + 1}] ({p['title']}) {p['text']}" for i, p in enumerate(passages)
    ]
    context = "\n\n".join(context_lines)
    return f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"


def generate_answer(question: str, retrieved_passages: list[dict]) -> str:
    """
    Call local Ollama API with up to 3 retries on transient errors.

    Parameters
    ----------
    question : str
    retrieved_passages : list of dicts with keys title, text, (passage_id, score)

    Returns
    -------
    str — the model's answer
    """
    prompt = _build_prompt(question, retrieved_passages)

    for attempt in range(3):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{_SYSTEM}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                    },
                },
                timeout=OLLAMA_TIMEOUT_SEC,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.RequestException:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
