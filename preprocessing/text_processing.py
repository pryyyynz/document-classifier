"""Text normalization, tokenization, and chunking utilities."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

try:
    import spacy  # type: ignore
except Exception:
    spacy = None  # type: ignore

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_text(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Lowercase, remove punctuation and stopwords.

    Args:
        text: Input text.
        stopwords: Optional iterable of stopwords to remove.

    Returns:
        Normalized text string.
    """
    lowered = text.lower()
    # Remove punctuation/symbols
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    cleaned = re.sub(r"_+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not stopwords:
        return cleaned

    stops = set(s.lower() for s in stopwords)
    tokens = [tok for tok in cleaned.split() if tok not in stops]
    return " ".join(tokens)


def get_spacy_tokenizer(model: str = "en_core_web_sm"):
    if spacy is None:
        raise ImportError(
            "spaCy is required for spaCy tokenizer: pip install spacy && python -m spacy download en_core_web_sm")
    # fast tokenizer-only
    nlp = spacy.load(model, disable=["ner", "parser", "tagger", "lemmatizer"])
    return nlp.tokenizer


def get_hf_tokenizer(model_name: str = "distilbert-base-uncased"):
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is required for HF tokenizer: pip install transformers")
    return AutoTokenizer.from_pretrained(model_name)


def chunk_tokens(tokens: Sequence[str], max_tokens: int, stride: int | None = None) -> List[List[str]]:
    """Split sequence of tokens into chunks with optional sliding window stride.

    If stride is None, non-overlapping chunks of size max_tokens are returned.
    If stride is provided (< max_tokens), overlapping windows are returned.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    if stride is None or stride >= max_tokens:
        return [list(tokens[i: i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

    chunks: List[List[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(list(tokens[start:end]))
        if end == len(tokens):
            break
        start += stride
    return chunks


def chunk_text_sliding_window(text: str, tokenizer, max_tokens: int, stride: int | None = None) -> List[str]:
    """Chunk text using provided tokenizer. Returns detokenized string chunks.

    For spaCy tokenizer: pass tokenizer that returns tokens with .text attributes.
    For HF tokenizer: pass a tokenizer with .tokenize method.
    """
    # Try to detect spaCy vs HF tokenizer
    token_texts: List[str]
    if hasattr(tokenizer, "tokenize"):
        token_texts = tokenizer.tokenize(text)
    else:
        token_texts = [t.text for t in tokenizer(text)]

    chunks = chunk_tokens(token_texts, max_tokens=max_tokens, stride=stride)
    return [" ".join(chunk) for chunk in chunks]


__all__ = [
    "normalize_text",
    "get_spacy_tokenizer",
    "get_hf_tokenizer",
    "chunk_tokens",
    "chunk_text_sliding_window",
]
