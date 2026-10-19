"""Aggregate metrics with explicit denominators and deterministic ordering."""

import re
from collections import Counter

from .adapters import Adapter


def language_metrics(texts: list[str], tokenizer: Adapter, context_tokens: int) -> dict:
    if not texts or context_tokens <= 0:
        raise ValueError("Nonempty text and positive context budget required")
    tokens = characters = words = 0
    unknown: int | None = 0
    fallback: int | None = 0
    histogram: Counter = Counter()
    for text in texts:
        if not text.strip():
            raise ValueError("Empty benchmark text")
        encoded = tokenizer.encode(text)
        if encoded.count <= 0:
            raise ValueError("Tokenizer produced no tokens for nonempty text")
        tokens += encoded.count
        characters += len(text)
        unknown = None if unknown is None or encoded.unknown is None else unknown + encoded.unknown
        fallback = (
            None
            if fallback is None or encoded.byte_fallback is None
            else fallback + encoded.byte_fallback
        )
        for word in re.findall(r"\S+", text):
            histogram[tokenizer.encode(word).count] += 1
            words += 1
    return {
        "tokens": tokens,
        "characters": characters,
        "words": words,
        "tokens_per_word": tokens / words,
        "characters_per_token": characters / tokens,
        "unknown_rate": unknown / tokens if unknown is not None else None,
        "byte_fallback_rate": fallback / tokens if fallback is not None else None,
        "isolated_word_pieces": {str(k): v for k, v in sorted(histogram.items())},
        "estimated_characters_per_context": context_tokens * characters / tokens,
    }


def compare(pairs: list[dict], tokenizer: Adapter, context_tokens: int) -> dict:
    tr = language_metrics([pair["tr"] for pair in pairs], tokenizer, context_tokens)
    en = language_metrics([pair["en"] for pair in pairs], tokenizer, context_tokens)
    return {"tr": tr, "en": en, "tr_en_token_ratio": tr["tokens"] / en["tokens"]}
