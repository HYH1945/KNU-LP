"""OCR ensemble helpers."""

from __future__ import annotations

from collections import Counter

from pipeline.ocr.ocr_model import run_batch_ocr
from pipeline.types import UploadedImage


def run_ensemble_ocr(images: list[UploadedImage]) -> str:
    """Run OCR for multiple images and vote character-by-character."""

    predictions = run_batch_ocr(images)
    return vote_texts(predictions)


def vote_texts(texts: list[str]) -> str:
    """Return the most frequent character at each position."""

    texts = [text for text in texts if text and text != "UNKNOWN"]
    if not texts:
        return "UNKNOWN"

    max_length = max(len(text) for text in texts)
    voted: list[str] = []
    for index in range(max_length):
        characters = [text[index] for text in texts if index < len(text)]
        if not characters:
            continue
        voted.append(Counter(characters).most_common(1)[0][0])

    return "".join(voted) or "UNKNOWN"
