"""OCR ensemble helpers."""

from collections import Counter

from pipeline.ocr.ocr_model import run_single_ocr
from pipeline.types import UploadedImage


def run_ensemble_ocr(images: list[UploadedImage]) -> str:
    """Run OCR for multiple images and vote character-by-character."""

    predictions = [run_single_ocr(image) for image in images]
    return vote_texts(predictions)


def vote_texts(texts: list[str]) -> str:
    """Return the most frequent character at each position."""

    if not texts:
        return "00가 0000"

    max_length = max(len(text) for text in texts)
    voted: list[str] = []
    for index in range(max_length):
        characters = [text[index] for text in texts if index < len(text)]
        if not characters:
            continue
        voted.append(Counter(characters).most_common(1)[0][0])

    return "".join(voted)
