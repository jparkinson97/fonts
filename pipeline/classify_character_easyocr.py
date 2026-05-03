#!/usr/bin/env python3
"""
Character Recognition using EasyOCR.
Covers full ASCII: letters, digits, punctuation, symbols.
Drop-in replacement for classify_character.py.
"""

import sys
import cv2
import numpy as np
import easyocr

# Initialised once at module load; model download happens on first use.
_reader = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grayscale + denoise + Otsu threshold for cleaner OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def classify(image: np.ndarray) -> str:
    processed = preprocess(image)
    results = _get_reader().readtext(processed, detail=0, paragraph=False)
    return results[0].strip() if results else ""


def recognize(image_path: str) -> str:
    """Load image, preprocess, and run EasyOCR."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return classify(image)


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_character_easyocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        print(f"Processing: {image_path}")
        text = recognize(image_path)
        print("Recognized text:")
        print(text)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
