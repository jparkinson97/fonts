#!/usr/bin/env python3
"""
Character Recognition using Tesseract OCR.
Covers full ASCII: letters, digits, punctuation, symbols.
"""

import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grayscale + denoise + Otsu threshold for cleaner OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def classify(image: np.ndarray):
    processed = preprocess(image)
    pil_image = Image.fromarray(processed)

    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(pil_image, config=config)
    return text.strip()

def recognize(image_path: str) -> str:
    """Load image, preprocess, and run Tesseract OCR."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    processed = preprocess(image)
    pil_image = Image.fromarray(processed)

    config = "--oem 3 --psm 6"

    text = pytesseract.image_to_string(pil_image, config=config)
    return text.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_character.py <image_path>")
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