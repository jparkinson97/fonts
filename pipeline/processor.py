#!/usr/bin/env python3
"""
Create font from image of text.
"""

import sys
import time
import cv2
import numpy as np
import pytesseract
from PIL import Image

from detect_character import segment_characters
from classify_character import preprocess

from create_woff2 import create_woff2


def _t(label: str, start: float):
    print(f"  {label}: {time.perf_counter() - start:.2f}s")


def _classify_single(image: np.ndarray) -> str:
    processed = preprocess(image)
    pil_image = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_image, config="--oem 3 --psm 10")
    return text.strip()


def build_font(image_path: str, output_path: str, font_name: str = "CustomFont") -> dict[str, np.ndarray]:
    t0 = time.perf_counter()

    t = time.perf_counter()
    crops = segment_characters(image_path)
    _t(f"segment_characters ({len(crops)} crops)", t)

    if not crops:
        raise ValueError(f"No characters detected in {image_path}")

    t = time.perf_counter()
    character_dict: dict[str, np.ndarray] = {}
    for crop in crops:
        char = _classify_single(crop)
        if len(char) == 1 and char not in character_dict:
            character_dict[char] = crop
    _t(f"classify ({len(character_dict)} unique chars)", t)

    if not character_dict:
        raise ValueError("No characters could be classified")

    t = time.perf_counter()
    create_woff2(character_dict, output_path, font_name)
    _t("create_woff2", t)

    _t("total", t0)
    return character_dict


def main():
    if len(sys.argv) < 2:
        print("Usage: python processor.py <image_path> [output.woff2] [FontName]")
        sys.exit(1)

    image_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.woff2"
    font_name = sys.argv[3] if len(sys.argv) > 3 else "CustomFont"

    try:
        print(f"Processing: {image_path}")
        character_dict = build_font(image_path, out_path, font_name)
        print(f"Recognised {len(character_dict)} unique character(s): {' '.join(sorted(character_dict))}")
        print(f"Saved font to {out_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
