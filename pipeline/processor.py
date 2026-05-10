#!/usr/bin/env python3
"""
Create font from image of text.
"""

import sys
import string
import time
import cv2
import numpy as np
import pytesseract
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from detect_character import segment_characters_with_boxes, attach_dot_above
from classify_character_easyocr import preprocess

from create_woff2 import create_woff2

_WHITELIST = string.ascii_letters + string.digits + r""".,!?;:()-"""
_OCR_CONFIG = (
    "--oem 3 --psm 10 "
    f"-c tessedit_char_whitelist={_WHITELIST}"
)
_TARGET_HEIGHT = 96  # px; Tesseract accuracy improves above ~64px
_PAD_RATIO = 0.20    # white border as fraction of the larger dimension
_BOX_EXPAND = 3      # px to add on each side of detected boxes before cropping


def _t(label: str, start: float):
    print(f"  {label}: {time.perf_counter() - start:.2f}s")


def _prepare_for_ocr(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape[:2]

    # uniform padding on all sides — fraction of the larger dimension
    pad = max(4, int(max(h, w) * _PAD_RATIO))
    padded = cv2.copyMakeBorder(binary, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=255)

    # upscale so height reaches _TARGET_HEIGHT (both dims, same factor)
    ph, pw = padded.shape[:2]
    if ph < _TARGET_HEIGHT:
        scale = _TARGET_HEIGHT / ph
        padded = cv2.resize(padded,
                            (int(pw * scale), int(ph * scale)),
                            interpolation=cv2.INTER_CUBIC)

    return padded


def _classify_single(image: np.ndarray) -> str:
    processed = preprocess(image)
    ready = _prepare_for_ocr(processed)
    pil_image = Image.fromarray(ready)
    text = pytesseract.image_to_string(pil_image, config=_OCR_CONFIG)
    return text.strip()


def build_char_dict(
    image_path: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, list[tuple[int, int, int, int]]],
    np.ndarray,
]:
    """Returns (char_crops, char_boxes, original_image) for all detected characters.

    progress_callback(completed, total) is called once after segmentation
    (with completed=0) and then after each crop is classified.
    """
    t0 = time.perf_counter()

    t = time.perf_counter()
    orig_img, boxes = segment_characters_with_boxes(image_path)
    img_h, img_w = orig_img.shape[:2]
    e = _BOX_EXPAND
    boxes = [
        (max(0, x - e), max(0, y - e),
         min(img_w - max(0, x - e), w + 2 * e),
         min(img_h - max(0, y - e), h + 2 * e))
        for x, y, w, h in boxes
    ]
    crops = [orig_img[y : y + h, x : x + w] for x, y, w, h in boxes]
    _t(f"segment_characters ({len(crops)} crops)", t)

    if not crops:
        raise ValueError(f"No characters detected in {image_path}")

    if progress_callback:
        progress_callback(0, len(crops))

    t = time.perf_counter()
    chars = [''] * len(crops)
    completed = 0
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_classify_single, crop): i for i, crop in enumerate(crops)}
        for future in as_completed(futures):
            chars[futures[future]] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, len(crops))

    # Median body height for sizing the dot-search window
    median_h = float(np.median([h for _, _, _, h in boxes])) if boxes else 0.0

    char_dict: dict[str, list[np.ndarray]] = {}
    box_dict:  dict[str, list[tuple[int, int, int, int]]] = {}
    for (crop, box), char in zip(zip(crops, boxes), chars):
        if len(char) != 1:
            continue
        if char in ("i", "j"):
            new_box = attach_dot_above(box, orig_img, median_h)
            if new_box != box:
                nx, ny, nw, nh = new_box
                crop = orig_img[ny : ny + nh, nx : nx + nw]
                box = new_box
        char_dict.setdefault(char, []).append(crop)
        box_dict.setdefault(char, []).append(box)
    _t(f"classify ({len(char_dict)} unique chars)", t)

    if not char_dict:
        raise ValueError("No characters could be classified")

    _t("total", t0)
    return char_dict, box_dict, orig_img


def build_font(image_path: str, output_path: str, font_name: str = "CustomFont") -> dict[str, np.ndarray]:
    character_dict, _, _ = build_char_dict(image_path)
    t = time.perf_counter()
    create_woff2(character_dict, output_path, font_name)
    _t("create_woff2", t)
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
