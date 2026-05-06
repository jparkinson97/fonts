import cv2
import numpy as np
from typing import Union


def _load(image: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image}")
        return img
    return image.copy()


def _find_sorted_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = img.shape[:2]
    min_area = (img_h * img_w) * 0.0001

    min_dim = max(5, int(min(img_h, img_w) * 0.01))

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        if w < min_dim or h < min_dim:
            continue
        if h >= img_h * 0.95:
            continue
        # discard anything touching the image border
        if x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return []

    median_h = float(np.median([h for _, _, _, h in boxes]))
    line_threshold = median_h * 0.6
    return sorted(boxes, key=lambda b: (round(b[1] / line_threshold), b[0]))


def segment_characters(image: Union[str, np.ndarray]) -> list[np.ndarray]:
    """
    Segment an image of text into individual character crops.

    Each returned array is a BGR np.ndarray compatible with classify_character.preprocess.

    Args:
        image: path to image file, or a BGR np.ndarray from cv2.imread.

    Returns:
        List of BGR character crops, sorted in reading order (top-to-bottom, left-to-right).
    """
    img = _load(image)
    boxes = _find_sorted_boxes(img)
    return [img[y : y + h, x : x + w] for x, y, w, h in boxes]


def segment_characters_with_boxes(
    image: Union[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Segment an image and return both the original image and bounding boxes.

    Args:
        image: path to image file, or a BGR np.ndarray from cv2.imread.

    Returns:
        (original_image, boxes) where boxes is a list of (x, y, w, h) tuples
        sorted in reading order.
    """
    img = _load(image)
    boxes = _find_sorted_boxes(img)
    return img, boxes


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python detect_character.py <image_path> [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "detected.png"

    img = _load(image_path)
    boxes = _find_sorted_boxes(img)
    print(f"Found {len(boxes)} character(s)")

    canvas = np.full_like(img, 255)
    for x, y, w, h in boxes:
        canvas[y : y + h, x : x + w] = img[y : y + h, x : x + w]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imwrite(out_path, canvas)
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
