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


def attach_dot_above(
    box: tuple[int, int, int, int],
    img: np.ndarray,
    median_h: float | None = None,
) -> tuple[int, int, int, int]:
    """Given an i/j body box and the original image, look directly above in
    grayscale with a local Otsu threshold to find the tittle, and return an
    expanded box that includes it. If no dot is found, returns the box unchanged.

    Run AFTER classification — when we already know the char is 'i' or 'j' —
    so we don't have to guess from geometry.
    """
    x, y, w, h = box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img_h, img_w = gray.shape

    # Search region above the body — sized relative to the body's own height
    # so it works regardless of overall page scale.
    ref_h = float(median_h) if median_h else float(h)
    search_h = max(8, int(ref_h * 0.8))
    sy1 = max(0, y - search_h)
    sy2 = y
    slack = max(3, w)
    sx1 = max(0, x - slack)
    sx2 = min(img_w, x + w + slack)
    if sy1 >= sy2 or sx1 >= sx2:
        return box

    region = gray[sy1:sy2, sx1:sx2]
    _, ink = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(ink)
    if coords is None:
        return box

    dx, dy, dw, dh = cv2.boundingRect(coords)

    if dw * dh < 4:
        return box
    if dh > ref_h * 0.6:        # too tall to be a tittle
        return box
    if dy <= 0:                 # touches top of search window — bleed from line above
        return box

    abs_x = sx1 + dx; abs_y = sy1 + dy
    nx  = min(x, abs_x);          ny  = min(y, abs_y)
    nx2 = max(x + w, abs_x + dw); ny2 = max(y + h, abs_y + dh)
    return (nx, ny, nx2 - nx, ny2 - ny)


def _find_sorted_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_h, img_w = img.shape[:2]
    min_area = (img_h * img_w) * 0.0001
    min_dim  = max(5, int(min(img_h, img_w) * 0.01))

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        if w < min_dim or h < min_dim:
            continue
        if h >= img_h * 0.95:
            continue
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
