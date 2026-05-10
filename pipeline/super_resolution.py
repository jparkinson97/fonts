"""Multi-frame super-resolution for glyph crops.

Aligns multiple instances of the same glyph in an upsampled space and
median-stacks them. Sub-pixel offsets between samples become intensity
gradients in the average, recovering smoother edges than any single
low-res crop carries on its own.
"""
import cv2
import numpy as np

UPSCALE     = 4
MAX_SAMPLES = 10
ECC_ITERS   = 80
ECC_EPS     = 1e-4


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _ink_centroid(gray: np.ndarray) -> tuple[float, float]:
    if gray.dtype == np.uint8:
        g8 = gray
    elif np.issubdtype(gray.dtype, np.floating) and gray.max() <= 1.0 + 1e-6:
        g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    else:
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = gray.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def merge_samples(crops: list[np.ndarray], max_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Return a single BGR crop produced by aligning and median-stacking up to
    `max_samples` of the inputs. Falls back to the first crop when N < 2 or
    alignment fails.
    """
    if not crops:
        raise ValueError("merge_samples requires at least one crop")
    if len(crops) == 1:
        return crops[0]

    # Pick the sample with the median area as the reference — least likely to
    # be a touched-neighbour outlier or a clipped fragment.
    samples = list(crops[:max_samples])
    areas = [c.shape[0] * c.shape[1] for c in samples]
    ref_idx = int(np.argsort(areas)[len(areas) // 2])
    ref_bgr = samples[ref_idx]

    rh, rw = ref_bgr.shape[:2]
    big_size = (rw * UPSCALE, rh * UPSCALE)

    ref_gray_big = _resize_to(_to_gray(ref_bgr), big_size).astype(np.float32) / 255.0
    rcx, rcy = _ink_centroid(ref_gray_big)

    stack = [ref_gray_big]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)

    for i, crop in enumerate(samples):
        if i == ref_idx:
            continue
        gray = _to_gray(crop)
        big = _resize_to(gray, big_size).astype(np.float32) / 255.0

        cx, cy = _ink_centroid(big)
        # Coarse pre-translation aligns centroids; ECC then refines.
        warp = np.array([[1.0, 0.0, rcx - cx],
                         [0.0, 1.0, rcy - cy]], dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                ref_gray_big, big, warp,
                motionType=cv2.MOTION_EUCLIDEAN,
                criteria=criteria,
            )
            aligned = cv2.warpAffine(
                big, warp, big_size,
                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT, borderValue=1.0,
            )
        except cv2.error:
            continue

        stack.append(aligned)

    merged = np.median(np.stack(stack, axis=0), axis=0)
    merged = np.clip(merged * 255.0, 0, 255).astype(np.uint8)
    # Return BGR so it slots into the existing pipeline unchanged.
    return cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
