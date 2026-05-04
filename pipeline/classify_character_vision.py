#!/usr/bin/env python3
"""
Character Recognition using Apple Vision framework.
Runs on the Neural Engine / GPU — no external drivers required.
Drop-in replacement for classify_character.py.
"""

import sys
import cv2
import numpy as np
import Quartz
import Vision


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grayscale + denoise + Otsu threshold for cleaner OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _to_cgimage(gray: np.ndarray) -> Quartz.CGImageRef:
    h, w = gray.shape
    data = gray.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    colorspace = Quartz.CGColorSpaceCreateDeviceGray()
    return Quartz.CGImageCreate(
        w, h, 8, 8, w,
        colorspace,
        Quartz.kCGBitmapByteOrderDefault,
        provider, None, False,
        Quartz.kCGRenderingIntentDefault,
    )


_PAD = 32  # white padding added around each crop before passing to Vision


def classify(image: np.ndarray) -> str:
    # Grayscale only — no binary threshold so Vision keeps gradient information
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, _PAD, _PAD, _PAD, _PAD, cv2.BORDER_CONSTANT, value=255)
    cgimage = _to_cgimage(padded)

    results = []

    def handler(request, error):
        if error:
            return
        for obs in request.results():
            candidates = obs.topCandidates_(1)
            if candidates:
                results.append(candidates[0].string())

    request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)
    request.setMinimumTextHeight_(0.05)  # accept small text relative to image height
    req_handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, {})
    req_handler.performRequests_error_([request], None)

    return results[0].strip() if results else ""


def recognize(image_path: str) -> str:
    """Load image, preprocess, and run Apple Vision OCR."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return classify(image)


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_character_vision.py <image_path>")
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
