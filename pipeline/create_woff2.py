import io
import re
import cv2
import numpy as np
import vtracer
from PIL import Image as PILImage
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.ttLib.tables._g_l_y_f import Glyph as EmptyGlyph
from fontTools.agl import UV2AGL

UPM = 1000
GLYPH_HEIGHT = 800
CU2QU_MAX_ERR = 1.0
OVERSAMPLE = 4  # upscale the crop before vtracer so sub-pixel detail (terminals, varying stroke width) survives
_NORM_MARGIN = 0.12  # whitespace added around tight ink bbox, as fraction of the larger ink dimension

ASCENDER = GLYPH_HEIGHT
DESCENDER = GLYPH_HEIGHT - UPM

# vtracer tuning — adjust these to trade smoothness vs. fidelity
VTRACER_PARAMS = dict(
    colormode      = "binary",
    filter_speckle = 4,      # discard blobs smaller than this (px²)
    corner_threshold = 60,   # angle (°) below which a point is a hard corner
    length_threshold = 4.0,  # ignore path segments shorter than this
    splice_threshold = 45,   # curve-fitting aggressiveness
    path_precision   = 3,    # decimal places in SVG output
)


def _normalize_ink(img: np.ndarray) -> np.ndarray:
    """Trim to tight ink bounding box then pad uniformly so every glyph fills the same proportion."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    pad = max(2, int(max(w, h) * _NORM_MARGIN))
    ih, iw = img.shape[:2]
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad); y2 = min(ih, y + h + pad)
    return img[y1:y2, x1:x2]


def _svg_to_pen(svg: str, pen, img_h: int, img_w: int):
    """Parse vtracer SVG paths and replay them into a fontTools pen.

    Applies a y-flip so image coords (origin top-left) become font coords
    (origin bottom-left). vtracer outer contours are CCW in image space;
    after flip they become CW — correct for TrueType.
    Cu2QuPen (reverse_direction=True) handles any remaining winding issues.
    """
    scale = GLYPH_HEIGHT / img_h

    def pt(x, y):
        return (float(x) * scale, float(img_h - y) * scale)

    # tokenise the path data from every <path d="..."> element
    for path_data in re.findall(r'd="([^"]+)"', svg):
        tokens = re.findall(r'[MCLZmclz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path_data)
        it = iter(tokens)
        for tok in it:
            if tok == 'M':
                x, y = float(next(it)), float(next(it))
                pen.moveTo(pt(x, y))
            elif tok == 'C':
                x1, y1 = float(next(it)), float(next(it))
                x2, y2 = float(next(it)), float(next(it))
                x,  y  = float(next(it)), float(next(it))
                pen.curveTo(pt(x1, y1), pt(x2, y2), pt(x, y))
            elif tok == 'L':
                x, y = float(next(it)), float(next(it))
                pen.lineTo(pt(x, y))
            elif tok == 'Z':
                pen.closePath()


def ndarray_to_glyph(image: np.ndarray):
    image    = _normalize_ink(image)
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    if OVERSAMPLE != 1:
        denoised = cv2.resize(
            denoised,
            (denoised.shape[1] * OVERSAMPLE, denoised.shape[0] * OVERSAMPLE),
            interpolation=cv2.INTER_CUBIC,
        )

    # vtracer expects a PNG; encode in memory to avoid temp files
    pil  = PILImage.fromarray(denoised)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    h, w = denoised.shape
    svg  = vtracer.convert_raw_image_to_svg(img_bytes, img_format="png", **VTRACER_PARAMS)

    tt_pen = TTGlyphPen(glyphSet=None)
    pen    = Cu2QuPen(tt_pen, max_err=CU2QU_MAX_ERR, reverse_direction=True)

    _svg_to_pen(svg, pen, h, w)

    glyph        = tt_pen.glyph()
    advance_width = int(w * (GLYPH_HEIGHT / h) + 100)
    return glyph, advance_width


def _glyph_name(char: str) -> str:
    return UV2AGL.get(ord(char), f"uni{ord(char):04X}")


def create_woff2(char_arrays: dict[str, np.ndarray], output_path: str, font_name: str = "CustomFont"):
    """
    Build a woff2 font from a mapping of character -> BGR np.ndarray crop.

    Args:
        char_arrays: dict mapping each character (e.g. 'a') to its image crop.
        output_path:  destination .woff2 file path.
        font_name:    font family name embedded in the name table.
    """
    glyph_order = [".notdef"]
    cmap = {}
    metrics = {}
    glyph_objects = {}

    for char, arr in char_arrays.items():
        name = _glyph_name(char)
        glyph, advance_width = ndarray_to_glyph(arr)
        glyph_order.append(name)
        cmap[ord(char)] = name
        metrics[name] = (advance_width, 0)
        glyph_objects[name] = glyph

    metrics[".notdef"] = (UPM // 2, 0)

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({".notdef": EmptyGlyph(), **glyph_objects})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)
    fb.setupNameTable({"familyName": font_name, "styleName": "Regular"})
    fb.setupOS2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        usWinAscent=ASCENDER,
        usWinDescent=abs(DESCENDER),
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=UPM)

    font = fb.font
    font.flavor = "woff2"
    font.save(output_path)
    return font


def generate_preview_ttf(char: str, crop: np.ndarray, family_name: str = "FontBuilderPreview") -> bytes:
    """Generate an in-memory TTF for a single character, for Qt preview use."""
    name = _glyph_name(char)
    glyph, advance_width = ndarray_to_glyph(crop)

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder([".notdef", name])
    fb.setupCharacterMap({ord(char): name})
    fb.setupGlyf({".notdef": EmptyGlyph(), name: glyph})
    fb.setupHorizontalMetrics({".notdef": (UPM // 2, 0), name: (advance_width, 0)})
    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)
    fb.setupNameTable({"familyName": family_name, "styleName": "Regular"})
    fb.setupOS2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        usWinAscent=ASCENDER,
        usWinDescent=abs(DESCENDER),
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=UPM)

    buf = io.BytesIO()
    fb.font.save(buf)
    return buf.getvalue()


def main():
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python create_woff2.py <crops_dir> [output.woff2] [FontName]")
        print("  crops_dir: directory of <char>.png files, e.g. a.png, b.png")
        sys.exit(1)

    crops_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output.woff2"
    font_name = sys.argv[3] if len(sys.argv) > 3 else "CustomFont"

    char_arrays = {}
    for fname in sorted(os.listdir(crops_dir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".png", ".jpg", ".jpeg") or len(stem) != 1:
            continue
        img = cv2.imread(os.path.join(crops_dir, fname))
        if img is not None:
            char_arrays[stem] = img

    if not char_arrays:
        print("No valid character images found (expected single-char filenames, e.g. a.png)")
        sys.exit(1)

    print(f"Building font for {len(char_arrays)} character(s): {' '.join(sorted(char_arrays))}")
    create_woff2(char_arrays, out_path, font_name)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
