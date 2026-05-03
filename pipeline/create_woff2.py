import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._g_l_y_f import Glyph as EmptyGlyph
from fontTools.agl import UV2AGL

UPM = 1000
GLYPH_HEIGHT = 800

ASCENDER = GLYPH_HEIGHT
DESCENDER = GLYPH_HEIGHT - UPM


def ndarray_to_glyph(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    scale = GLYPH_HEIGHT / h

    def to_em(x, y):
        # y-flip converts image coords (origin top-left) to font coords (origin bottom-left)
        # After flip, OpenCV outer contours (CCW in image space) become CW — correct for TrueType
        return (float(x) * scale, float(h - y) * scale)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tt_pen = TTGlyphPen(glyphSet=None)
    for cnt in contours:
        epsilon = max(0.5, 0.01 * cv2.arcLength(cnt, closed=True))
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        tt_pen.moveTo(to_em(*approx[0]))
        for pt in approx[1:]:
            tt_pen.lineTo(to_em(*pt))
        tt_pen.closePath()

    glyph = tt_pen.glyph()
    advance_width = int(w * scale + 100)
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
