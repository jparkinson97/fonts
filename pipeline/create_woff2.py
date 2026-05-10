import io
import os
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

    # vtracer emits one <path> per contour, each with its own
    # transform="translate(tx, ty)". Honour it when replaying.
    path_re = re.compile(
        r'<path\b([^>]*)>',
        re.DOTALL,
    )
    for attrs in path_re.findall(svg):
        d_match = re.search(r'd="([^"]+)"', attrs)
        if not d_match:
            continue
        path_data = d_match.group(1)
        tx = ty = 0.0
        t_match = re.search(
            r'transform="translate\(\s*([-+]?\d*\.?\d+)\s*[,\s]\s*([-+]?\d*\.?\d+)\s*\)"',
            attrs,
        )
        if t_match:
            tx = float(t_match.group(1))
            ty = float(t_match.group(2))

        def lpt(x, y, _tx=tx, _ty=ty):
            return pt(x + _tx, y + _ty)

        tokens = re.findall(r'[MCLZmclz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', path_data)
        it = iter(tokens)
        for tok in it:
            if tok == 'M':
                x, y = float(next(it)), float(next(it))
                pen.moveTo(lpt(x, y))
            elif tok == 'C':
                x1, y1 = float(next(it)), float(next(it))
                x2, y2 = float(next(it)), float(next(it))
                x,  y  = float(next(it)), float(next(it))
                pen.curveTo(lpt(x1, y1), lpt(x2, y2), lpt(x, y))
            elif tok == 'L':
                x, y = float(next(it)), float(next(it))
                pen.lineTo(lpt(x, y))
            elif tok == 'Z':
                pen.closePath()


def _add_tittle_to_pen(pen, processed: np.ndarray, img_h: int):
    """Append a circular tittle as a separate contour to the glyph pen.

    Diameter = thinnest stem width (min over rows of the longest ink run in
    that row). Centered on the topmost ink pixel's column, placed
    `stem_width` pixels above the topmost ink pixel (image-space).
    """
    _, mask = cv2.threshold(processed, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) == 0:
        return
    bin_rows = mask > 0
    row_stems = []
    for y in range(bin_rows.shape[0]):
        row = bin_rows[y].astype(np.int8)
        if not row.any():
            continue
        diffs = np.diff(np.concatenate(([0], row, [0])))
        runs = np.where(diffs == -1)[0] - np.where(diffs == 1)[0]
        row_stems.append(int(runs.max()))
    if not row_stems:
        return
    min_run = int(np.median(row_stems))
    if min_run < 2:
        return

    ys, xs = np.where(bin_rows)
    top_y = int(ys.min())
    cx_img = float(np.median(xs[ys == top_y]))
    radius_img = min_run / 2.0
    gap_img    = float(min_run)
    cy_img = top_y - gap_img - radius_img

    scale = GLYPH_HEIGHT / img_h
    cx = cx_img * scale
    cy = (img_h - cy_img) * scale
    r  = radius_img * scale
    k  = 0.5522847498307936 * r  # cubic-bezier circle handle length

    # CW in font space (Y-up). The body contours are written via pt() which
    # flips Y (reversing winding); Cu2QuPen(reverse_direction=True) then
    # reverses again. We skip pt(), so we pre-apply the opposite winding
    # to end up matching the body after Cu2QuPen reverses us.
    if os.environ.get("FONT_DEBUG"):
        print(f"[tittle] img_h={img_h} top_y={top_y} stem={min_run} "
              f"cx_img={cx_img:.1f} cy_img={cy_img:.1f} radius_img={radius_img:.1f} "
              f"-> cx={cx:.1f} cy={cy:.1f} r={r:.1f}")
    pen.moveTo((cx + r, cy))
    pen.curveTo((cx + r, cy - k), (cx + k, cy - r), (cx, cy - r))
    pen.curveTo((cx - k, cy - r), (cx - r, cy - k), (cx - r, cy))
    pen.curveTo((cx - r, cy + k), (cx - k, cy + r), (cx, cy + r))
    pen.curveTo((cx + k, cy + r), (cx + r, cy + k), (cx + r, cy))
    pen.closePath()


def ndarray_to_glyph(image: np.ndarray, raw: bool = False, char: str | None = None):
    """If raw=True, skip _normalize_ink, denoise, and speckle filtering —
    preserves small features like i/j tittles."""
    if not raw:
        image = _normalize_ink(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    processed = gray if raw else cv2.fastNlMeansDenoising(gray, h=10)

    if OVERSAMPLE != 1:
        processed = cv2.resize(
            processed,
            (processed.shape[1] * OVERSAMPLE, processed.shape[0] * OVERSAMPLE),
            interpolation=cv2.INTER_CUBIC,
        )

    if char in ("i", "j"):
        # Strip any existing tittle / stray ink — we redraw a clean circular
        # dot below — by keeping only the tallest connected component (the
        # body). Then top-pad with white so the new tittle fits inside the
        # image bounds (and thus inside ASCENDER in font space).
        _, _m = cv2.threshold(processed, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n_cc, _labels, _stats, _ = cv2.connectedComponentsWithStats(_m, connectivity=8)
        if n_cc > 1:
            heights = _stats[1:, cv2.CC_STAT_HEIGHT]
            body_label = 1 + int(np.argmax(heights))
            body_mask = (_labels == body_label).astype(np.uint8) * 255
            # Repaint processed: white everywhere except where body ink was.
            processed = np.where(body_mask > 0, processed, 255).astype(np.uint8)
            _m = body_mask
        if cv2.countNonZero(_m):
            row_stems = []
            for _y in range(_m.shape[0]):
                _row = (_m[_y] > 0).astype(np.int8)
                if not _row.any():
                    continue
                _d = np.diff(np.concatenate(([0], _row, [0])))
                _runs = np.where(_d == -1)[0] - np.where(_d == 1)[0]
                row_stems.append(int(_runs.max()))
            stem = int(np.median(row_stems)) if row_stems else 0
            if stem >= 2:
                pad = int(stem * 2.5)  # gap (=stem) + diameter (=stem) + slack
                processed = cv2.copyMakeBorder(processed, pad, 0, 0, 0,
                                               cv2.BORDER_CONSTANT, value=255)

    pil  = PILImage.fromarray(processed)
    buf  = io.BytesIO()
    pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    h, w = processed.shape
    params = {**VTRACER_PARAMS, "filter_speckle": 0, "length_threshold": 0.0} if raw else VTRACER_PARAMS
    svg  = vtracer.convert_raw_image_to_svg(img_bytes, img_format="png", **params)
    if os.environ.get("FONT_DEBUG"):
        print(f"[debug] processed shape={processed.shape}  scale={GLYPH_HEIGHT/h:.3f}")
        with open("/tmp/font_debug.svg", "w") as _f:
            _f.write(svg)
        cv2.imwrite("/tmp/font_debug_processed.png", processed)
        print("[debug] wrote /tmp/font_debug.svg and /tmp/font_debug_processed.png")

    tt_pen = TTGlyphPen(glyphSet=None)
    pen    = Cu2QuPen(tt_pen, max_err=CU2QU_MAX_ERR, reverse_direction=True)

    _svg_to_pen(svg, pen, h, w)

    if char in ("i", "j"):
        _add_tittle_to_pen(pen, processed, h)

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
        glyph, advance_width = ndarray_to_glyph(arr, raw=char in ("i", "j"), char=char)
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
    glyph, advance_width = ndarray_to_glyph(crop, raw=char in ("i", "j"), char=char)

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
