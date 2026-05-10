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

from super_resolution import merge_samples

UPM = 1000
CU2QU_MAX_ERR = 1.0
OVERSAMPLE = 4  # upscale the crop before vtracer so sub-pixel detail (terminals, varying stroke width) survives
_NORM_MARGIN = 0.12  # whitespace added around tight ink bbox, as fraction of the larger ink dimension

# Three vertical zones — ascender, x-height, descender. x-height zone is 1.6×
# the ascender (and descender) zone so lowercase round letters end up shorter
# than caps/ascenders, the way real fonts do.
ASCENDER  = 800              # baseline -> cap-height/ascender top
X_RATIO   = 2              # x-height zone / asc zone (== / desc zone)
X_HEIGHT  = int(ASCENDER * X_RATIO / (1.0 + X_RATIO))  # 492
DESCENDER = -(ASCENDER - X_HEIGHT)  # -308; symmetric with ascender zone
GLYPH_HEIGHT = ASCENDER  # kept for any legacy reference

DESCENDER_CHARS = set("gpqy")
ASCENDER_LOWER  = set("bdfhklt")
X_HEIGHT_CHARS  = set("acemnorsuvwxz")
# i/j are special-cased (tittle handling).


def _char_metrics(char: str) -> tuple[int, int]:
    """Return (target_height, baseline_offset) for a non-i/j character.

    The image is scaled so its height maps to `target_height` font units,
    and the bottom of the image lands at y=`baseline_offset` in font space.
    """
    if char in DESCENDER_CHARS:
        return X_HEIGHT - DESCENDER, DESCENDER          # 800, -308
    if char in ASCENDER_LOWER or (len(char) == 1 and (char.isupper() or char.isdigit())):
        return ASCENDER, 0                              # 800, 0
    if char in X_HEIGHT_CHARS:
        return X_HEIGHT, 0                              # 492, 0
    # Punctuation / fallback: sit on baseline at x-height.
    return X_HEIGHT, 0

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


def _svg_to_pen(svg: str, pen, img_h: int, img_w: int, scale: float, baseline: float = 0.0):
    """Parse vtracer SVG paths and replay them into a fontTools pen.

    `scale` is image-px → font-units. `baseline` is the font-space y at which
    the bottom of the image sits (negative for descender chars).
    """
    def pt(x, y):
        return (float(x) * scale, (float(img_h) - float(y)) * scale + baseline)

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


def _add_tittle_to_pen(pen, processed: np.ndarray, img_h: int, scale: float, baseline: float = 0.0):
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

    cx = cx_img * scale
    cy = (img_h - cy_img) * scale + baseline
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

    body_height: int | None = None  # set for i/j after CC isolation
    if char in ("i", "j"):
        # Strip any existing tittle / stray ink by keeping only the tallest
        # connected component (the body). Pad above with white so the
        # programmatic tittle fits inside the image bounds.
        _, _m = cv2.threshold(processed, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n_cc, _labels, _stats, _ = cv2.connectedComponentsWithStats(_m, connectivity=8)
        if n_cc > 1:
            heights = _stats[1:, cv2.CC_STAT_HEIGHT]
            body_idx = int(np.argmax(heights))
            body_label = 1 + body_idx
            body_height = int(_stats[body_label, cv2.CC_STAT_HEIGHT])
            body_mask = (_labels == body_label).astype(np.uint8) * 255
            processed = np.where(body_mask > 0, processed, 255).astype(np.uint8)
            _m = body_mask
        else:
            body_height = int(processed.shape[0])
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

    # Per-class scaling + baseline offset.
    if char == "i":
        # Body fills x-height; the top-pad pixels naturally extend above into
        # the ascender zone where the programmatic tittle lives.
        scale    = X_HEIGHT / max(1, body_height or h)
        baseline = 0.0
    elif char == "j":
        # Body spans descender to x-height (full asc-height worth of pixels).
        scale    = (X_HEIGHT - DESCENDER) / max(1, body_height or h)
        baseline = float(DESCENDER)
    else:
        target_h, baseline_int = _char_metrics(char or "")
        scale    = target_h / h
        baseline = float(baseline_int)

    params = {**VTRACER_PARAMS, "filter_speckle": 0, "length_threshold": 0.0} if raw else VTRACER_PARAMS
    svg  = vtracer.convert_raw_image_to_svg(img_bytes, img_format="png", **params)
    if os.environ.get("FONT_DEBUG"):
        print(f"[debug] char={char!r} processed={processed.shape} scale={scale:.3f} baseline={baseline:.1f}")
        with open("/tmp/font_debug.svg", "w") as _f:
            _f.write(svg)
        cv2.imwrite("/tmp/font_debug_processed.png", processed)

    tt_pen = TTGlyphPen(glyphSet=None)
    pen    = Cu2QuPen(tt_pen, max_err=CU2QU_MAX_ERR, reverse_direction=True)

    _svg_to_pen(svg, pen, h, w, scale, baseline)

    if char in ("i", "j"):
        _add_tittle_to_pen(pen, processed, h, scale, baseline)

    glyph         = tt_pen.glyph()
    advance_width = int(w * scale + 100)
    return glyph, advance_width


def _glyph_name(char: str) -> str:
    return UV2AGL.get(ord(char), f"uni{ord(char):04X}")


def create_woff2(
    char_arrays: dict[str, "np.ndarray | list[np.ndarray]"],
    output_path: str,
    font_name: str = "CustomFont",
    super_resolution: bool = False,
):
    """
    Build a woff2 font from a mapping of character -> BGR np.ndarray crop, or
    character -> list of crops when `super_resolution=True` (multiple samples
    are aligned and median-stacked into a single sharper crop).
    """
    glyph_order = [".notdef"]
    cmap = {}
    metrics = {}
    glyph_objects = {}

    for char, arr in char_arrays.items():
        if isinstance(arr, list):
            arr = merge_samples(arr) if super_resolution else arr[0]
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


def generate_preview_ttf(
    char: str,
    crop: "np.ndarray | list[np.ndarray]",
    family_name: str = "FontBuilderPreview",
    super_resolution: bool = False,
) -> bytes:
    """Generate an in-memory TTF for a single character, for Qt preview use.

    `crop` may be a single BGR ndarray or a list of crops. When
    `super_resolution=True` and a list is given, samples are aligned and
    median-stacked into a single sharper crop before glyph generation.
    """
    if isinstance(crop, list):
        crop = merge_samples(crop) if super_resolution else crop[0]
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
