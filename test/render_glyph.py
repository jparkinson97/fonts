"""Run a single crop through the i/j glyph pipeline and render the resulting
glyph to a PNG so we can eyeball whether the tittle landed correctly.

Usage:  python test/render_glyph.py [char]
        char defaults to 'i'. Reads test/<char>_input.png, writes
        test/<char>_output.png.
"""
import os
import sys
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))

from create_woff2 import generate_preview_ttf  # noqa: E402

CHAR = sys.argv[1] if len(sys.argv) > 1 else "i"
in_path  = os.path.join(HERE, f"{CHAR}_input.png")
out_path = os.path.join(HERE, f"{CHAR}_output.png")
ttf_path = os.path.join(HERE, f"{CHAR}_output.ttf")

if not os.path.exists(in_path):
    print(f"missing {in_path}; run make_input.py first")
    sys.exit(1)

crop = cv2.imread(in_path)
ttf  = generate_preview_ttf(CHAR, crop, family_name="TestFont")
with open(ttf_path, "wb") as f:
    f.write(ttf)

# Render the glyph to a PNG via PIL/freetype
SIZE = 400
canvas = Image.new("RGB", (SIZE, SIZE), "white")
draw   = ImageDraw.Draw(canvas)
font   = ImageFont.truetype(io.BytesIO(ttf), size=300)
# Center it visually
bbox = draw.textbbox((0, 0), CHAR, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
draw.text(((SIZE - tw)//2 - bbox[0], (SIZE - th)//2 - bbox[1]),
          CHAR, fill="black", font=font)

# Side-by-side: input crop (scaled up) + rendered glyph
ch, cw = crop.shape[:2]
scale = SIZE // max(ch, cw)
crop_big = cv2.resize(crop, (cw * scale, ch * scale), interpolation=cv2.INTER_NEAREST)
crop_big_pil = Image.fromarray(cv2.cvtColor(crop_big, cv2.COLOR_BGR2RGB))

side = Image.new("RGB", (SIZE + crop_big_pil.width + 20, SIZE), "lightgray")
side.paste(crop_big_pil, (0, (SIZE - crop_big_pil.height)//2))
side.paste(canvas, (crop_big_pil.width + 20, 0))
side.save(out_path)
print(f"wrote {out_path}  ({CHAR})")
