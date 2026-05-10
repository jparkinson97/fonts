"""Generate a synthetic pixel-art 'i' test crop matching what the GUI feeds
to ndarray_to_glyph. Body + already-present dot, like the user's screenshot.

Replace test/i_input.png with a real crop any time — the rest of the harness
just reads that file.
"""
import os
import numpy as np
import cv2

H, W = 36, 18  # roughly the proportions in the user's screenshot
img = np.full((H, W, 3), 255, dtype=np.uint8)

# Existing (ugly square) dot — pipeline should strip and redraw a circular one.
img[1:5, 7:12] = 0

# Body
img[9:34, 7:12]  = 0      # main vertical stem
img[9:13, 4:9]   = 0      # top-left flag (ear)
img[33:35, 4:14] = 0      # bottom serif

out = os.path.join(os.path.dirname(__file__), "i_input.png")
cv2.imwrite(out, img)
print(f"wrote {out}  shape={img.shape}")
