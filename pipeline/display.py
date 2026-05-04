import os
import tempfile
import subprocess
from fontTools.ttLib import TTFont


def display_font(woff2_path: str, font_name: str = "PreviewFont"):
    """
    Open a browser tab showing all glyphs in the font rendered using @font-face.

    Args:
        woff2_path: path to the .woff2 file.
        font_name:  font-family name to use in CSS (must match what was used when building).
    """
    font = TTFont(woff2_path)
    cmap = font.getBestCmap()
    if not cmap:
        raise ValueError("Font has no cmap table — no characters to display")

    chars = sorted(chr(cp) for cp in cmap)
    abs_path = os.path.abspath(woff2_path)

    char_cells = "".join(
        f'<div class="cell"><div class="glyph">{c}</div><div class="label">{c} U+{ord(c):04X}</div></div>'
        for c in chars
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @font-face {{
    font-family: '{font_name}';
    src: url('file://{abs_path}') format('woff2');
  }}
  body {{ background: #f5f5f5; font-family: sans-serif; padding: 2rem; }}
  h1 {{ font-size: 1rem; color: #666; margin-bottom: 1.5rem; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .cell {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
    width: 80px;
  }}
  .glyph {{
    font-family: '{font_name}';
    font-size: 48px;
    line-height: 1;
    margin-bottom: 6px;
  }}
  .label {{ font-size: 10px; color: #999; }}
</style>
</head>
<body>
<h1>{os.path.basename(woff2_path)} — {len(chars)} glyph(s)</h1>
<div class="grid">{char_cells}</div>
</body>
</html>"""

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(html)
        tmp = f.name

    subprocess.run(["open", tmp])


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python display.py <font.woff2> [FontName]")
        sys.exit(1)

    woff2_path = sys.argv[1]
    font_name = sys.argv[2] if len(sys.argv) > 2 else "PreviewFont"
    display_font(woff2_path, font_name)


if __name__ == "__main__":
    main()
