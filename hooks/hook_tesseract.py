# Runtime hook — runs inside the frozen app before any user code.
# Points pytesseract at the bundled tesseract binary and tessdata.
import os
import sys
import pytesseract

base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))

_tess_exe = "tesseract.exe" if sys.platform == "win32" else "tesseract"
tesseract_bin = os.path.join(base, "tesseract", _tess_exe)
tessdata_dir  = os.path.join(base, "tesseract", "tessdata")

if os.path.isfile(tesseract_bin):
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin
    os.environ["TESSDATA_PREFIX"] = tessdata_dir
