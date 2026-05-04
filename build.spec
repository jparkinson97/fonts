# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Tesseract bundling ────────────────────────────────────────────────────────
# macOS: read paths from environment or fall back to Homebrew defaults.
# Windows: set TESSERACT_PREFIX to your Tesseract install dir before building.
if sys.platform == "darwin":
    _tess_prefix   = os.environ.get("TESSERACT_PREFIX", "/usr/local/opt/tesseract")
    _lept_prefix   = os.environ.get("LEPTONICA_PREFIX", "/usr/local/opt/leptonica")
    _tess_bin      = os.path.join(_tess_prefix, "bin", "tesseract")
    _tess_binaries = [
        (_tess_bin,                                                        "tesseract"),
        (os.path.join(_tess_prefix, "lib", "libtesseract.5.dylib"),       "tesseract"),
        (os.path.join(_lept_prefix, "lib", "libleptonica.6.dylib"),       "tesseract"),
    ]
    _tess_data = [(os.path.join(_tess_prefix, "share", "tessdata", "eng.traineddata"),
                   os.path.join("tesseract", "tessdata"))]
elif sys.platform == "win32":
    _tess_prefix   = os.environ.get("TESSERACT_PREFIX", r"C:\Program Files\Tesseract-OCR")
    _tess_binaries = [
        (os.path.join(_tess_prefix, "tesseract.exe"), "tesseract"),
    ]
    _tess_data = [(os.path.join(_tess_prefix, "tessdata", "eng.traineddata"),
                   os.path.join("tesseract", "tessdata"))]
else:
    _tess_binaries = []
    _tess_data     = []

hidden_imports = [
    # cv2
    "cv2",
    # PIL
    "PIL._imaging",
    "PIL.Image",
    "PIL.ImageOps",
    # fontTools
    *collect_submodules("fontTools"),
    # easyocr
    *collect_submodules("easyocr"),
    # pytesseract
    "pytesseract",
    # PyQt6
    *collect_submodules("PyQt6"),
    # pipeline modules (added to pathex below, imported as top-level)
    "detect_character",
    "classify_character",
    "classify_character_easyocr",
    "classify_character_vision",
    "create_woff2",
    "processor",
]

datas = [
    *collect_data_files("easyocr"),
    *collect_data_files("fontTools"),
    *_tess_data,
]

a = Analysis(
    ["app.py"],
    pathex=["pipeline"],        # pipeline modules importable as top-level
    binaries=_tess_binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=["hooks/hook_tesseract.py"],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FontBuilder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FontBuilder",
)

# macOS: wrap in a .app bundle
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Font Builder.app",
        bundle_identifier="com.fontbuilder.app",
        info_plist={
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,  # allow dark mode
            "CFBundleShortVersionString": "1.0.0",
        },
    )
