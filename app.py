#!/usr/bin/env python3
import sys
import os

import cv2
import numpy as np
from PIL import Image as PILImage

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QGridLayout,
    QFrame, QMessageBox, QSizePolicy, QDialog, QLineEdit,
)
from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))
from processor import build_char_dict
from create_woff2 import create_woff2

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#141414"
SURFACE = "#1f1f1f"
CARD    = "#2a2a2a"
BORDER  = "#333333"
ACCENT  = "#4f8ef7"
ACCENT_H= "#3a7ae0"
SEL     = "#1a3a6e"
SEL_B   = "#4f8ef7"
TEXT    = "#f0f0f0"
MUTED   = "#7a7a7a"
ERROR   = "#e05a5a"

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: -apple-system, 'Segoe UI', sans-serif; }}
QScrollArea  {{ border: none; background: {BG}; }}
QScrollBar:vertical {{
    background: {SURFACE}; width: 8px; border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 32px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QPushButton {{
    background: {CARD}; color: {MUTED}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 8px 20px; font-size: 13px;
}}
QPushButton:hover  {{ background: {BORDER}; color: {TEXT}; }}
QPushButton#browse {{ background: {ACCENT}; color: white; border: none; }}
QPushButton#browse:hover {{ background: {ACCENT_H}; }}
QPushButton#save   {{ background: {ACCENT}; color: white; border: none; }}
QPushButton#save:hover   {{ background: {ACCENT_H}; }}
QPushButton#save:disabled {{
    background: {CARD}; color: {MUTED}; border: 1px solid {BORDER};
}}
"""


# ── helpers ───────────────────────────────────────────────────────────────────
def crop_to_pixmap(crop: np.ndarray, size: int) -> QPixmap:
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = PILImage.fromarray(img_rgb).convert("RGBA")
    pil.thumbnail((size, size), PILImage.LANCZOS)
    qimg = QImage(pil.tobytes(), pil.width, pil.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg).scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


# ── worker thread ─────────────────────────────────────────────────────────────
class Worker(QThread):
    done  = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.done.emit(build_char_dict(self.path))
        except Exception as e:
            self.error.emit(str(e))


# ── instance picker dialog ────────────────────────────────────────────────────
class PickerDialog(QDialog):
    """Shows all detected instances of a character.

    After exec():
      .chosen      — index of the selected instance
      .reassign_to — non-empty str if user wants to move chosen instance to a different char
    """

    THUMB = 80
    COLS  = 4

    def __init__(self, char: str, crops: list[np.ndarray], current: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Edit  "{char}"')
        self.setModal(True)
        self.setStyleSheet(STYLESHEET + f"QDialog {{ background: {SURFACE}; }}")
        self.chosen      = current
        self.reassign_to = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(14)

        # ── instance grid ──
        lbl = QLabel(f"{len(crops)} instance(s) — click one to select it")
        lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        outer.addWidget(lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(min(len(crops) // self.COLS + 1, 4) * (self.THUMB + 40) + 20)
        inner = QWidget()
        grid  = QGridLayout(inner)
        grid.setSpacing(10)
        grid.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._btns: list[QPushButton] = []
        for i, crop in enumerate(crops):
            btn = QPushButton()
            btn.setFixedSize(self.THUMB + 20, self.THUMB + 20)
            btn.setIcon(QIcon(crop_to_pixmap(crop, self.THUMB)))
            btn.setIconSize(QSize(self.THUMB, self.THUMB))
            btn.setCheckable(True)
            btn.setChecked(i == current)
            btn.setStyleSheet(self._btn_style(i == current))
            btn.clicked.connect(lambda _, idx=i: self._pick(idx))
            grid.addWidget(btn, i // self.COLS, i % self.COLS)
            self._btns.append(btn)

        # ── divider ──
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {BORDER};")
        outer.addWidget(div)

        # ── reassign row ──
        reassign_lbl = QLabel("Reassign selected instance to character:")
        reassign_lbl.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        outer.addWidget(reassign_lbl)

        row = QHBoxLayout()
        self._char_input = QLineEdit()
        self._char_input.setMaxLength(1)
        self._char_input.setPlaceholderText("type a character…")
        self._char_input.setFixedHeight(34)
        self._char_input.setStyleSheet(f"""
            QLineEdit {{
                background: {CARD}; color: {TEXT}; border: 1px solid {BORDER};
                border-radius: 6px; padding: 0 10px; font-size: 18px;
            }}
            QLineEdit:focus {{ border-color: {ACCENT}; }}
        """)
        row.addWidget(self._char_input, stretch=1)

        reassign_btn = QPushButton("Reassign")
        reassign_btn.setFixedHeight(34)
        reassign_btn.setObjectName("browse")
        reassign_btn.clicked.connect(self._do_reassign)
        row.addWidget(reassign_btn)
        outer.addLayout(row)

        # ── cancel ──
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        outer.addWidget(cancel, alignment=Qt.AlignmentFlag.AlignRight)

    def _btn_style(self, selected: bool) -> str:
        border = SEL_B if selected else BORDER
        bg     = SEL   if selected else CARD
        return f"""
            QPushButton {{
                background: {bg}; border: 2px solid {border}; border-radius: 8px;
            }}
            QPushButton:hover {{ border-color: {ACCENT}; }}
        """

    def _pick(self, idx: int):
        self.chosen = idx
        for i, btn in enumerate(self._btns):
            btn.setStyleSheet(self._btn_style(i == idx))
            btn.setChecked(i == idx)
        self.accept()

    def _do_reassign(self):
        char = self._char_input.text().strip()
        if not char:
            return
        self.reassign_to = char
        self.accept()


# ── character card ────────────────────────────────────────────────────────────
class CharCard(QFrame):
    clicked = pyqtSignal(str)
    THUMB = 72

    def __init__(self, char: str, crop: np.ndarray, count: int, parent=None):
        super().__init__(parent)
        self.char = char
        self._apply_style(selected=False)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(108, 136)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._thumb_lbl = QLabel()
        self._thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._thumb_lbl)
        self.set_crop(crop)

        char_lbl = QLabel(char)
        char_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        char_lbl.setStyleSheet(f"color: {TEXT}; font-size: 18px; font-weight: 600; background: transparent; border: none;")
        layout.addWidget(char_lbl)

        cp_lbl = QLabel(f"U+{ord(char):04X}")
        cp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cp_lbl.setStyleSheet(f"color: {MUTED}; font-size: 10px; background: transparent; border: none;")
        layout.addWidget(cp_lbl)

        if count > 1:
            cnt_lbl = QLabel(f"{count} instances")
            cnt_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cnt_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 9px; background: transparent; border: none;")
            layout.addWidget(cnt_lbl)

    def set_crop(self, crop: np.ndarray):
        self._thumb_lbl.setPixmap(crop_to_pixmap(crop, self.THUMB))

    def _apply_style(self, selected: bool):
        border = SEL_B if selected else BORDER
        bg     = SEL   if selected else CARD
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg}; border: 1px solid {border}; border-radius: 10px;
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.char)


# ── main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    COLS = 6

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Font Builder")
        self.resize(780, 600)
        self.setMinimumSize(520, 400)
        self.setStyleSheet(STYLESHEET)

        self._all_crops: dict[str, list[np.ndarray]] = {}
        self._selected:  dict[str, int] = {}           # char → chosen index
        self._cards:     dict[str, CharCard] = {}
        self._worker: Worker | None = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        toolbar = QWidget()
        toolbar.setStyleSheet(f"background: {SURFACE}; border-bottom: 1px solid {BORDER};")
        toolbar.setFixedHeight(56)
        hbox = QHBoxLayout(toolbar)
        hbox.setContentsMargins(16, 0, 16, 0)
        hbox.setSpacing(12)

        browse_btn = QPushButton("Browse image…")
        browse_btn.setObjectName("browse")
        browse_btn.setFixedHeight(34)
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.clicked.connect(self._browse)
        hbox.addWidget(browse_btn)

        self._status = QLabel("No file selected")
        self._status.setStyleSheet(f"color: {MUTED}; font-size: 13px;")
        hbox.addWidget(self._status, stretch=1)

        self._save_btn = QPushButton("Save font…")
        self._save_btn.setObjectName("save")
        self._save_btn.setFixedHeight(34)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save)
        hbox.addWidget(self._save_btn)

        vbox.addWidget(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background: {BG};")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(20, 20, 20, 20)
        self._grid_layout.setSpacing(10)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._empty_label = QLabel("Open an image to extract characters")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {MUTED}; font-size: 14px;")
        self._grid_layout.addWidget(self._empty_label, 0, 0, 1, 6)

        scroll.setWidget(self._grid_widget)
        vbox.addWidget(scroll, stretch=1)

    # ── pipeline ──────────────────────────────────────────────────────────────
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image",
            filter="Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self._status.setText(f"Processing {os.path.basename(path)}…")
        self._save_btn.setEnabled(False)
        self._clear_grid()

        self._worker = Worker(path)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, all_crops: dict[str, list[np.ndarray]]):
        self._all_crops = all_crops
        self._selected  = {char: 0 for char in all_crops}
        self._status.setText(f"{len(all_crops)} character(s) found — click any to change instance")
        self._save_btn.setEnabled(True)
        self._populate_grid()

    def _on_error(self, msg: str):
        self._status.setStyleSheet(f"color: {ERROR}; font-size: 13px;")
        self._status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Processing failed", msg)

    # ── grid ──────────────────────────────────────────────────────────────────
    def _clear_grid(self):
        self._cards.clear()
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_grid(self):
        self._clear_grid()
        cols = max(1, (self.width() - 60) // 118)
        for i, char in enumerate(sorted(self._all_crops)):
            crops = self._all_crops[char]
            card  = CharCard(char, crops[self._selected[char]], len(crops))
            card.clicked.connect(self._on_card_clicked)
            self._grid_layout.addWidget(card, i // cols, i % cols)
            self._cards[char] = card

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._all_crops:
            self._populate_grid()

    # ── picker ────────────────────────────────────────────────────────────────
    def _on_card_clicked(self, char: str):
        crops   = self._all_crops[char]
        current = self._selected[char]

        dlg = PickerDialog(char, crops, current, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        if dlg.reassign_to:
            self._reassign(char, dlg.chosen, dlg.reassign_to)
        else:
            self._selected[char] = dlg.chosen
            self._cards[char].set_crop(crops[dlg.chosen])

    def _reassign(self, from_char: str, idx: int, to_char: str):
        crop = self._all_crops[from_char].pop(idx)

        if not self._all_crops[from_char]:
            del self._all_crops[from_char]
            del self._selected[from_char]
        else:
            self._selected[from_char] = min(self._selected[from_char], len(self._all_crops[from_char]) - 1)

        self._all_crops.setdefault(to_char, []).append(crop)
        self._selected.setdefault(to_char, len(self._all_crops[to_char]) - 1)

        self._populate_grid()

    # ── save ──────────────────────────────────────────────────────────────────
    def _save(self):
        if not self._all_crops:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save font",
            filter="WOFF2 font (*.woff2);;All files (*.*)",
        )
        if not path:
            return
        if not path.endswith(".woff2"):
            path += ".woff2"
        try:
            font_name   = os.path.splitext(os.path.basename(path))[0]
            char_dict   = {char: self._all_crops[char][self._selected[char]] for char in self._all_crops}
            create_woff2(char_dict, path, font_name)
            QMessageBox.information(self, "Saved", f"Font saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
