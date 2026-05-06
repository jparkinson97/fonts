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
from PyQt6.QtCore import Qt, QThread, QSize, pyqtSignal, QRect, QPoint, QByteArray
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QColor, QFont, QFontDatabase

if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))
from processor import build_char_dict
from create_woff2 import create_woff2, generate_preview_ttf

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
    done  = pyqtSignal(object)  # emits (char_dict, box_dict, orig_img)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            self.done.emit(build_char_dict(self.path))
        except Exception as e:
            self.error.emit(str(e))


# ── preview worker ────────────────────────────────────────────────────────────
class PreviewWorker(QThread):
    done  = pyqtSignal(bytes, str)
    error = pyqtSignal(str)

    def __init__(self, char: str, crop: np.ndarray):
        super().__init__()
        self.char = char
        self.crop = crop.copy()

    def run(self):
        try:
            self.done.emit(generate_preview_ttf(self.char, self.crop), self.char)
        except Exception as e:
            self.error.emit(str(e))


# ── crop editor widget ────────────────────────────────────────────────────────
class CropEditorWidget(QWidget):
    """Shows a padded region of the original image with a draggable bounding box."""

    HANDLE_SIZE = 10

    def __init__(self, orig_img: np.ndarray, box: tuple, parent=None):
        super().__init__(parent)
        self._orig_img = orig_img
        self._box = list(box)  # [x, y, w, h] — mutable during drag

        # Fixed view region: crop + generous padding. Stays constant so
        # scale doesn't jump mid-drag when the box changes.
        self._view_rect = self._compute_view_rect()
        self._view_pixmap = self._build_view_pixmap()

        self._drag_handle: str | None = None
        self._drag_origin: QPoint | None = None
        self._box_at_drag_start: list | None = None

        self.setMinimumSize(320, 260)

    def _compute_view_rect(self) -> tuple[int, int, int, int]:
        x, y, w, h = self._box
        pad = max(80, max(w, h) // 2)
        img_h, img_w = self._orig_img.shape[:2]
        vx  = max(0, x - pad)
        vy  = max(0, y - pad)
        vx2 = min(img_w, x + w + pad)
        vy2 = min(img_h, y + h + pad)
        return vx, vy, vx2 - vx, vy2 - vy

    def _build_view_pixmap(self) -> QPixmap:
        vx, vy, vw, vh = self._view_rect
        region = np.ascontiguousarray(
            cv2.cvtColor(self._orig_img[vy : vy + vh, vx : vx + vw], cv2.COLOR_BGR2RGB)
        )
        h, w = region.shape[:2]
        qimg = QImage(region.tobytes(), w, h, region.strides[0], QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _scale(self) -> float:
        return min(
            self.width()  / self._view_pixmap.width(),
            self.height() / self._view_pixmap.height(),
        )

    def _offset(self) -> tuple[float, float]:
        s = self._scale()
        return (
            (self.width()  - self._view_pixmap.width()  * s) / 2,
            (self.height() - self._view_pixmap.height() * s) / 2,
        )

    def _img_to_w(self, ix: float, iy: float) -> tuple[float, float]:
        vx, vy, _, _ = self._view_rect
        s = self._scale()
        ox, oy = self._offset()
        return (ix - vx) * s + ox, (iy - vy) * s + oy

    def _handle_rects(self) -> list[tuple[QRect, str]]:
        x, y, w, h = self._box
        cx, cy = x + w // 2, y + h // 2
        hh = self.HANDLE_SIZE // 2
        pts = [
            (x,     y,      "tl"), (cx,    y,      "t"),  (x + w, y,      "tr"),
            (x + w, cy,     "r"),
            (x + w, y + h,  "br"), (cx,    y + h,  "b"),  (x,     y + h,  "bl"),
            (x,     cy,     "l"),
        ]
        result = []
        for ix, iy, role in pts:
            wx, wy = self._img_to_w(ix, iy)
            result.append((QRect(int(wx - hh), int(wy - hh), self.HANDLE_SIZE, self.HANDLE_SIZE), role))
        return result

    def _hit_handle(self, pos: QPoint) -> str | None:
        for rect, role in self._handle_rects():
            if rect.adjusted(-6, -6, 6, 6).contains(pos):
                return role
        return None

    @staticmethod
    def _cursor_for(role: str):
        if role in ("tl", "br"): return Qt.CursorShape.SizeFDiagCursor
        if role in ("tr", "bl"): return Qt.CursorShape.SizeBDiagCursor
        if role in ("t",  "b"):  return Qt.CursorShape.SizeVerCursor
        if role in ("l",  "r"):  return Qt.CursorShape.SizeHorCursor
        return Qt.CursorShape.ArrowCursor

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.fillRect(self.rect(), QColor(BG))

        s = self._scale()
        ox, oy = self._offset()
        pw = int(self._view_pixmap.width()  * s)
        ph = int(self._view_pixmap.height() * s)
        painter.drawPixmap(int(ox), int(oy), pw, ph, self._view_pixmap)

        x, y, w, h = self._box
        wx1, wy1 = self._img_to_w(x,     y)
        wx2, wy2 = self._img_to_w(x + w, y + h)
        pen = QPen(QColor(ACCENT))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))

        for rect, _ in self._handle_rects():
            painter.fillRect(rect, QColor(ACCENT))
            painter.setPen(QPen(QColor("white")))
            painter.drawRect(rect)

        painter.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        role = self._hit_handle(event.pos())
        if role:
            self._drag_handle = role
            self._drag_origin = event.pos()
            self._box_at_drag_start = list(self._box)
            self.setCursor(self._cursor_for(role))

    def mouseMoveEvent(self, event):
        if self._drag_handle is None:
            role = self._hit_handle(event.pos())
            self.setCursor(self._cursor_for(role) if role else Qt.CursorShape.ArrowCursor)
            return

        s = self._scale()
        dx = (event.pos().x() - self._drag_origin.x()) / s
        dy = (event.pos().y() - self._drag_origin.y()) / s

        bx, by, bw, bh = self._box_at_drag_start
        img_h, img_w = self._orig_img.shape[:2]
        role = self._drag_handle

        if "l" in role:
            new_x = max(1, min(bx + int(dx), bx + bw - 10))
            bw = bw - (new_x - bx)
            bx = new_x
        if "r" in role:
            bw = max(10, min(int(bw + dx), img_w - bx - 1))
        if "t" in role:
            new_y = max(1, min(by + int(dy), by + bh - 10))
            bh = bh - (new_y - by)
            by = new_y
        if "b" in role:
            bh = max(10, min(int(bh + dy), img_h - by - 1))

        self._box = [bx, by, bw, bh]
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_handle = None
            self._drag_origin = None
            self._box_at_drag_start = None

    def get_box(self) -> tuple[int, int, int, int]:
        return tuple(self._box)


# ── crop editor dialog ────────────────────────────────────────────────────────
class CropEditorDialog(QDialog):
    def __init__(self, orig_img: np.ndarray, box: tuple, char: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Edit crop — "{char}"')
        self.setModal(True)
        self.setMinimumSize(560, 440)
        self.setStyleSheet(STYLESHEET + f"QDialog {{ background: {SURFACE}; }}")
        self.new_box = box

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        self._editor = CropEditorWidget(orig_img, box, self)
        outer.addWidget(self._editor, stretch=1)

        hint = QLabel("Drag the handles to adjust the crop boundary")
        hint.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        outer.addWidget(hint)

        row = QHBoxLayout()
        row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(cancel_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("browse")
        apply_btn.setFixedHeight(34)
        apply_btn.clicked.connect(self._apply)
        row.addWidget(apply_btn)
        outer.addLayout(row)

    def _apply(self):
        self.new_box = self._editor.get_box()
        self.accept()


# ── instance picker dialog ────────────────────────────────────────────────────
class PickerDialog(QDialog):
    THUMB = 80
    COLS  = 4

    def __init__(
        self,
        char: str,
        crops: list[np.ndarray],
        boxes: list[tuple],
        orig_img: np.ndarray | None,
        current: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f'Edit  "{char}"')
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setStyleSheet(STYLESHEET + f"QDialog {{ background: {SURFACE}; }}")

        self._char    = char
        self._crops   = [c.copy() for c in crops]
        self._boxes   = list(boxes)
        self._orig_img = orig_img
        self.chosen   = current
        self._pending_reassignments: list[tuple] = []
        self._preview_worker: PreviewWorker | None = None
        self._preview_font_id = -1

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        # ── instance count ──
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        outer.addWidget(self._count_lbl)

        # ── scrollable instance grid ──
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._inner  = QWidget()
        self._grid   = QGridLayout(self._inner)
        self._grid.setSpacing(10)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

        self._btns: list[QPushButton] = []
        self._rebuild_thumbnails()

        # ── edit crop ──
        self._edit_crop_btn = QPushButton("Edit Crop")
        self._edit_crop_btn.setObjectName("browse")
        self._edit_crop_btn.setFixedHeight(34)
        self._edit_crop_btn.setEnabled(orig_img is not None and bool(self._boxes))
        self._edit_crop_btn.clicked.connect(self._do_edit_crop)
        outer.addWidget(self._edit_crop_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        outer.addWidget(self._divider())

        # ── reassign ──
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

        self._reassign_status = QLabel("")
        self._reassign_status.setStyleSheet(f"color: {ACCENT}; font-size: 11px;")
        outer.addWidget(self._reassign_status)

        outer.addWidget(self._divider())

        # ── font preview ──
        preview_hdr = QLabel("Font preview:")
        preview_hdr.setStyleSheet(f"color: {TEXT}; font-size: 12px;")
        outer.addWidget(preview_hdr)

        preview_row = QHBoxLayout()
        self._preview_lbl = QLabel(char)
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setFixedSize(120, 120)
        self._preview_lbl.setStyleSheet(f"""
            QLabel {{
                background: white; color: black;
                border: 1px solid {BORDER}; border-radius: 6px;
                font-size: 13px;
            }}
        """)
        preview_row.addWidget(self._preview_lbl)

        preview_ctrl = QVBoxLayout()
        self._preview_status = QLabel("Click Generate to preview")
        self._preview_status.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._preview_status.setWordWrap(True)
        preview_ctrl.addWidget(self._preview_status)
        gen_btn = QPushButton("Generate Preview")
        gen_btn.setFixedHeight(34)
        gen_btn.clicked.connect(self._do_generate_preview)
        preview_ctrl.addWidget(gen_btn)
        preview_ctrl.addStretch()
        preview_row.addLayout(preview_ctrl)
        outer.addLayout(preview_row)

        outer.addWidget(self._divider())

        # ── bottom buttons ──
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        done_btn = QPushButton("Done")
        done_btn.setObjectName("save")
        done_btn.setFixedHeight(34)
        done_btn.clicked.connect(self.accept)
        btn_row.addWidget(done_btn)
        outer.addLayout(btn_row)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _divider(self) -> QFrame:
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {BORDER};")
        return div

    def _rebuild_thumbnails(self):
        for btn in self._btns:
            btn.deleteLater()
        self._btns.clear()
        while self._grid.count():
            item = self._grid.takeAt(0)
            if w := item.widget():
                w.deleteLater()

        for i, crop in enumerate(self._crops):
            btn = QPushButton()
            btn.setFixedSize(self.THUMB + 20, self.THUMB + 20)
            btn.setIcon(QIcon(crop_to_pixmap(crop, self.THUMB)))
            btn.setIconSize(QSize(self.THUMB, self.THUMB))
            btn.setCheckable(True)
            btn.setChecked(i == self.chosen)
            btn.setStyleSheet(self._btn_style(i == self.chosen))
            btn.clicked.connect(lambda _, idx=i: self._pick(idx))
            self._grid.addWidget(btn, i // self.COLS, i % self.COLS)
            self._btns.append(btn)

        n = len(self._crops)
        self._count_lbl.setText(f"{n} instance(s) — click one to select")
        rows = max(1, min((n + self.COLS - 1) // self.COLS, 4))
        self._scroll.setFixedHeight(rows * (self.THUMB + 40) + 20)

    def _btn_style(self, selected: bool) -> str:
        border = SEL_B if selected else BORDER
        bg     = SEL   if selected else CARD
        return f"""
            QPushButton {{
                background: {bg}; border: 2px solid {border}; border-radius: 8px;
            }}
            QPushButton:hover {{ border-color: {ACCENT}; }}
        """

    # ── actions ───────────────────────────────────────────────────────────────
    def _pick(self, idx: int):
        self.chosen = idx
        for i, btn in enumerate(self._btns):
            btn.setStyleSheet(self._btn_style(i == idx))
            btn.setChecked(i == idx)

    def _do_reassign(self):
        target = self._char_input.text().strip()
        if not target or not self._crops:
            return
        idx  = self.chosen
        crop = self._crops.pop(idx)
        box  = self._boxes.pop(idx) if idx < len(self._boxes) else (0, 0, crop.shape[1], crop.shape[0])
        self._pending_reassignments.append((crop, box, target))
        self.chosen = max(0, min(idx, len(self._crops) - 1))
        self._char_input.clear()
        chars = [r[2] for r in self._pending_reassignments]
        self._reassign_status.setText("Pending: " + ", ".join(f"→ '{c}'" for c in chars))
        self._rebuild_thumbnails()

    def _do_edit_crop(self):
        if self._orig_img is None or self.chosen >= len(self._boxes):
            return
        dlg = CropEditorDialog(self._orig_img, self._boxes[self.chosen], self._char, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_box = dlg.new_box
        self._boxes[self.chosen] = new_box
        x, y, w, h = new_box
        img_h, img_w = self._orig_img.shape[:2]
        new_crop = self._orig_img[
            max(0, y) : min(img_h, y + h),
            max(0, x) : min(img_w, x + w),
        ].copy()
        self._crops[self.chosen] = new_crop
        self._btns[self.chosen].setIcon(QIcon(crop_to_pixmap(new_crop, self.THUMB)))

    def _do_generate_preview(self):
        if self._preview_worker and self._preview_worker.isRunning():
            return
        if not self._crops:
            return
        self._preview_status.setText("Generating…")
        self._preview_worker = PreviewWorker(self._char, self._crops[self.chosen])
        self._preview_worker.done.connect(self._on_preview_done)
        self._preview_worker.error.connect(
            lambda e: self._preview_status.setText(f"Error: {e}")
        )
        self._preview_worker.start()

    def _on_preview_done(self, ttf_bytes: bytes, char: str):
        if self._preview_font_id >= 0:
            QFontDatabase.removeApplicationFont(self._preview_font_id)
        font_data = QByteArray(ttf_bytes)
        self._preview_font_id = QFontDatabase.addApplicationFontFromData(font_data)
        if self._preview_font_id < 0:
            self._preview_status.setText("Could not load preview font")
            return
        families = QFontDatabase.applicationFontFamilies(self._preview_font_id)
        if families:
            self._preview_lbl.setStyleSheet(f"""
                QLabel {{
                    background: white; color: black;
                    border: 1px solid {BORDER}; border-radius: 6px;
                    font-family: "{families[0]}"; font-size: 72pt;
                }}
            """)
            self._preview_lbl.setText(char)
            self._preview_status.setText("Preview ready")
        else:
            self._preview_status.setText("Font loaded but no families found")

    # ── result accessors ──────────────────────────────────────────────────────
    def get_updated_crops(self) -> list[np.ndarray]:
        return self._crops

    def get_updated_boxes(self) -> list[tuple]:
        return self._boxes

    def get_pending_reassignments(self) -> list[tuple]:
        return self._pending_reassignments


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

        self._all_crops:      dict[str, list[np.ndarray]] = {}
        self._all_boxes:      dict[str, list[tuple]]      = {}
        self._original_image: np.ndarray | None           = None
        self._selected:       dict[str, int]              = {}
        self._cards:          dict[str, CharCard]         = {}
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

    def _on_done(self, result):
        char_dict, box_dict, orig_img = result
        self._all_crops      = char_dict
        self._all_boxes      = box_dict
        self._original_image = orig_img
        self._selected       = {char: 0 for char in char_dict}
        self._status.setText(f"{len(char_dict)} character(s) found — click any to edit")
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
        dlg = PickerDialog(
            char,
            self._all_crops[char],
            self._all_boxes.get(char, []),
            self._original_image,
            self._selected[char],
            self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Apply crop/box edits for this character
        self._all_crops[char] = dlg.get_updated_crops()
        self._all_boxes[char] = dlg.get_updated_boxes()

        # Apply pending reassignments
        for reassign_crop, reassign_box, to_char in dlg.get_pending_reassignments():
            self._all_crops.setdefault(to_char, []).append(reassign_crop)
            self._all_boxes.setdefault(to_char, []).append(reassign_box)
            self._selected.setdefault(to_char, len(self._all_crops[to_char]) - 1)

        # Update or remove this character
        if self._all_crops.get(char):
            self._selected[char] = min(dlg.chosen, len(self._all_crops[char]) - 1)
        else:
            self._all_crops.pop(char, None)
            self._all_boxes.pop(char, None)
            self._selected.pop(char, None)

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
            font_name = os.path.splitext(os.path.basename(path))[0]
            char_dict = {char: self._all_crops[char][self._selected[char]] for char in self._all_crops}
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
