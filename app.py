"""PyQt6 user interface for the four-track recorder."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from audio_engine import AudioEngine, Track
from compressor import CompressorSettings
from eq import EQBand, default_eq_bands
from storage import save_session

PALETTE = {
    "background": "#fff3df",
    "primary": "#5a4031",
    "accent": "#7cbf9b",
    "accent_active": "#9fd7b5",
    "record": "#c0392b",
    "record_active": "#ff4b3e",
    "record_arm": "#f2c078",
    "highlight": "#f8dfc8",
    "timeline_bg": "#fcebd8",
    "timeline_grid": "#eedcc7",
    "timeline_major": "#dec7ad",
    "waveform": "#314d79",
    "waveform_fill": "#8fb5de",
    "waveform_live": "#ff8b75",
}


def create_transport_icon(kind: str, size: int = 64) -> QtGui.QIcon:
    """Create an in-memory icon for transport controls."""

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    rect = QtCore.QRectF(6, 6, size - 12, size - 12)

    if kind == "record":
        painter.setBrush(QtGui.QColor(PALETTE["record_active"]))
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["record_active"])))
        painter.drawEllipse(rect)
    elif kind == "play":
        painter.setBrush(QtGui.QColor(PALETTE["accent_active"]))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        path = QtGui.QPainterPath()
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right(), rect.center().y())
        path.lineTo(rect.left(), rect.bottom())
        path.closeSubpath()
        painter.drawPath(path)
    else:
        painter.setBrush(QtGui.QColor(PALETTE["accent"]))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 8, 8)

    painter.end()
    return QtGui.QIcon(pixmap)


class GridWidget(QtWidgets.QWidget):
    """Widget that paints a faint grid background."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D401,N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(PALETTE["timeline_bg"]))

        major_pen = QtGui.QPen(QtGui.QColor(PALETTE["timeline_major"]))
        major_pen.setWidth(2)
        minor_pen = QtGui.QPen(QtGui.QColor(PALETTE["timeline_grid"]))
        minor_pen.setWidth(1)

        bar_width = 160
        beats = 4
        for bar in range(0, max(1, self.width() // bar_width + 2)):
            x = bar * bar_width
            painter.setPen(major_pen)
            painter.drawLine(x, 0, x, self.height())
            for beat in range(1, beats):
                sub_x = x + int(bar_width / beats * beat)
                painter.setPen(minor_pen)
                painter.drawLine(sub_x, 0, sub_x, self.height())

        painter.setPen(minor_pen)
        mid_y = self.height() // 2
        painter.drawLine(0, mid_y, self.width(), mid_y)
        super().paintEvent(event)


class WaveformView(QtWidgets.QWidget):
    """Simplified visualisation of audio waveforms for each track."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._waveform: np.ndarray | None = None
        self._live_mode = False
        self.setMinimumHeight(140)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def set_waveform(self, audio: np.ndarray | None) -> None:
        if audio is None or audio.size == 0:
            self._waveform = None
        else:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            max_val = peak if peak != 0 else 1.0
            self._waveform = (audio / max_val).astype(np.float32)
        self.update()

    def set_live_mode(self, active: bool) -> None:
        if self._live_mode != active:
            self._live_mode = active
            self.update()

    def _build_path(self, width: int, height: int) -> QtGui.QPainterPath | None:
        if self._waveform is None or width <= 0:
            return None
        samples = self._waveform
        step = max(1, len(samples) // width)
        center_y = height / 2
        scale_y = center_y * 0.9
        path = QtGui.QPainterPath()
        path.moveTo(0, center_y)
        for x in range(width):
            start = x * step
            end = min(len(samples), start + step)
            window = samples[start:end]
            if window.size == 0:
                level = 0.0
            else:
                level = float(window.max())
            y = center_y - level * scale_y
            path.lineTo(x, y)
        path.lineTo(width, center_y)
        for x in range(width, -1, -1):
            start = x * step
            end = min(len(samples), start + step)
            window = samples[start:end]
            if window.size == 0:
                level = 0.0
            else:
                level = float(window.min())
            y = center_y - level * scale_y
            path.lineTo(x, y)
        path.closeSubpath()
        return path

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D401,N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        if self._live_mode:
            gradient.setColorAt(0.0, QtGui.QColor(PALETTE["waveform_live"]).lighter(140))
            gradient.setColorAt(1.0, QtGui.QColor(PALETTE["waveform_live"]).lighter(190))
        else:
            gradient.setColorAt(0.0, QtGui.QColor(255, 255, 255, 120))
            gradient.setColorAt(1.0, QtGui.QColor(PALETTE["highlight"]))
        painter.fillRect(rect, gradient)

        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["timeline_grid"])))
        for y in range(0, rect.height(), 30):
            painter.drawLine(rect.left(), y, rect.right(), y)

        path = self._build_path(rect.width(), rect.height())
        if path is None:
            painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["primary"])))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No audio yet")
        else:
            fill_color = QtGui.QColor(PALETTE["waveform_fill"])
            if self._live_mode:
                fill_color = QtGui.QColor(PALETTE["waveform_live"])
            painter.setBrush(fill_color)
            painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["waveform"]), 1.2))
            painter.drawPath(path)

        super().paintEvent(event)


class EQCurveWidget(QtWidgets.QWidget):
    """Visual preview of the EQ response curve."""

    def __init__(self, bands: List[EQBand], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._bands = [EQBand(b.low_freq, b.high_freq, b.gain_db) for b in bands]
        self.setMinimumHeight(200)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_bands(self, bands: List[EQBand]) -> None:
        self._bands = [EQBand(b.low_freq, b.high_freq, b.gain_db) for b in bands]
        self.update()

    @staticmethod
    def _gaussian_response(freqs: np.ndarray, band: EQBand) -> np.ndarray:
        low = max(band.low_freq, 20)
        high = max(band.high_freq, low + 1)
        center = np.sqrt(low * high)
        width = max(np.log10(high) - np.log10(low), 1e-3)
        sigma = width / 2.5
        exponent = (np.log10(freqs) - np.log10(center)) / sigma
        return band.gain_db * np.exp(-0.5 * exponent**2)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D401,N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(12, 12, -12, -12)

        bg = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg.setColorAt(0.0, QtGui.QColor(PALETTE["highlight"]))
        bg.setColorAt(1.0, QtGui.QColor(255, 255, 255, 180))
        painter.fillRect(rect, bg)

        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["timeline_grid"])) )
        for step in range(5):
            y = rect.top() + step * rect.height() / 4
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))

        freq_ticks = [20, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000]
        min_log = np.log10(20)
        max_log = np.log10(20000)
        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["timeline_major"])) )
        for freq in freq_ticks:
            pos = (np.log10(freq) - min_log) / (max_log - min_log)
            x = rect.left() + pos * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())

        if not self._bands:
            painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["primary"])))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No EQ bands")
            return

        freqs = np.logspace(min_log, max_log, num=400)
        response = np.zeros_like(freqs)
        for band in self._bands:
            response += self._gaussian_response(freqs, band)

        path = QtGui.QPainterPath()
        mid_y = rect.center().y()
        max_gain = max(12.0, float(np.max(np.abs(response))) or 0.0)
        gain_scale = (rect.height() / 2) * 0.85 / max_gain
        for idx, (freq, gain) in enumerate(zip(freqs, response)):
            pos = (np.log10(freq) - min_log) / (max_log - min_log)
            x = rect.left() + pos * rect.width()
            y = mid_y - gain * gain_scale
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE["waveform"]), 2.5))
        painter.drawPath(path)

        fill_path = QtGui.QPainterPath(path)
        fill_path.lineTo(rect.right(), mid_y)
        fill_path.lineTo(rect.left(), mid_y)
        fill_path.closeSubpath()
        painter.fillPath(fill_path, QtGui.QColor(PALETTE["waveform_fill"]).lighter(140))


class EQDialog(QtWidgets.QDialog):
    def __init__(self, bands: List[EQBand], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("8-Band EQ")
        self._bands = [EQBand(b.low_freq, b.high_freq, b.gain_db) for b in bands]
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        self._curve = EQCurveWidget(self._bands)
        layout.addWidget(self._curve)

        control_grid = QtWidgets.QHBoxLayout()
        control_grid.setSpacing(16)
        layout.addLayout(control_grid)

        self._gain_sliders: List[QtWidgets.QSlider] = []
        self._range_labels: List[QtWidgets.QLabel] = []
        self._low_spins: List[QtWidgets.QDoubleSpinBox] = []
        self._high_spins: List[QtWidgets.QDoubleSpinBox] = []
        self._gain_labels: List[QtWidgets.QLabel] = []

        for idx, band in enumerate(self._bands):
            column = QtWidgets.QWidget()
            column_layout = QtWidgets.QVBoxLayout(column)
            column_layout.setSpacing(6)
            column_layout.setContentsMargins(0, 0, 0, 0)

            header = QtWidgets.QLabel(f"Band {idx + 1}")
            header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            header.setStyleSheet("font-weight: 600;")
            column_layout.addWidget(header)

            range_label = QtWidgets.QLabel()
            range_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            column_layout.addWidget(range_label)
            self._range_labels.append(range_label)

            gain_label = QtWidgets.QLabel()
            gain_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            column_layout.addWidget(gain_label)
            self._gain_labels.append(gain_label)

            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
            slider.setRange(-180, 180)
            slider.setSingleStep(5)
            slider.setValue(int(band.gain_db * 10))
            slider.setTickInterval(30)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
            slider.valueChanged.connect(lambda value, i=idx: self._on_gain_changed(i, value))
            column_layout.addWidget(slider, stretch=1)
            self._gain_sliders.append(slider)

            low_spin = QtWidgets.QDoubleSpinBox()
            low_spin.setRange(20.0, 19900.0)
            low_spin.setDecimals(1)
            low_spin.setSuffix(" Hz")
            low_spin.setValue(band.low_freq)
            low_spin.valueChanged.connect(lambda value, i=idx: self._on_low_changed(i, value))
            column_layout.addWidget(low_spin)
            self._low_spins.append(low_spin)

            high_spin = QtWidgets.QDoubleSpinBox()
            high_spin.setRange(30.0, 20000.0)
            high_spin.setDecimals(1)
            high_spin.setSuffix(" Hz")
            high_spin.setValue(band.high_freq)
            high_spin.valueChanged.connect(lambda value, i=idx: self._on_high_changed(i, value))
            column_layout.addWidget(high_spin)
            self._high_spins.append(high_spin)

            control_grid.addWidget(column)

        self._sync_labels()

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _sync_labels(self) -> None:
        for idx, band in enumerate(self._bands):
            self._range_labels[idx].setText(f"{int(band.low_freq)} – {int(band.high_freq)} Hz")
            self._gain_labels[idx].setText(f"{band.gain_db:+.1f} dB")
        self._curve.set_bands(self._bands)

    def _on_gain_changed(self, index: int, value: int) -> None:
        self._bands[index].gain_db = value / 10
        self._gain_labels[index].setText(f"{self._bands[index].gain_db:+.1f} dB")
        self._curve.set_bands(self._bands)

    def _on_low_changed(self, index: int, value: float) -> None:
        high = self._bands[index].high_freq
        if value >= high - 10:
            value = high - 10
            self._low_spins[index].blockSignals(True)
            self._low_spins[index].setValue(value)
            self._low_spins[index].blockSignals(False)
        self._bands[index].low_freq = max(20.0, value)
        self._sync_labels()

    def _on_high_changed(self, index: int, value: float) -> None:
        low = self._bands[index].low_freq
        if value <= low + 10:
            value = low + 10
            self._high_spins[index].blockSignals(True)
            self._high_spins[index].setValue(value)
            self._high_spins[index].blockSignals(False)
        self._bands[index].high_freq = min(20000.0, value)
        self._sync_labels()

    def settings(self) -> List[EQBand]:
        return [EQBand(b.low_freq, b.high_freq, b.gain_db) for b in self._bands]


class CompressorDialog(QtWidgets.QDialog):
    def __init__(self, settings: CompressorSettings | None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compressor")
        layout = QtWidgets.QFormLayout(self)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setRange(-60, 0)
        self.threshold.setValue(settings.threshold_db if settings else -18)
        self.ratio = QtWidgets.QDoubleSpinBox()
        self.ratio.setRange(1, 20)
        self.ratio.setSingleStep(0.1)
        self.ratio.setValue(settings.ratio if settings else 4.0)
        self.attack = QtWidgets.QDoubleSpinBox()
        self.attack.setRange(0.1, 200)
        self.attack.setValue(settings.attack_ms if settings else 10.0)
        self.release = QtWidgets.QDoubleSpinBox()
        self.release.setRange(5, 500)
        self.release.setValue(settings.release_ms if settings else 100.0)
        self.makeup = QtWidgets.QDoubleSpinBox()
        self.makeup.setRange(-12, 12)
        self.makeup.setValue(settings.makeup_gain_db if settings else 0.0)
        layout.addRow("Threshold (dB)", self.threshold)
        layout.addRow("Ratio", self.ratio)
        layout.addRow("Attack (ms)", self.attack)
        layout.addRow("Release (ms)", self.release)
        layout.addRow("Make-up (dB)", self.makeup)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def settings(self) -> CompressorSettings:
        return CompressorSettings(
            threshold_db=self.threshold.value(),
            ratio=self.ratio.value(),
            attack_ms=self.attack.value(),
            release_ms=self.release.value(),
            makeup_gain_db=self.makeup.value(),
        )


class TrackWidget(QtWidgets.QFrame):
    record_toggled = QtCore.pyqtSignal(int, bool)
    mono_toggled = QtCore.pyqtSignal(int, bool)
    gain_changed = QtCore.pyqtSignal(int, float)
    eq_requested = QtCore.pyqtSignal(int)
    compressor_requested = QtCore.pyqtSignal(int)
    crop_requested = QtCore.pyqtSignal(int, float, float)
    move_requested = QtCore.pyqtSignal(int, int)

    def __init__(self, index: int, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setObjectName("trackFrame")
        self.setMinimumHeight(220)
        self.setStyleSheet(
            "#trackFrame {"
            "    background-color: rgba(255, 244, 226, 0.7);"
            "    border-radius: 18px;"
            "    border: 1px solid %s;"
            "    box-shadow: 0 12px 30px rgba(154, 127, 92, 0.18);"
            "}" % PALETTE["highlight"]
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 18, 18, 18)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(12)
        self.title = QtWidgets.QLabel(f"Track {index + 1}")
        self.title.setStyleSheet(
            "font-size: 18px; font-weight: 600; letter-spacing: 0.5px; color: %s;" % PALETTE["primary"]
        )
        header.addWidget(self.title)
        header.addStretch()
        self.record_button = QtWidgets.QPushButton("Arm")
        self.record_button.setCheckable(True)
        self.record_button.setToolTip("Arm track for recording")
        self.record_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.record_button.setStyleSheet(
            "QPushButton {"
            "    background-color: transparent;"
            "    color: %s;"
            "    border-radius: 20px;"
            "    border: 2px solid %s;"
            "    padding: 10px 22px;"
            "    font-weight: bold;"
            "}"
            "QPushButton:checked {"
            "    background-color: %s;"
            "    color: white;"
            "    border-color: %s;"
            "}"
            % (PALETTE["primary"], PALETTE["record"], PALETTE["record_arm"], PALETTE["record"])
        )
        self.record_button.toggled.connect(self._on_record_toggled)
        header.addWidget(self.record_button)
        self.mono_button = QtWidgets.QPushButton("Stereo")
        self.mono_button.setCheckable(True)
        self.mono_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.mono_button.setStyleSheet(
            "QPushButton {"
            "    background-color: rgba(76, 140, 107, 40);"
            "    color: %s;"
            "    border-radius: 16px;"
            "    padding: 10px 20px;"
            "}"
            "QPushButton:checked {"
            "    background-color: %s;"
            "    color: white;"
            "}"
            % (PALETTE["primary"], PALETTE["accent"])
        )
        self.mono_button.toggled.connect(self._on_mono_toggled)
        header.addWidget(self.mono_button)
        layout.addLayout(header)

        self.waveform = WaveformView()
        layout.addWidget(self.waveform)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(12)
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-120, 120)
        self.gain_slider.setValue(0)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        gain_label = QtWidgets.QLabel("Gain (dB)")
        gain_label.setStyleSheet("font-weight: 600;")
        control_layout.addWidget(gain_label)
        control_layout.addWidget(self.gain_slider)
        layout.addLayout(control_layout)

        processing_layout = QtWidgets.QHBoxLayout()
        processing_layout.setSpacing(10)
        self.eq_button = QtWidgets.QPushButton("EQ")
        self.eq_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.eq_button.clicked.connect(lambda: self.eq_requested.emit(self.index))
        self.eq_button.setStyleSheet(
            "QPushButton {"
            "    background-color: rgba(76, 140, 107, 0.25);"
            f"    color: {PALETTE['primary']};"
            "    border-radius: 12px;"
            "    padding: 10px 16px;"
            "    font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "    background-color: rgba(98, 177, 136, 0.5);"
            "}"
        )
        processing_layout.addWidget(self.eq_button)
        self.comp_button = QtWidgets.QPushButton("Comp")
        self.comp_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.comp_button.clicked.connect(lambda: self.compressor_requested.emit(self.index))
        self.comp_button.setStyleSheet(
            "QPushButton {"
            "    background-color: rgba(42, 79, 110, 0.2);"
            "    color: white;"
            "    border-radius: 12px;"
            "    padding: 10px 16px;"
            "    font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "    background-color: rgba(42, 79, 110, 0.35);"
            "}"
        )
        processing_layout.addWidget(self.comp_button)
        layout.addLayout(processing_layout)

        edit_layout = QtWidgets.QHBoxLayout()
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 9999)
        self.start_spin.setSuffix(" s")
        self.start_spin.setFixedHeight(40)
        self.start_spin.setStyleSheet("QDoubleSpinBox { padding: 6px 10px; border-radius: 10px; }")
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0, 9999)
        self.end_spin.setSuffix(" s")
        self.end_spin.setFixedHeight(40)
        self.end_spin.setStyleSheet("QDoubleSpinBox { padding: 6px 10px; border-radius: 10px; }")
        crop_button = QtWidgets.QPushButton("Crop")
        crop_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        crop_button.setStyleSheet(
            "QPushButton {"
            "    background-color: rgba(192, 57, 43, 0.2);"
            "    color: %s;"
            "    border-radius: 12px;"
            "    padding: 10px 18px;"
            "    font-weight: 600;"
            "}" % PALETTE["record"]
            + "QPushButton:hover { background-color: rgba(255, 75, 62, 0.35); }"
        )
        crop_button.clicked.connect(self._emit_crop)
        edit_layout.addWidget(QtWidgets.QLabel("Start"))
        edit_layout.addWidget(self.start_spin)
        edit_layout.addWidget(QtWidgets.QLabel("End"))
        edit_layout.addWidget(self.end_spin)
        edit_layout.addWidget(crop_button)
        layout.addLayout(edit_layout)

        move_layout = QtWidgets.QHBoxLayout()
        move_layout.addWidget(QtWidgets.QLabel("Move to"))
        self.move_combo = QtWidgets.QComboBox()
        self.move_combo.addItems(["Track 1", "Track 2", "Track 3", "Track 4"])
        self.move_combo.setFixedHeight(40)
        self.move_combo.setStyleSheet("QComboBox { padding: 6px 12px; border-radius: 10px; }")
        move_button = QtWidgets.QPushButton("Move")
        move_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        move_button.setStyleSheet(
            "QPushButton {"
            "    background-color: rgba(91, 70, 54, 0.2);"
            f"    color: {PALETTE['primary']};"
            "    border-radius: 12px;"
            "    padding: 10px 18px;"
            "    font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "    background-color: rgba(91, 70, 54, 0.35);"
            "}"
        )
        move_button.clicked.connect(self._emit_move)
        move_layout.addWidget(self.move_combo)
        move_layout.addWidget(move_button)
        layout.addLayout(move_layout)

    def update_from_track(self, track: Track, live_audio: np.ndarray | None = None) -> None:
        self.title.setText(track.name or f"Track {self.index + 1}")
        self._sync_button(self.record_button, track.armed)
        self.record_button.setText("Armed" if track.armed else "Arm")
        self._sync_button(self.mono_button, not track.stereo)
        self.mono_button.setText("Mono" if self.mono_button.isChecked() else "Stereo")
        self.gain_slider.blockSignals(True)
        self.gain_slider.setValue(int(track.gain_db * 10))
        self.gain_slider.blockSignals(False)
        if live_audio is not None:
            self.waveform.set_waveform(live_audio)
            self.waveform.set_live_mode(True)
        else:
            self.waveform.set_waveform(track.data)
            self.waveform.set_live_mode(False)

    @staticmethod
    def _sync_button(button: QtWidgets.QAbstractButton, state: bool) -> None:
        old = button.blockSignals(True)
        button.setChecked(state)
        button.blockSignals(old)

    def _on_record_toggled(self, state: bool) -> None:
        self.record_button.setText("Armed" if state else "Arm")
        self.record_toggled.emit(self.index, state)

    def _on_mono_toggled(self, state: bool) -> None:
        self.mono_button.setText("Mono" if state else "Stereo")
        self.mono_toggled.emit(self.index, state)

    def _on_gain_changed(self, value: int) -> None:
        self.gain_changed.emit(self.index, value / 10)

    def _emit_crop(self) -> None:
        self.crop_requested.emit(self.index, self.start_spin.value(), self.end_spin.value())

    def _emit_move(self) -> None:
        dest = self.move_combo.currentIndex()
        if dest != self.index:
            self.move_requested.emit(self.index, dest)


class TapeDeckView(QtWidgets.QWidget):
    record = QtCore.pyqtSignal()
    stop = QtCore.pyqtSignal()
    play = QtCore.pyqtSignal()
    play_stopped = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("4-Track Tape Deck")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 22px; font-weight: 600; letter-spacing: 0.4px; color: %s;" % PALETTE["primary"]
        )
        layout.addWidget(title)
        self.track_buttons: List[QtWidgets.QPushButton] = []
        for i in range(4):
            button = QtWidgets.QPushButton(f"Track {i + 1}")
            button.setCheckable(True)
            button.setStyleSheet(
                "QPushButton {"
                f"    background-color: rgba(255, 255, 255, 0.3); color: {PALETTE['primary']};"
                "    border-radius: 16px; padding: 14px 18px; font-size: 16px;"
                "    border: 2px solid %s;"
                "}"
                "QPushButton:checked {"
                f"    background-color: {PALETTE['record_arm']}; color: white;"
                f"    border-color: {PALETTE['record']};"
                "}"
                % PALETTE["timeline_grid"]
            )
            layout.addWidget(button)
            self.track_buttons.append(button)
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(20)
        controls.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setCheckable(True)
        self.play_button.setToolTip("Play")
        self.play_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.play_button.setIcon(create_transport_icon("play"))
        self.play_button.setIconSize(QtCore.QSize(48, 48))
        self.play_button.setFixedSize(96, 96)
        self.play_button.setStyleSheet(
            "QPushButton {"
            f"    background-color: {PALETTE['accent']};"
            "    border-radius: 48px;"
            "    border: none;"
            "}"
            "QPushButton:checked {"
            f"    background-color: {PALETTE['accent_active']};"
            "    border: 3px solid rgba(255, 255, 255, 120);"
            "}"
        )
        self.play_button.toggled.connect(self._on_play_toggled)
        controls.addWidget(self.play_button)
        self.record_button = QtWidgets.QPushButton()
        self.record_button.setCheckable(True)
        self.record_button.setToolTip("Record")
        self.record_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.record_button.setIcon(create_transport_icon("record"))
        self.record_button.setIconSize(QtCore.QSize(48, 48))
        self.record_button.setFixedSize(96, 96)
        self.record_button.setStyleSheet(
            "QPushButton {"
            f"    background-color: {PALETTE['record']};"
            "    border-radius: 48px;"
            "    border: none;"
            "}"
            "QPushButton:checked {"
            f"    background-color: {PALETTE['record_active']};"
            "    border: 3px solid rgba(255, 255, 255, 150);"
            "}"
        )
        self.record_button.toggled.connect(self._toggle_record)
        controls.addWidget(self.record_button)
        layout.addLayout(controls)

    def _toggle_record(self, state: bool) -> None:
        if state:
            self.record.emit()
        else:
            self.stop.emit()

    def _on_play_toggled(self, state: bool) -> None:
        if state:
            self.play.emit()
        else:
            self.play_stopped.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Python 4-Track Recorder")
        self.resize(1200, 800)
        self.engine = AudioEngine()
        self._build_ui()
        self._update_device_lists()
        self._last_playing_state = False
        self._last_recording_state = False
        self._transport_timer = QtCore.QTimer(self)
        self._transport_timer.setInterval(200)
        self._transport_timer.timeout.connect(self._poll_engine_state)
        self._transport_timer.start()
        self._refresh_track_views()

    # UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(24, 24, 24, 24)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(16)
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setCheckable(True)
        self.play_button.setIcon(create_transport_icon("play"))
        self.play_button.setIconSize(QtCore.QSize(48, 48))
        self.play_button.setFixedSize(88, 88)
        self.play_button.setToolTip("Play session")
        self.play_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.play_button.setStyleSheet(
            "QPushButton {"
            f"    background-color: {PALETTE['accent']};"
            "    border-radius: 44px;"
            "    border: none;"
            "}"
            "QPushButton:checked {"
            f"    background-color: {PALETTE['accent_active']};"
            "    border: 3px solid rgba(255, 255, 255, 160);"
            "}"
        )
        self.play_button.toggled.connect(self._toggle_playback)
        top_bar.addWidget(self.play_button)
        self.record_button = QtWidgets.QPushButton()
        self.record_button.setCheckable(True)
        self.record_button.setIcon(create_transport_icon("record"))
        self.record_button.setIconSize(QtCore.QSize(48, 48))
        self.record_button.setFixedSize(88, 88)
        self.record_button.setToolTip("Record armed tracks")
        self.record_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.record_button.setStyleSheet(
            "QPushButton {"
            f"    background-color: {PALETTE['record']};"
            "    border-radius: 44px;"
            "    border: none;"
            "}"
            "QPushButton:checked {"
            f"    background-color: {PALETTE['record_active']};"
            "    border: 3px solid rgba(255, 255, 255, 180);"
            "}"
        )
        self.record_button.toggled.connect(self._toggle_recording)
        top_bar.addWidget(self.record_button)
        top_bar.addSpacing(12)
        tempo_label = QtWidgets.QLabel("Tempo")
        tempo_label.setStyleSheet("font-weight: 600;")
        top_bar.addWidget(tempo_label)
        self.tempo_spin = QtWidgets.QSpinBox()
        self.tempo_spin.setRange(40, 240)
        self.tempo_spin.setValue(self.engine.tempo)
        self.tempo_spin.valueChanged.connect(self._tempo_changed)
        top_bar.addWidget(self.tempo_spin)
        self.tempo_spin.setFixedHeight(44)
        self.tempo_spin.setStyleSheet(
            "QSpinBox { padding: 6px 12px; font-size: 16px; border-radius: 10px; }"
        )
        self.click_checkbox = QtWidgets.QCheckBox("Click Track")
        self.click_checkbox.setChecked(True)
        self.click_checkbox.toggled.connect(self._toggle_click)
        top_bar.addWidget(self.click_checkbox)
        self.click_checkbox.setStyleSheet("QCheckBox { font-size: 16px; }")
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.addItems(["44100", "48000"])
        self.sample_rate_combo.setFixedHeight(44)
        self.sample_rate_combo.setStyleSheet(
            "QComboBox { padding: 6px 12px; font-size: 16px; border-radius: 10px; }"
        )
        sample_label = QtWidgets.QLabel("Export Rate")
        sample_label.setStyleSheet("font-weight: 600;")
        top_bar.addWidget(sample_label)
        top_bar.addWidget(self.sample_rate_combo)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["wav", "mp3"])
        self.format_combo.setFixedHeight(44)
        self.format_combo.setStyleSheet(
            "QComboBox { padding: 6px 12px; font-size: 16px; border-radius: 10px; }"
        )
        format_label = QtWidgets.QLabel("Format")
        format_label.setStyleSheet("font-weight: 600;")
        top_bar.addWidget(format_label)
        top_bar.addWidget(self.format_combo)
        export_button = QtWidgets.QPushButton("Export Mix")
        export_button.clicked.connect(self._export_mix)
        export_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        export_button.setStyleSheet(
            "QPushButton {"
            f"    background-color: {PALETTE['accent']};"
            "    color: white;"
            "    padding: 12px 20px;"
            "    border-radius: 16px;"
            "    font-weight: 600;"
            "}"
            "QPushButton:hover {"
            f"    background-color: {PALETTE['accent_active']};"
            "}"
        )
        top_bar.addWidget(export_button)
        save_button = QtWidgets.QPushButton("Save Session")
        save_button.clicked.connect(self._save_session)
        save_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        save_button.setStyleSheet(
            "QPushButton {"
            "    background-color: rgba(91, 70, 54, 0.2);"
            f"    color: {PALETTE['primary']};"
            "    padding: 12px 20px;"
            "    border-radius: 16px;"
            "    font-weight: 600;"
            "}"
            "QPushButton:hover {"
            f"    background-color: rgba(91, 70, 54, 0.35);"
            "}"
        )
        top_bar.addWidget(save_button)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        device_bar = QtWidgets.QHBoxLayout()
        self.input_combo = QtWidgets.QComboBox()
        self.input_combo.currentIndexChanged.connect(self._set_input_device)
        self.input_combo.setFixedHeight(44)
        self.input_combo.setStyleSheet("QComboBox { padding: 6px 12px; border-radius: 10px; }")
        device_bar.addWidget(QtWidgets.QLabel("Input"))
        device_bar.addWidget(self.input_combo)
        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.currentIndexChanged.connect(self._set_output_device)
        self.output_combo.setFixedHeight(44)
        self.output_combo.setStyleSheet("QComboBox { padding: 6px 12px; border-radius: 10px; }")
        device_bar.addWidget(QtWidgets.QLabel("Output"))
        device_bar.addWidget(self.output_combo)
        main_layout.addLayout(device_bar)

        self.grid_container = GridWidget()
        self.grid_layout = QtWidgets.QVBoxLayout(self.grid_container)
        self.grid_layout.setSpacing(18)
        self.grid_layout.setContentsMargins(24, 24, 24, 24)
        self.track_widgets: List[TrackWidget] = []
        for i in range(4):
            widget = TrackWidget(i)
            widget.record_toggled.connect(self._arm_track)
            widget.mono_toggled.connect(self._toggle_track_mono)
            widget.gain_changed.connect(self._set_track_gain)
            widget.eq_requested.connect(self._edit_track_eq)
            widget.compressor_requested.connect(self._edit_track_compressor)
            widget.crop_requested.connect(self._crop_track)
            widget.move_requested.connect(self._move_track)
            self.track_widgets.append(widget)
            self.grid_layout.addWidget(widget)
        self.grid_layout.addStretch()

        self.master_section = QtWidgets.QGroupBox("Master")
        master_layout = QtWidgets.QHBoxLayout(self.master_section)
        self.master_gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.master_gain_slider.setRange(-120, 120)
        self.master_gain_slider.setValue(0)
        self.master_gain_slider.valueChanged.connect(self._set_master_gain)
        master_layout.addWidget(QtWidgets.QLabel("Gain"))
        master_layout.addWidget(self.master_gain_slider)
        master_eq_button = QtWidgets.QPushButton("EQ")
        master_eq_button.clicked.connect(self._edit_master_eq)
        master_layout.addWidget(master_eq_button)
        master_comp_button = QtWidgets.QPushButton("Comp")
        master_comp_button.clicked.connect(self._edit_master_compressor)
        master_layout.addWidget(master_comp_button)

        self.tape_deck = TapeDeckView()
        self.tape_deck.record.connect(self._toggle_record_button_on)
        self.tape_deck.stop.connect(self._stop_recording)
        self.tape_deck.play.connect(self._handle_tape_play)
        self.tape_deck.play_stopped.connect(self._handle_tape_stop)
        for idx, button in enumerate(self.tape_deck.track_buttons):
            button.toggled.connect(lambda state, i=idx: self._arm_track(i, state))
        self.stack_container = QtWidgets.QWidget()
        self.orientation_stack = QtWidgets.QStackedLayout(self.stack_container)
        self.track_area = QtWidgets.QScrollArea()
        self.track_area.setWidgetResizable(True)
        self.track_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.track_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.track_area.setWidget(self.grid_container)
        self.track_area.setStyleSheet("QScrollArea { background: transparent; }")
        self.orientation_stack.addWidget(self.track_area)
        self.orientation_stack.addWidget(self.tape_deck)
        main_layout.addWidget(self.stack_container, 1)
        self.orientation_stack.setCurrentWidget(self.track_area)
        main_layout.addWidget(self.master_section)

    def _refresh_track_views(self) -> None:
        for idx, (widget, track) in enumerate(zip(self.track_widgets, self.engine.tracks)):
            live_audio = self.engine.get_live_waveform(idx)
            widget.update_from_track(track, live_audio)

    @staticmethod
    def _apply_button_glow(button: QtWidgets.QAbstractButton, color: str, active: bool) -> None:
        if active:
            effect = QtWidgets.QGraphicsDropShadowEffect(button)
            effect.setColor(QtGui.QColor(color))
            effect.setBlurRadius(45)
            effect.setOffset(0)
            button.setGraphicsEffect(effect)
        else:
            button.setGraphicsEffect(None)

    def _handle_tape_play(self) -> None:
        if not self.play_button.isChecked():
            self.play_button.setChecked(True)
        else:
            self._toggle_playback(True)

    def _handle_tape_stop(self) -> None:
        if self.play_button.isChecked():
            self.play_button.setChecked(False)
        else:
            self._toggle_playback(False)

    def _toggle_playback(self, state: bool) -> None:
        if state:
            self._set_button_state(self.tape_deck.play_button, True)
            if self.record_button.isChecked() and not self.engine.is_recording():
                self.engine.start_recording()
            self.engine.start_playback()
            if not self.engine.is_playing():
                self._set_button_state(self.play_button, False)
                self._set_button_state(self.tape_deck.play_button, False)
                self._apply_button_glow(self.play_button, PALETTE["accent_active"], False)
                self._apply_button_glow(self.tape_deck.play_button, PALETTE["accent_active"], False)
            else:
                self._apply_button_glow(self.play_button, PALETTE["accent_active"], True)
                self._apply_button_glow(self.tape_deck.play_button, PALETTE["accent_active"], True)
        else:
            self.engine.stop_playback()
            self._apply_button_glow(self.play_button, PALETTE["accent_active"], False)
            self._apply_button_glow(self.tape_deck.play_button, PALETTE["accent_active"], False)
            self._set_button_state(self.tape_deck.play_button, False)

    # Slots ------------------------------------------------------------
    def _toggle_recording(self, state: bool) -> None:
        if state:
            self._set_button_state(self.tape_deck.record_button, True)
            self._apply_button_glow(self.record_button, PALETTE["record_active"], True)
            self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], True)
            self.engine.start_recording()
            if not self.engine.is_recording():
                self._apply_button_glow(self.record_button, PALETTE["record_active"], False)
                self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], False)
                self._set_button_state(self.record_button, False)
                self._set_button_state(self.tape_deck.record_button, False)
        else:
            self._stop_recording()

    def _stop_recording(self) -> None:
        self._apply_button_glow(self.record_button, PALETTE["record_active"], False)
        self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], False)
        self._set_button_state(self.record_button, False)
        self._set_button_state(self.tape_deck.record_button, False)
        self.engine.stop_recording()
        self._refresh_track_views()

    def _toggle_record_button_on(self) -> None:
        if not self.record_button.isChecked():
            self.record_button.setChecked(True)
        else:
            self.engine.start_recording()
            self._apply_button_glow(self.record_button, PALETTE["record_active"], True)
            self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], True)
            if not self.engine.is_recording():
                self._apply_button_glow(self.record_button, PALETTE["record_active"], False)
                self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], False)
                self._set_button_state(self.record_button, False)
                self._set_button_state(self.tape_deck.record_button, False)

    def _poll_engine_state(self) -> None:
        playing = self.engine.is_playing()
        recording = self.engine.is_recording()

        if playing:
            self._set_button_state(self.play_button, True)
            self._set_button_state(self.tape_deck.play_button, True)
            self._apply_button_glow(self.play_button, PALETTE["accent_active"], True)
            self._apply_button_glow(self.tape_deck.play_button, PALETTE["accent_active"], True)
        elif self._last_playing_state:
            self._apply_button_glow(self.play_button, PALETTE["accent_active"], False)
            self._apply_button_glow(self.tape_deck.play_button, PALETTE["accent_active"], False)
            self._set_button_state(self.play_button, False)
            self._set_button_state(self.tape_deck.play_button, False)

        if recording:
            self._set_button_state(self.record_button, True)
            self._set_button_state(self.tape_deck.record_button, True)
            self._apply_button_glow(self.record_button, PALETTE["record_active"], True)
            self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], True)
            self._refresh_track_views()
        elif self._last_recording_state:
            self._apply_button_glow(self.record_button, PALETTE["record_active"], False)
            self._apply_button_glow(self.tape_deck.record_button, PALETTE["record_active"], False)
            self._set_button_state(self.record_button, False)
            self._set_button_state(self.tape_deck.record_button, False)
            self._refresh_track_views()

        self._last_playing_state = playing
        self._last_recording_state = recording

    def _tempo_changed(self, value: int) -> None:
        self.engine.set_tempo(value)

    def _toggle_click(self, state: bool) -> None:
        self.engine.click_enabled = state

    def _set_input_device(self, index: int) -> None:
        if index < 0:
            return
        device_id, _ = self._input_devices[index]
        self.engine.set_input_device(device_id)

    def _set_output_device(self, index: int) -> None:
        if index < 0:
            return
        device_id, _ = self._output_devices[index]
        self.engine.set_output_device(device_id)

    def _export_mix(self) -> None:
        sample_rate = int(self.sample_rate_combo.currentText())
        fmt = self.format_combo.currentText()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Mix", "mix." + fmt, f"*.{fmt}")
        if file_name:
            self.engine.export(file_name, sample_rate=sample_rate, fmt=fmt)

    def _save_session(self) -> None:
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Session", "session.json", "*.json")
        if file_name:
            save_session(Path(file_name), self.engine.tracks, self.engine.tempo)

    def _arm_track(self, index: int, state: bool) -> None:
        self.engine.tracks[index].armed = state
        self._set_button_state(self.tape_deck.track_buttons[index], state)
        self._set_button_state(self.track_widgets[index].record_button, state)
        live_audio = self.engine.get_live_waveform(index)
        self.track_widgets[index].update_from_track(self.engine.tracks[index], live_audio)

    @staticmethod
    def _set_button_state(button: QtWidgets.QAbstractButton, state: bool) -> None:
        old = button.blockSignals(True)
        button.setChecked(state)
        button.blockSignals(old)

    def _toggle_track_mono(self, index: int, state: bool) -> None:
        track = self.engine.tracks[index]
        if state:
            track.convert_to_mono()
        else:
            track.convert_to_stereo()
        if track.raw_data is not None:
            track.apply_processing()
        self._refresh_track_views()

    def _set_track_gain(self, index: int, gain_db: float) -> None:
        self.engine.tracks[index].gain_db = gain_db
        if self.engine.tracks[index].data is not None:
            self.engine.tracks[index].apply_processing()
        self._refresh_track_views()

    def _edit_track_eq(self, index: int) -> None:
        track = self.engine.tracks[index]
        if not track.eq_bands:
            track.eq_bands = default_eq_bands()
        dialog = EQDialog(track.eq_bands, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            track.eq_bands = dialog.settings()
            if track.data is not None:
                track.apply_processing()
            self._refresh_track_views()

    def _edit_track_compressor(self, index: int) -> None:
        track = self.engine.tracks[index]
        dialog = CompressorDialog(track.compressor, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            track.compressor = dialog.settings()
            if track.data is not None:
                track.apply_processing()
            self._refresh_track_views()

    def _crop_track(self, index: int, start: float, end: float) -> None:
        self.engine.crop_track(index, start, end)
        self._refresh_track_views()

    def _move_track(self, source: int, dest: int) -> None:
        self.engine.move_region(source, dest)
        self._refresh_track_views()

    def _set_master_gain(self, value: int) -> None:
        self.engine.master_gain_db = value / 10

    def _edit_master_eq(self) -> None:
        if not self.engine.master_eq:
            self.engine.master_eq = default_eq_bands()
        dialog = EQDialog(self.engine.master_eq, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.engine.master_eq = dialog.settings()

    def _edit_master_compressor(self) -> None:
        dialog = CompressorDialog(self.engine.master_compressor, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.engine.master_compressor = dialog.settings()

    def _update_device_lists(self) -> None:
        self._input_devices = AudioEngine.list_input_devices()
        self._output_devices = AudioEngine.list_output_devices()
        self.input_combo.clear()
        self.output_combo.clear()
        if self._input_devices:
            for _, name in self._input_devices:
                self.input_combo.addItem(name)
            self.input_combo.setEnabled(True)
            self.input_combo.setCurrentIndex(0)
            self._set_input_device(0)
        else:
            self.input_combo.addItem("No input devices found")
            self.input_combo.setEnabled(False)
        if self._output_devices:
            for _, name in self._output_devices:
                self.output_combo.addItem(name)
            self.output_combo.setEnabled(True)
            self.output_combo.setCurrentIndex(0)
            self._set_output_device(0)
        else:
            self.output_combo.addItem("No output devices found")
            self.output_combo.setEnabled(False)
        if not self._input_devices or not self._output_devices:
            QtWidgets.QMessageBox.warning(
                self,
                "Audio devices unavailable",
                (
                    "The recorder could not find an audio input or output device. "
                    "The application will stay open, but recording and playback "
                    "require an available audio interface."
                ),
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: D401,N802
        super().resizeEvent(event)
        if self.height() > self.width():
            self.orientation_stack.setCurrentWidget(self.tape_deck)
        else:
            self.orientation_stack.setCurrentWidget(self.track_area)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(
        (
            "QWidget {{ background-color: {background}; color: {primary}; font-size: 16px; }}"
            "QLabel {{ font-size: 16px; }}"
            "QGroupBox {{ border: 2px solid rgba(91, 70, 54, 0.25); border-radius: 16px; margin-top: 12px; padding: 16px; }}"
            "QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; font-weight: 600; }}"
            "QSlider::groove:horizontal {{ height: 8px; background: rgba(91, 70, 54, 0.2); border-radius: 4px; }}"
            "QSlider::handle:horizontal {{ background: {accent_active}; width: 22px; margin: -7px 0; border-radius: 11px; }}"
            "QCheckBox {{ spacing: 10px; }}"
        ).format(**PALETTE)
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

