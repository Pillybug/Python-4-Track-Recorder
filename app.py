"""PyQt6 user interface for the four-track recorder."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from PyQt6 import QtCore, QtGui, QtWidgets

from audio_engine import AudioEngine
from compressor import CompressorSettings
from eq import EQBand, default_eq_bands
from storage import save_session

PALETTE = {
    "background": "#fbffc1",
    "primary": "#a16317",
    "accent": "#d4973a",
    "record": "#d85555",
    "record_arm": "#f68787",
    "highlight": "#fde488",
}


class GridWidget(QtWidgets.QWidget):
    """Widget that paints a faint grid background."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: D401,N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(PALETTE["background"]))
        pen = QtGui.QPen(QtGui.QColor(PALETTE["highlight"]))
        pen.setWidth(1)
        painter.setPen(pen)
        spacing = 30
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)
        super().paintEvent(event)


class EQDialog(QtWidgets.QDialog):
    def __init__(self, bands: List[EQBand], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("8-Band EQ")
        self._sliders: List[QtWidgets.QSlider] = []
        layout = QtWidgets.QGridLayout(self)
        for idx, band in enumerate(bands):
            label = QtWidgets.QLabel(f"{int(band.low_freq)}-{int(band.high_freq)} Hz")
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(int(band.gain_db))
            slider.setTickInterval(3)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksLeft)
            layout.addWidget(label, 0, idx)
            layout.addWidget(slider, 1, idx)
            self._sliders.append(slider)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box, 2, 0, 1, len(bands))

    def gains(self) -> List[float]:
        return [slider.value() for slider in self._sliders]


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
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(f"Track {index + 1}")
        self.title.setStyleSheet(f"color: {PALETTE['primary']}; font-weight: bold;")
        header.addWidget(self.title)
        self.record_button = QtWidgets.QPushButton("Arm")
        self.record_button.setCheckable(True)
        self.record_button.setStyleSheet(f"background-color: {PALETTE['record']}; color: white; font-weight: bold;")
        self.record_button.toggled.connect(self._on_record_toggled)
        header.addWidget(self.record_button)
        self.mono_button = QtWidgets.QPushButton("Stereo")
        self.mono_button.setCheckable(True)
        self.mono_button.toggled.connect(self._on_mono_toggled)
        header.addWidget(self.mono_button)
        layout.addLayout(header)

        control_layout = QtWidgets.QHBoxLayout()
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-120, 120)
        self.gain_slider.setValue(0)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)
        control_layout.addWidget(QtWidgets.QLabel("Gain (dB)"))
        control_layout.addWidget(self.gain_slider)
        layout.addLayout(control_layout)

        processing_layout = QtWidgets.QHBoxLayout()
        self.eq_button = QtWidgets.QPushButton("EQ")
        self.eq_button.clicked.connect(lambda: self.eq_requested.emit(self.index))
        processing_layout.addWidget(self.eq_button)
        self.comp_button = QtWidgets.QPushButton("Comp")
        self.comp_button.clicked.connect(lambda: self.compressor_requested.emit(self.index))
        processing_layout.addWidget(self.comp_button)
        layout.addLayout(processing_layout)

        edit_layout = QtWidgets.QHBoxLayout()
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setRange(0, 9999)
        self.start_spin.setSuffix(" s")
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setRange(0, 9999)
        self.end_spin.setSuffix(" s")
        crop_button = QtWidgets.QPushButton("Crop")
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
        move_button = QtWidgets.QPushButton("Move")
        move_button.clicked.connect(self._emit_move)
        move_layout.addWidget(self.move_combo)
        move_layout.addWidget(move_button)
        layout.addLayout(move_layout)

    def _on_record_toggled(self, state: bool) -> None:
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

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("4-Track Tape Deck")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: 20px; color: {PALETTE['primary']}; font-weight: bold;")
        layout.addWidget(title)
        self.track_buttons: List[QtWidgets.QPushButton] = []
        for i in range(4):
            button = QtWidgets.QPushButton(f"Track {i + 1} Arm")
            button.setCheckable(True)
            button.setStyleSheet(
                f"background-color: {PALETTE['record_arm']}; color: white; font-weight: bold; padding: 12px;"
            )
            layout.addWidget(button)
            self.track_buttons.append(button)
        controls = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setStyleSheet(
            f"background-color: {PALETTE['accent']}; color: white; font-weight: bold; padding: 12px 24px;"
        )
        self.play_button.clicked.connect(self.play.emit)
        controls.addWidget(self.play_button)
        self.record_button = QtWidgets.QPushButton("Record")
        self.record_button.setCheckable(True)
        self.record_button.setStyleSheet(
            f"background-color: {PALETTE['record']}; color: white; font-weight: bold; padding: 12px 24px;"
        )
        self.record_button.toggled.connect(self._toggle_record)
        controls.addWidget(self.record_button)
        layout.addLayout(controls)

    def _toggle_record(self, state: bool) -> None:
        if state:
            self.record.emit()
        else:
            self.stop.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Python 4-Track Recorder")
        self.resize(1200, 800)
        self.engine = AudioEngine()
        self._build_ui()
        self._update_device_lists()

    # UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top_bar = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setStyleSheet(
            f"background-color: {PALETTE['accent']}; color: white; font-weight: bold; padding: 8px 16px;"
        )
        self.play_button.clicked.connect(self.engine.start_playback)
        top_bar.addWidget(self.play_button)
        self.record_button = QtWidgets.QPushButton("Record")
        self.record_button.setCheckable(True)
        self.record_button.setStyleSheet(
            f"background-color: {PALETTE['record']}; color: white; font-weight: bold; padding: 8px 16px;"
        )
        self.record_button.toggled.connect(self._toggle_recording)
        top_bar.addWidget(self.record_button)
        self.tempo_spin = QtWidgets.QSpinBox()
        self.tempo_spin.setRange(40, 240)
        self.tempo_spin.setValue(self.engine.tempo)
        self.tempo_spin.valueChanged.connect(self._tempo_changed)
        top_bar.addWidget(QtWidgets.QLabel("Tempo"))
        top_bar.addWidget(self.tempo_spin)
        self.click_checkbox = QtWidgets.QCheckBox("Click Track")
        self.click_checkbox.setChecked(True)
        self.click_checkbox.toggled.connect(self._toggle_click)
        top_bar.addWidget(self.click_checkbox)
        self.sample_rate_combo = QtWidgets.QComboBox()
        self.sample_rate_combo.addItems(["44100", "48000"])
        top_bar.addWidget(QtWidgets.QLabel("Export Rate"))
        top_bar.addWidget(self.sample_rate_combo)
        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItems(["wav", "mp3"])
        top_bar.addWidget(QtWidgets.QLabel("Format"))
        top_bar.addWidget(self.format_combo)
        export_button = QtWidgets.QPushButton("Export Mix")
        export_button.clicked.connect(self._export_mix)
        top_bar.addWidget(export_button)
        save_button = QtWidgets.QPushButton("Save Session")
        save_button.clicked.connect(self._save_session)
        top_bar.addWidget(save_button)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        device_bar = QtWidgets.QHBoxLayout()
        self.input_combo = QtWidgets.QComboBox()
        self.input_combo.currentIndexChanged.connect(self._set_input_device)
        device_bar.addWidget(QtWidgets.QLabel("Input"))
        device_bar.addWidget(self.input_combo)
        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.currentIndexChanged.connect(self._set_output_device)
        device_bar.addWidget(QtWidgets.QLabel("Output"))
        device_bar.addWidget(self.output_combo)
        main_layout.addLayout(device_bar)

        self.grid_container = GridWidget()
        self.grid_layout = QtWidgets.QVBoxLayout(self.grid_container)
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
        self.tape_deck.play.connect(self.engine.start_playback)
        for idx, button in enumerate(self.tape_deck.track_buttons):
            button.toggled.connect(lambda state, i=idx: self._arm_track(i, state))
        self.stack_container = QtWidgets.QWidget()
        self.orientation_stack = QtWidgets.QStackedLayout(self.stack_container)
        self.orientation_stack.addWidget(self.grid_container)
        self.orientation_stack.addWidget(self.tape_deck)
        main_layout.addWidget(self.stack_container, 1)
        self.orientation_stack.setCurrentWidget(self.grid_container)
        main_layout.addWidget(self.master_section)

    # Slots ------------------------------------------------------------
    def _toggle_recording(self, state: bool) -> None:
        if state:
            self.engine.start_recording()
        else:
            self._stop_recording()

    def _stop_recording(self) -> None:
        self.record_button.setChecked(False)
        self._set_button_state(self.tape_deck.record_button, False)
        self.engine.stop_recording()

    def _toggle_record_button_on(self) -> None:
        if not self.record_button.isChecked():
            self.record_button.setChecked(True)
        else:
            self.engine.start_recording()

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

    def _set_track_gain(self, index: int, gain_db: float) -> None:
        self.engine.tracks[index].gain_db = gain_db
        if self.engine.tracks[index].data is not None:
            self.engine.tracks[index].apply_processing()

    def _edit_track_eq(self, index: int) -> None:
        track = self.engine.tracks[index]
        if not track.eq_bands:
            track.eq_bands = default_eq_bands()
        dialog = EQDialog(track.eq_bands, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            gains = dialog.gains()
            for band, gain in zip(track.eq_bands, gains):
                band.gain_db = gain
            if track.data is not None:
                track.apply_processing()

    def _edit_track_compressor(self, index: int) -> None:
        track = self.engine.tracks[index]
        dialog = CompressorDialog(track.compressor, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            track.compressor = dialog.settings()
            if track.data is not None:
                track.apply_processing()

    def _crop_track(self, index: int, start: float, end: float) -> None:
        self.engine.crop_track(index, start, end)

    def _move_track(self, source: int, dest: int) -> None:
        self.engine.move_region(source, dest)

    def _set_master_gain(self, value: int) -> None:
        self.engine.master_gain_db = value / 10

    def _edit_master_eq(self) -> None:
        if not self.engine.master_eq:
            self.engine.master_eq = default_eq_bands()
        dialog = EQDialog(self.engine.master_eq, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            gains = dialog.gains()
            for band, gain in zip(self.engine.master_eq, gains):
                band.gain_db = gain

    def _edit_master_compressor(self) -> None:
        dialog = CompressorDialog(self.engine.master_compressor, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.engine.master_compressor = dialog.settings()

    def _update_device_lists(self) -> None:
        self._input_devices = AudioEngine.list_input_devices()
        self._output_devices = AudioEngine.list_output_devices()
        self.input_combo.clear()
        self.output_combo.clear()
        for _, name in self._input_devices:
            self.input_combo.addItem(name)
        for _, name in self._output_devices:
            self.output_combo.addItem(name)
        if self._input_devices:
            self.input_combo.setCurrentIndex(0)
            self._set_input_device(0)
        if self._output_devices:
            self.output_combo.setCurrentIndex(0)
            self._set_output_device(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: D401,N802
        super().resizeEvent(event)
        if self.height() > self.width():
            self.orientation_stack.setCurrentWidget(self.tape_deck)
        else:
            self.orientation_stack.setCurrentWidget(self.grid_container)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(
        f"QWidget {{ background-color: {PALETTE['background']}; color: {PALETTE['primary']}; }}"
        "QPushButton { border-radius: 4px; }"
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

