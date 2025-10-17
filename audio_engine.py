"""Core audio engine for the four-track recorder."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd

from compressor import CompressorSettings, apply_compressor
from eq import EQBand, apply_eq
from metronome import generate_click_track


@dataclass
class Track:
    """Represents a single audio track."""

    name: str
    sample_rate: int
    max_seconds: float
    armed: bool = False
    stereo: bool = True
    eq_bands: List[EQBand] = field(default_factory=list)
    compressor: Optional[CompressorSettings] = None
    gain_db: float = 0.0
    data: np.ndarray | None = None
    raw_data: np.ndarray | None = None

    def set_data(self, audio: np.ndarray, sample_rate: Optional[int] = None) -> None:
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        if not self.stereo and audio.shape[1] == 2:
            audio = audio.mean(axis=1, keepdims=True)
        if self.stereo and audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        max_samples = int(self.max_seconds * self.sample_rate)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
        processed = audio.astype(np.float32)
        self.raw_data = processed.copy()
        self.data = processed

    def toggle_arm(self) -> None:
        self.armed = not self.armed

    def convert_to_mono(self) -> None:
        self.stereo = False
        if self.raw_data is not None:
            self.raw_data = self.raw_data.mean(axis=1, keepdims=True)
            self.data = self.raw_data.copy()

    def convert_to_stereo(self) -> None:
        self.stereo = True
        if self.raw_data is not None and self.raw_data.shape[1] == 1:
            self.raw_data = np.repeat(self.raw_data, 2, axis=1)
            self.data = self.raw_data.copy()

    def apply_processing(self) -> None:
        if self.raw_data is None:
            return
        processed = self.raw_data.copy()
        if self.eq_bands:
            processed = apply_eq(processed, self.sample_rate, self.eq_bands)
        if self.compressor:
            processed = apply_compressor(processed, self.compressor)
        if self.gain_db:
            gain = 10 ** (self.gain_db / 20)
            processed *= gain
        self.data = processed


class AudioEngine:
    """High-level engine that records, stores, and renders tracks."""

    def __init__(self, tempo: int = 120, max_bars: int = 100, time_signature: Tuple[int, int] = (4, 4)) -> None:
        self.tempo = tempo
        self.max_bars = max_bars
        self.time_signature = time_signature
        self.sample_rate = 44100
        self.click_enabled = True
        beats_per_bar = time_signature[0]
        seconds_per_beat = 60.0 / tempo
        self.max_seconds = max_bars * beats_per_bar * seconds_per_beat
        self.tracks: List[Track] = [
            Track(name=f"Track {i+1}", sample_rate=self.sample_rate, max_seconds=self.max_seconds)
            for i in range(4)
        ]
        self._live_lock = threading.Lock()
        self._live_waveforms: List[np.ndarray | None] = [None] * len(self.tracks)
        self.master_eq: List[EQBand] = []
        self.master_compressor: Optional[CompressorSettings] = None
        self.master_gain_db: float = 0.0
        self.input_device: Optional[int] = None
        self.output_device: Optional[int] = None
        self._record_buffer: List[np.ndarray] = []
        self._record_thread: Optional[threading.Thread] = None
        self._record_stop = threading.Event()
        self._is_recording = False
        self._play_thread: Optional[threading.Thread] = None
        self._is_playing = False

    # Device management -------------------------------------------------
    @staticmethod
    def list_input_devices() -> List[Tuple[int, str]]:
        devices: List[Tuple[int, str]] = []
        try:
            queried = sd.query_devices()
        except Exception as exc:  # pragma: no cover - depends on host audio stack
            print("Unable to query input devices:", exc)
            return devices
        for idx, device in enumerate(queried):
            if device["max_input_channels"] >= 1:
                devices.append((idx, device["name"]))
        return devices

    @staticmethod
    def list_output_devices() -> List[Tuple[int, str]]:
        devices: List[Tuple[int, str]] = []
        try:
            queried = sd.query_devices()
        except Exception as exc:  # pragma: no cover - depends on host audio stack
            print("Unable to query output devices:", exc)
            return devices
        for idx, device in enumerate(queried):
            if device["max_output_channels"] >= 1:
                devices.append((idx, device["name"]))
        return devices

    def set_input_device(self, device_id: int) -> None:
        try:
            info = sd.query_devices(device_id)
        except Exception as exc:  # pragma: no cover - depends on host audio stack
            print("Unable to select input device:", exc)
            return
        if info["max_input_channels"] < 1:
            raise ValueError("Selected device has no input channels")
        self.input_device = device_id

    def set_output_device(self, device_id: int) -> None:
        try:
            info = sd.query_devices(device_id)
        except Exception as exc:  # pragma: no cover - depends on host audio stack
            print("Unable to select output device:", exc)
            return
        if info["max_output_channels"] < 1:
            raise ValueError("Selected device has no output channels")
        self.output_device = device_id

    def set_tempo(self, tempo: int) -> None:
        self.tempo = tempo
        beats_per_bar = self.time_signature[0]
        seconds_per_beat = 60.0 / tempo
        self.max_seconds = self.max_bars * beats_per_bar * seconds_per_beat
        for track in self.tracks:
            track.max_seconds = self.max_seconds

    # Recording ---------------------------------------------------------
    def start_recording(self) -> None:
        if self._is_recording:
            return
        self._record_buffer = []
        self._record_stop.clear()
        self._is_recording = True
        armed_indices = [idx for idx, track in enumerate(self.tracks) if track.armed]
        if not armed_indices:
            self._is_recording = False
            return
        with self._live_lock:
            for idx in range(len(self._live_waveforms)):
                self._live_waveforms[idx] = None

        def _record_worker() -> None:
            channels = 2 if any(track.stereo for track in self.tracks) else 1
            try:
                if self.input_device is not None:
                    device_info = sd.query_devices(self.input_device)
                else:
                    device_info = sd.query_devices(sd.default.device[0])
                channels = min(channels, max(1, int(device_info["max_input_channels"])), 2)
            except Exception:  # pragma: no cover - device lookup failure fallback
                channels = min(channels, 2)
            frames_per_buffer = 1024
            duration_limit = int(self.max_seconds * self.sample_rate)
            recorded = []

            def callback(indata, frames, time, status):  # noqa: ANN001
                if status:
                    print("Recording status:", status)
                recorded.append(indata.copy())
                total_frames = sum(chunk.shape[0] for chunk in recorded)
                with self._live_lock:
                    for track_index in armed_indices:
                        track = self.tracks[track_index]
                        chunk = indata.astype(np.float32)
                        if not track.stereo and chunk.shape[1] == 2:
                            chunk = chunk.mean(axis=1, keepdims=True)
                        elif track.stereo and chunk.shape[1] == 1:
                            chunk = np.repeat(chunk, 2, axis=1)
                        existing = self._live_waveforms[track_index]
                        if existing is None:
                            preview = chunk.copy()
                        else:
                            preview = np.concatenate((existing, chunk), axis=0)
                        max_samples = int(track.max_seconds * track.sample_rate)
                        if preview.shape[0] > max_samples:
                            preview = preview[-max_samples:]
                        self._live_waveforms[track_index] = preview
                if total_frames >= duration_limit or self._record_stop.is_set():
                    raise sd.CallbackStop()

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=channels,
                callback=callback,
                device=self.input_device,
                blocksize=frames_per_buffer,
            ):
                while not self._record_stop.is_set():
                    sd.sleep(100)

            if recorded:
                audio = np.concatenate(recorded, axis=0)
            else:
                audio = np.zeros((0, channels), dtype=np.float32)
            self._record_buffer = audio
            self._is_recording = False

        self._record_thread = threading.Thread(target=_record_worker, daemon=True)
        self._record_thread.start()

    def stop_recording(self) -> None:
        if not self._is_recording:
            return
        self._record_stop.set()
        if self._record_thread:
            self._record_thread.join()
        audio = self._record_buffer
        if audio is None or audio.size == 0:
            with self._live_lock:
                for idx in range(len(self._live_waveforms)):
                    self._live_waveforms[idx] = None
            return
        for idx, track in enumerate(self.tracks):
            if track.armed:
                track.set_data(audio, self.sample_rate)
                track.apply_processing()
            with self._live_lock:
                self._live_waveforms[idx] = None

    def get_live_waveform(self, index: int) -> np.ndarray | None:
        with self._live_lock:
            if not (0 <= index < len(self._live_waveforms)):
                return None
            data = self._live_waveforms[index]
            if data is None:
                return None
            return data.copy()

    # Playback ----------------------------------------------------------
    def _render_mixdown(self) -> np.ndarray:
        active_tracks = [t.data for t in self.tracks if t.data is not None]
        if not active_tracks:
            return np.zeros((0, 2), dtype=np.float32)
        max_len = max(t.shape[0] for t in active_tracks)
        mix = np.zeros((max_len, 2), dtype=np.float32)
        for track in self.tracks:
            if track.data is None:
                continue
            data = track.data
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            padded = np.zeros_like(mix)
            padded[: data.shape[0], : data.shape[1]] = data
            mix[: data.shape[0]] += padded[: data.shape[0]]
        if self.click_enabled:
            click = generate_click_track(max_len / self.sample_rate, self.tempo, self.sample_rate)
            mix[: click.shape[0]] += click[: mix.shape[0]]
        if self.master_eq:
            mix = apply_eq(mix, self.sample_rate, self.master_eq)
        if self.master_compressor:
            mix = apply_compressor(mix, self.master_compressor)
        if self.master_gain_db:
            mix *= 10 ** (self.master_gain_db / 20)
        mix = np.clip(mix, -1.0, 1.0)
        return mix

    def start_playback(self) -> None:
        if self._is_playing:
            return
        mix = self._render_mixdown()
        if mix.size == 0:
            return
        self._is_playing = True

        def _play_worker() -> None:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                device=self.output_device,
            ) as stream:
                stream.write(mix)
            self._is_playing = False

        self._play_thread = threading.Thread(target=_play_worker, daemon=True)
        self._play_thread.start()

    def stop_playback(self) -> None:
        if not self._is_playing:
            return
        if self._play_thread:
            self._play_thread.join()
        self._is_playing = False

    def is_recording(self) -> bool:
        return self._is_recording

    def is_playing(self) -> bool:
        return self._is_playing

    # Editing -----------------------------------------------------------
    def crop_track(self, track_index: int, start_sec: float, end_sec: float) -> None:
        track = self.tracks[track_index]
        if track.raw_data is None:
            return
        start = int(max(start_sec, 0) * track.sample_rate)
        end = int(min(end_sec, track.raw_data.shape[0] / track.sample_rate) * track.sample_rate)
        track.raw_data = track.raw_data[start:end]
        track.apply_processing()

    def move_region(self, source_index: int, dest_index: int) -> None:
        source = self.tracks[source_index]
        dest = self.tracks[dest_index]
        if source.raw_data is None:
            return
        dest.set_data(source.raw_data.copy(), source.sample_rate)
        dest.apply_processing()

    # Export ------------------------------------------------------------
    def export(self, path: str, sample_rate: int = 44100, fmt: str = "wav") -> None:
        import soundfile as sf

        mix = self._render_mixdown()
        if mix.size == 0:
            raise RuntimeError("Nothing to export")
        if sample_rate != self.sample_rate:
            mix = self._resample(mix, sample_rate)
        if fmt.lower() not in {"wav", "mp3"}:
            raise ValueError("Unsupported format")
        subtype = "PCM_16" if fmt.lower() == "wav" else "PCM_16"
        sf.write(path, mix, sample_rate, subtype=subtype, format=fmt.upper())

    @staticmethod
    def _resample(data: np.ndarray, target_rate: int) -> np.ndarray:
        from scipy.signal import resample

        num_samples = int(len(data) * target_rate / 44100)
        resampled = resample(data, num_samples)
        return resampled.astype(np.float32)

