"""Simple multi-band equalizer implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class EQBand:
    """Represents a single EQ band."""

    low_freq: float
    high_freq: float
    gain_db: float


def default_eq_bands() -> List[EQBand]:
    """Return a default set of 8 bands covering the audible spectrum."""
    return [
        EQBand(20, 60, 0.0),
        EQBand(60, 150, 0.0),
        EQBand(150, 400, 0.0),
        EQBand(400, 1000, 0.0),
        EQBand(1000, 2400, 0.0),
        EQBand(2400, 6000, 0.0),
        EQBand(6000, 12000, 0.0),
        EQBand(12000, 20000, 0.0),
    ]


def apply_eq(audio: np.ndarray, sample_rate: int, bands: Iterable[EQBand]) -> np.ndarray:
    """Apply EQ bands by scaling FFT bins."""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    processed = np.zeros_like(audio)
    freqs = np.fft.rfftfreq(audio.shape[0], d=1.0 / sample_rate)
    band_scales = np.ones_like(freqs)
    for band in bands:
        gain = 10 ** (band.gain_db / 20)
        mask = (freqs >= band.low_freq) & (freqs < band.high_freq)
        band_scales[mask] *= gain
    for ch in range(audio.shape[1]):
        spectrum = np.fft.rfft(audio[:, ch])
        spectrum *= band_scales
        processed[:, ch] = np.fft.irfft(spectrum, n=audio.shape[0]).real
    return processed.astype(np.float32)

