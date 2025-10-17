"""Simple compressor implementation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CompressorSettings:
    threshold_db: float = -18.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_gain_db: float = 0.0


def apply_compressor(audio: np.ndarray, settings: CompressorSettings) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    sample_rate = 44100
    attack_coeff = np.exp(-1.0 / (0.001 * settings.attack_ms * sample_rate))
    release_coeff = np.exp(-1.0 / (0.001 * settings.release_ms * sample_rate))
    threshold = 10 ** (settings.threshold_db / 20)
    makeup = 10 ** (settings.makeup_gain_db / 20)
    envelope = np.zeros(audio.shape[0])
    gain = np.ones(audio.shape[0])
    for i in range(audio.shape[0]):
        peak = np.max(np.abs(audio[i]))
        if peak > envelope[i - 1] if i > 0 else 0:
            envelope[i] = attack_coeff * (envelope[i - 1] if i > 0 else 0) + (1 - attack_coeff) * peak
        else:
            envelope[i] = release_coeff * (envelope[i - 1] if i > 0 else 0) + (1 - release_coeff) * peak
        if envelope[i] > threshold:
            over = envelope[i] / threshold
            gain_reduction = over ** (1 - 1 / settings.ratio)
            gain[i] = 1 / gain_reduction
        else:
            gain[i] = 1.0
    processed = audio * gain[:, np.newaxis] * makeup
    return processed.astype(np.float32)

