"""Metronome generator for click track."""
from __future__ import annotations

import numpy as np


def generate_click_track(duration_sec: float, tempo: int, sample_rate: int, beats_per_bar: int = 4) -> np.ndarray:
    beat_interval = 60.0 / tempo
    total_samples = int(duration_sec * sample_rate)
    click = np.zeros((total_samples, 2), dtype=np.float32)
    beat_samples = int(beat_interval * sample_rate)
    click_length = int(0.02 * sample_rate)
    high_beep = 0.9
    low_beep = 0.6
    current_sample = 0
    beat_count = 0
    while current_sample < total_samples:
        amplitude = high_beep if beat_count % beats_per_bar == 0 else low_beep
        end_sample = min(current_sample + click_length, total_samples)
        envelope = np.linspace(1, 0, end_sample - current_sample)
        click[current_sample:end_sample] += amplitude * envelope[:, None]
        current_sample += beat_samples
        beat_count += 1
    return click

