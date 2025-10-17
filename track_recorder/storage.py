"""Session storage utilities."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

from .audio_engine import Track
from .compressor import CompressorSettings
from .eq import EQBand


class SessionEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, EQBand):
            return asdict(obj)
        if isinstance(obj, CompressorSettings):
            return asdict(obj)
        return super().default(obj)


def save_session(path: Path, tracks: List[Track], tempo: int) -> None:
    payload = {
        "tempo": tempo,
        "tracks": [
            {
                "name": track.name,
                "sample_rate": track.sample_rate,
                "armed": track.armed,
                "stereo": track.stereo,
                "eq_bands": [asdict(b) for b in track.eq_bands],
                "compressor": asdict(track.compressor) if track.compressor else None,
                "gain_db": track.gain_db,
                "data": track.data.tolist() if track.data is not None else None,
                "raw_data": track.raw_data.tolist() if track.raw_data is not None else None,
            }
            for track in tracks
        ],
    }
    path.write_text(json.dumps(payload, cls=SessionEncoder))

