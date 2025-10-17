# Python 4-Track Recorder

This project implements a desktop-focused four-track recorder inspired by the original mobile concept.  
The application is written in Python and powered by PyQt6 for the interface, `sounddevice` for audio I/O, and
NumPy-based signal processing for equalization, compression, and mixing.

## Features

- Four simultaneous tracks with record arm toggles, gain, EQ, compressor, mono/stereo conversion, and clip editing.
- Tempo control up to 100 bars with an optional click track/metronome.
- Master section with global EQ, compressor, and gain.
- Tape deck inspired layout when the window is taller than it is wide.
- Support for built-in device audio and two-channel USB interfaces such as the SSL2+.
- Export options for WAV or MP3 at 44.1 kHz or 48 kHz.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The script depends on native audio drivers accessible through PortAudio (via `sounddevice`). Ensure your system has
appropriate audio device permissions. MP3 export support relies on the version of `libsndfile` available on your
platform.

## Running the app

Launch the recorder with:

```bash
python -m track_recorder.app
```

Connect your interface (or use your system microphone), select the input/output devices from the drop-down menus, set a tempo,
arm one or more tracks, and press **Record**. Press **Play** to review your takes, crop and move regions between tracks,
and export when satisfied.

