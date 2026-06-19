#!/usr/bin/env python3
"""
Audio preprocessing for ASR
===========================

Cheap, safe front-end cleanup applied to 16 kHz mono float32 audio before
MedASR sees it. "Garbage in, garbage out" — fixing the signal is often a bigger
accuracy lever than tweaking the model.

Steps:
  1. DC offset removal.
  2. High-pass filter (~80 Hz) to drop HVAC hum, table rumble, handling noise.
     Speech energy lives above this, so it's safe.
  3. Peak normalization to a consistent level, so quiet recordings get good SNR.

All steps are conservative and meaning-preserving. Near-silent input is left
alone (so normalization doesn't blow up noise).
"""

import numpy as np
from scipy.signal import butter, sosfilt


def preprocess(
    audio: np.ndarray,
    sr: int = 16000,
    highpass_hz: float = 80.0,
    target_peak: float = 0.97,
    silence_rms: float = 1e-3,
) -> np.ndarray:
    """Return cleaned 16 kHz mono float32 audio."""
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return x

    # 1) DC offset
    x = x - float(np.mean(x))

    # Leave near-silent buffers untouched (avoid amplifying noise).
    rms = float(np.sqrt(np.mean(x**2)))
    if rms < silence_rms:
        return x

    # 2) High-pass (2nd-order Butterworth)
    sos = butter(2, highpass_hz / (sr / 2.0), btype="highpass", output="sos")
    x = sosfilt(sos, x).astype(np.float32)

    # 3) Peak normalization with a little headroom
    peak = float(np.max(np.abs(x)))
    if peak > 1e-4:
        x = (x * (target_peak / peak)).astype(np.float32)

    return x
