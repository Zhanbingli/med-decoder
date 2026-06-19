#!/usr/bin/env python3
"""
VAD Segmenter - turn a frame stream into utterance segments
===========================================================

The old pipeline transcribed fixed 2-second chunks, which cut words at
arbitrary boundaries and hurt accuracy. This segmenter instead accumulates
audio and emits a *complete utterance* whenever it detects a speech pause,
so MedASR always receives coherent audio.

It uses a simple energy (RMS) gate with hangover, which needs no model
download and runs in microseconds. Good enough for v1; can be swapped for
silero-vad later behind the same interface.

Flow:
    frames (16 kHz mono float32)  ->  Segmenter.accept(frame)
        on detected pause  ->  on_segment(utterance: np.ndarray)

Tunables:
    threshold     RMS above which a frame counts as speech
    start_ms      contiguous speech needed to open a segment
    silence_ms    contiguous silence that closes a segment
    max_ms        force-flush a segment this long (runaway protection)
    pad_ms        audio kept before/after speech to avoid clipping onsets
"""

import threading
from typing import Callable, Optional

import numpy as np

TARGET_SR = 16000
SegmentCallback = Callable[[np.ndarray], None]


class VADSegmenter:
    def __init__(
        self,
        on_segment: SegmentCallback,
        sample_rate: int = TARGET_SR,
        threshold: float = 0.012,
        start_ms: int = 150,
        silence_ms: int = 700,
        max_ms: int = 25000,
        pad_ms: int = 200,
    ):
        self.on_segment = on_segment
        self.sr = sample_rate
        self.threshold = threshold
        self.start_samples = int(sample_rate * start_ms / 1000)
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.max_samples = int(sample_rate * max_ms / 1000)
        self.pad_samples = int(sample_rate * pad_ms / 1000)

        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self._buf = []            # accumulated samples of the current segment
        self._in_speech = False
        self._speech_run = 0      # consecutive speech samples (pre-trigger)
        self._silence_run = 0     # consecutive silence samples (in speech)
        self._pre_roll = np.zeros(0, dtype=np.float32)  # padding before speech

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def accept(self, frame: np.ndarray) -> None:
        """Feed one frame of 16 kHz mono float32 audio."""
        if frame is None or frame.size == 0:
            return
        is_speech = self._rms(frame) >= self.threshold

        with self._lock:
            if not self._in_speech:
                # Keep a rolling pre-roll buffer so we don't clip word onsets.
                self._pre_roll = np.concatenate([self._pre_roll, frame])[
                    -self.pad_samples :
                ]
                if is_speech:
                    self._speech_run += frame.size
                    if self._speech_run >= self.start_samples:
                        # Open a segment, seeded with the pre-roll padding.
                        self._in_speech = True
                        self._buf = [self._pre_roll.copy()]
                        self._silence_run = 0
                else:
                    self._speech_run = 0
            else:
                self._buf.append(frame)
                if is_speech:
                    self._silence_run = 0
                else:
                    self._silence_run += frame.size
                    if self._silence_run >= self.silence_samples:
                        self._flush_locked()
                        return

                seg_len = sum(len(b) for b in self._buf)
                if seg_len >= self.max_samples:
                    self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf:
            segment = np.concatenate(self._buf).astype(np.float32)
        else:
            segment = np.zeros(0, dtype=np.float32)
        self._reset()
        if segment.size > 0:
            # Call outside the data structures but we still hold the lock; the
            # callback is expected to be cheap (enqueue). Keeps ordering simple.
            self.on_segment(segment)

    def flush(self) -> None:
        """Force-close any in-progress segment (e.g. at end of session)."""
        with self._lock:
            if self._in_speech and self._buf:
                self._flush_locked()
            else:
                self._reset()
