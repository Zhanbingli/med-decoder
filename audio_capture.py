#!/usr/bin/env python3
"""
Audio Capture - Local microphone / file audio sources
=====================================================

Replaces the old ESP32 + TCP audio path. Audio now comes from a commercial
USB microphone plugged directly into the machine (or from a WAV file for
development / testing / demo).

All sources emit 16 kHz mono float32 frames to a callback, so the rest of the
pipeline (VAD segmenter -> MedASR -> MedGemma) is agnostic to where audio
came from.

Design rule (same as the old network thread): the audio callback must stay
cheap. The sounddevice callback only enqueues raw frames; a separate pump
thread does resampling and forwards to the consumer.

Usage:
    from audio_capture import MicSource, FileSource, list_input_devices

    src = MicSource()                       # default input device
    src.start(on_frame=lambda f: ...)       # f: np.float32, 16kHz mono
    ...
    src.stop()
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

TARGET_SR = 16000  # MedASR expects 16 kHz mono
FrameCallback = Callable[[np.ndarray], None]


def list_input_devices() -> List[dict]:
    """Return the available input (capture) devices."""
    import sounddevice as sd

    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append(
                {
                    "index": idx,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_samplerate": int(dev["default_samplerate"]),
                }
            )
    return devices


class AudioSource(ABC):
    """Common interface for anything that produces 16 kHz mono float32 frames."""

    def __init__(self):
        self._callback: Optional[FrameCallback] = None
        self._running = False

    @abstractmethod
    def start(self, on_frame: FrameCallback) -> None:
        """Begin producing frames, invoking on_frame(frame) per chunk."""

    @abstractmethod
    def stop(self) -> None:
        """Stop producing frames and release resources."""

    @property
    def running(self) -> bool:
        return self._running


class MicSource(AudioSource):
    """
    Live capture from a local input device via sounddevice (PortAudio).

    Captures at the device's native rate and resamples to 16 kHz with a
    stateful soxr resampler (no inter-block artifacts).
    """

    def __init__(self, device: Optional[object] = None, frame_ms: int = 32):
        super().__init__()
        self.device = device  # name, index, or None for system default
        self.frame_ms = frame_ms
        self._stream = None
        self._raw_q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=256)
        self._pump_thread: Optional[threading.Thread] = None
        self._resampler = None
        self._src_rate = None

    def _device_samplerate(self) -> int:
        import sounddevice as sd

        info = sd.query_devices(self.device, kind="input")
        return int(info["default_samplerate"])

    def _on_audio(self, indata, frames, time_info, status):
        # PortAudio thread: keep it minimal. Mix to mono and enqueue a copy.
        if status:
            # Overflows etc. are non-fatal; drop and continue.
            pass
        mono = indata if indata.ndim == 1 else indata.mean(axis=1)
        try:
            self._raw_q.put_nowait(mono.astype(np.float32, copy=True))
        except queue.Full:
            pass  # backpressure: drop oldest by simply skipping

    def _pump(self):
        import soxr

        while self._running:
            try:
                chunk = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None:
                break
            out = self._resampler.resample_chunk(chunk)
            if len(out) and self._callback:
                self._callback(np.asarray(out, dtype=np.float32))

    def start(self, on_frame: FrameCallback) -> None:
        import sounddevice as sd
        import soxr

        self._callback = on_frame
        self._src_rate = self._device_samplerate()
        self._resampler = soxr.ResampleStream(
            self._src_rate, TARGET_SR, 1, dtype="float32"
        )
        self._running = True

        self._pump_thread = threading.Thread(target=self._pump, daemon=True)
        self._pump_thread.start()

        blocksize = int(self._src_rate * self.frame_ms / 1000)
        self._stream = sd.InputStream(
            samplerate=self._src_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=blocksize,
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._raw_q.put(None)
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=1.0)
            self._pump_thread = None


class FileSource(AudioSource):
    """
    Replay a WAV file as if it were a live mic. Used for development, tests and
    the no-hardware demo. Emits the same 16 kHz mono float32 frames as MicSource.

    Args:
        path: audio file path (any soundfile-readable format)
        realtime: if True, sleep between frames to mimic real-time pacing;
                  if False, emit as fast as possible (for tests)
        frame_ms: frame size in milliseconds
    """

    def __init__(self, path: str, realtime: bool = False, frame_ms: int = 32):
        super().__init__()
        self.path = path
        self.realtime = realtime
        self.frame_ms = frame_ms
        self._thread: Optional[threading.Thread] = None
        self.finished = threading.Event()

    def _load_16k_mono(self) -> np.ndarray:
        import soundfile as sf
        import soxr

        data, sr = sf.read(self.path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = soxr.resample(data, sr, TARGET_SR)
        return np.asarray(data, dtype=np.float32)

    def _run(self):
        audio = self._load_16k_mono()
        frame_len = int(TARGET_SR * self.frame_ms / 1000)
        for start in range(0, len(audio), frame_len):
            if not self._running:
                break
            frame = audio[start : start + frame_len]
            if self._callback:
                self._callback(frame)
            if self.realtime:
                time.sleep(self.frame_ms / 1000)
        self.finished.set()
        self._running = False

    def start(self, on_frame: FrameCallback) -> None:
        self._callback = on_frame
        self._running = True
        self.finished.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the whole file has been emitted."""
        return self.finished.wait(timeout)


if __name__ == "__main__":
    print("Available input devices:")
    try:
        for d in list_input_devices():
            print(
                f"  [{d['index']}] {d['name']} "
                f"({d['channels']}ch @ {d['default_samplerate']}Hz)"
            )
    except Exception as e:
        print(f"  Could not enumerate devices: {e}")
