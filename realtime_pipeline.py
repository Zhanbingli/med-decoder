#!/usr/bin/env python3
"""
Realtime Pipeline - Mic/File -> VAD -> MedASR -> MedGemma
========================================================

Queue-decoupled pipeline. Audio enters from a local microphone (or a WAV file
for testing) instead of the old ESP32/TCP path. Heavy inference runs on its own
worker thread, never on the audio capture thread.

Stages:
    AudioSource(frame) -> VADSegmenter -> seg_q -> ASR worker -> transcript
                                                            \-> MedGemma summary

Usage:
    python realtime_pipeline.py --list-devices
    python realtime_pipeline.py --mic                 # live microphone
    python realtime_pipeline.py --mic --device 1
    python realtime_pipeline.py --file audio.wav      # replay a file
    python realtime_pipeline.py --file               # default MedASR test wav
"""

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from audio_capture import FileSource, MicSource, list_input_devices  # noqa: E402
from vad_segmenter import VADSegmenter  # noqa: E402

try:
    from unified_model_manager import (  # noqa: E402
        UnifiedModelManager,
        create_patient_info,
        ModelStatus,
    )
except ImportError as e:
    print(f"[Error] Failed to import UnifiedModelManager: {e}")
    print("  Make sure you're running from the med_exper directory")
    sys.exit(1)

# Default MedASR test audio shipped with the model (used when --file has no arg)
_DEFAULT_TEST_WAV = (
    Path.home()
    / ".cache/huggingface/hub/models--google--medasr/snapshots"
)


def _find_default_wav() -> Optional[str]:
    if _DEFAULT_TEST_WAV.exists():
        hits = list(_DEFAULT_TEST_WAV.glob("*/test_audio.wav"))
        if hits:
            return str(hits[0])
    return None


class RealtimePipeline:
    """Real-time mic/file -> transcription -> structured note pipeline."""

    _SENTINEL = object()

    def __init__(self, verbose: bool = True, summary_template: str = "general"):
        self.verbose = verbose
        self.summary_template = summary_template
        self.manager: Optional[UnifiedModelManager] = None
        self.initialized = False

        self.seg_q: "queue.Queue" = queue.Queue(maxsize=64)
        self.segmenter: Optional[VADSegmenter] = None
        self._asr_thread: Optional[threading.Thread] = None

        self.full_transcription = ""
        self._transcript_lock = threading.Lock()
        self.session_start: Optional[float] = None

        self.stats = {"segments": 0, "asr_seconds": 0.0, "audio_seconds": 0.0}

    def log(self, message: str):
        if self.verbose:
            print(f"[Pipeline] {message}")

    # ------------------------------------------------------------------ setup
    def initialize(self) -> bool:
        print("\n" + "=" * 60)
        print("  Initializing Real-Time Pipeline")
        print("=" * 60)

        self.manager = UnifiedModelManager()
        results = self.manager.load_all(verbose=False)

        if not results.get("medasr"):
            print("  ✗ MedASR failed to load")
            return False
        print("  ✓ MedASR loaded")

        if self.manager.medgemma.status == ModelStatus.READY:
            print(f"  ✓ MedGemma ready ({self.manager.medgemma.model_name})")
        else:
            print(
                f"  ⚠ MedGemma not ready ({self.manager.medgemma.info.error_message});"
                " transcription will still run"
            )

        self.initialized = True
        print("-" * 60)
        print("  Pipeline ready")
        print("-" * 60)
        return True

    # --------------------------------------------------------------- workers
    def _on_segment(self, segment: np.ndarray):
        """VAD callback (cheap): just enqueue the utterance for ASR."""
        try:
            self.seg_q.put_nowait(segment)
        except queue.Full:
            self.log("ASR queue full, dropping a segment")

    def _asr_loop(self):
        while True:
            item = self.seg_q.get()
            if item is self._SENTINEL:
                break
            self._transcribe_segment(item)

    def _transcribe_segment(self, segment: np.ndarray):
        if not (self.manager and self.manager.medasr.is_available()):
            return
        try:
            start = time.time()
            result = self.manager.medasr.transcribe_array(segment, sample_rate=16000)
            elapsed = time.time() - start
            text = result.text.strip()

            self.stats["segments"] += 1
            self.stats["asr_seconds"] += elapsed
            self.stats["audio_seconds"] += len(segment) / 16000

            if text:
                with self._transcript_lock:
                    self.full_transcription += (" " + text) if self.full_transcription else text
                print(f"\n[Transcript] ({elapsed:.2f}s) {text}")
        except Exception as e:
            print(f"[Pipeline] ASR error: {e}")

    # --------------------------------------------------------------- session
    def _start_session(self):
        self.full_transcription = ""
        self.session_start = time.time()
        self.stats = {"segments": 0, "asr_seconds": 0.0, "audio_seconds": 0.0}
        self.segmenter = VADSegmenter(on_segment=self._on_segment)
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()
        print("\n" + "=" * 60)
        print("  Session started")
        print("=" * 60)

    def _end_session(self):
        # Flush the tail utterance, then drain the ASR queue.
        if self.segmenter:
            self.segmenter.flush()
        self.seg_q.put(self._SENTINEL)
        if self._asr_thread:
            self._asr_thread.join()

        elapsed = time.time() - self.session_start if self.session_start else 0.0
        rtf = (
            self.stats["asr_seconds"] / self.stats["audio_seconds"]
            if self.stats["audio_seconds"]
            else 0.0
        )
        print("\n" + "=" * 60)
        print("  Session ended")
        print("=" * 60)
        print(f"  Duration       : {elapsed:.1f}s")
        print(f"  Segments        : {self.stats['segments']}")
        print(f"  ASR realtime fac: {rtf:.2f}x")
        print(f"  Transcript chars: {len(self.full_transcription)}")

        if self.full_transcription.strip():
            print(f"\n[Full Transcript]\n  {self.full_transcription}\n")
            self._generate_summary(self.full_transcription)
        print("=" * 60)

    def _generate_summary(self, transcription: str):
        if not (self.manager and self.manager.medgemma.is_available()):
            print("[Pipeline] MedGemma unavailable, skipping note generation")
            return
        try:
            patient = create_patient_info(name="Patient", age=0, gender="unknown")
            start = time.time()
            record = self.manager.generate_record(
                transcription=transcription,
                patient_info=patient,
                template=self.summary_template,
            )
            print(f"\n[Outpatient Note] (generated in {time.time() - start:.1f}s)")
            for label, value in [
                ("Chief Complaint", record.chief_complaint),
                ("HPI", record.present_history),
                ("Assessment", record.assessment),
                ("Plan", record.plan),
            ]:
                if value:
                    print(f"  {label}: {value}")
        except Exception as e:
            print(f"[Pipeline] Summary error: {e}")

    # ------------------------------------------------------------------ runs
    def run_file(self, path: str):
        if not self.initialized and not self.initialize():
            return
        print(f"\nReplaying file: {path}")
        self._start_session()
        source = FileSource(path, realtime=False)
        source.start(on_frame=self.segmenter.accept)
        source.wait()
        self._end_session()

    def run_mic(self, device=None, max_seconds: Optional[float] = None):
        if not self.initialized and not self.initialize():
            return
        self._start_session()
        source = MicSource(device=device)
        source.start(on_frame=self.segmenter.accept)
        print("\n  \U0001f3a4 Recording... press Ctrl+C to stop\n")
        try:
            start = time.time()
            while True:
                time.sleep(0.2)
                if max_seconds and (time.time() - start) >= max_seconds:
                    break
        except KeyboardInterrupt:
            print("\n  Stopping...")
        finally:
            source.stop()
            self._end_session()

    def shutdown(self):
        if self.manager:
            self.manager.unload_all()


def main():
    parser = argparse.ArgumentParser(
        description="Real-time mic/file -> MedASR -> MedGemma pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mic", action="store_true", help="Capture from microphone")
    parser.add_argument(
        "--file",
        nargs="?",
        const="__default__",
        help="Replay a WAV file (no arg = MedASR test audio)",
    )
    parser.add_argument("--device", default=None, help="Input device index or name")
    parser.add_argument(
        "--seconds", type=float, default=None, help="Auto-stop mic after N seconds"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List input devices and exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        print("Input devices:")
        for d in list_input_devices():
            print(
                f"  [{d['index']}] {d['name']} "
                f"({d['channels']}ch @ {d['default_samplerate']}Hz)"
            )
        return

    pipeline = RealtimePipeline()

    if args.file is not None:
        path = args.file
        if path == "__default__":
            path = _find_default_wav()
            if not path:
                print("No default test audio found; pass --file <path>")
                return
        pipeline.run_file(path)
    elif args.mic:
        device = args.device
        if device is not None and str(device).isdigit():
            device = int(device)
        pipeline.run_mic(device=device, max_seconds=args.seconds)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
