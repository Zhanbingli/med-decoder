#!/usr/bin/env python3
"""
CardioVoice Quick Demo Script
=============================

Demonstrates the core pipeline (MedASR transcription + MedGemma note
generation) without any hardware, using the real UnifiedModelManager.

Modes:
    python demo.py --mode llm        # note generation from a sample transcript
    python demo.py --mode asr        # transcribe the bundled MedASR test audio
    python demo.py --mode full       # transcribe audio -> generate note
    python demo.py --mode info       # show model status only

Requirements:
    - Ollama running (ollama serve) with an LLM available
      (default qwen3.5:9b; override with CARDIOVOICE_LLM_MODEL)
    - conda env `medgemma` (see CLAUDE.md)
"""

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from unified_model_manager import (  # noqa: E402
    UnifiedModelManager,
    create_patient_info,
    ModelStatus,
)


def _default_test_wav():
    hits = glob.glob(
        str(
            Path.home()
            / ".cache/huggingface/hub/models--google--medasr/snapshots/*/test_audio.wav"
        )
    )
    return hits[0] if hits else None


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


SAMPLE_TRANSCRIPTION = """
Doctor: Good morning, Mr. Johnson. What brings you in today?
Patient: I've been having chest pain for the past three days.
Doctor: Can you describe the pain? Where is it located?
Patient: It's in the center of my chest, and it sometimes radiates to my left arm.
Doctor: How would you rate the pain on a scale of 1 to 10?
Patient: About a 6. It gets worse when I exert myself.
Doctor: Any associated symptoms?
Patient: Yes, I get short of breath when I walk up stairs.
Doctor: Any history of heart disease in your family?
Patient: My father had a heart attack at age 55.
"""


def _print_record(record):
    for label, value in [
        ("Chief Complaint", record.chief_complaint),
        ("History of Present Illness", record.present_history),
        ("Past Medical History", record.past_history),
        ("Cardiovascular Exam", record.cardiovascular_exam),
        ("ECG Findings", record.ecg_findings),
        ("Assessment", record.assessment),
        ("Plan", record.plan),
    ]:
        if value:
            print(f"\n{label}:\n  {value}")


def demo_info(manager):
    print_header("Model Status")
    manager.load_all(verbose=True)
    for name, info in manager.get_status().items():
        print(f"  {name:10s}: {info.status.value} "
              f"(backend={info.backend.value}, load={info.load_time:.1f}s)")


def demo_llm(manager):
    print_header("CardioVoice: MedGemma Note Generation")
    if not manager.medgemma.load():
        print("  ✗ LLM unavailable. Is Ollama running? "
              f"({manager.medgemma.info.error_message})")
        return
    print(f"  ✓ Using model: {manager.medgemma.model_name}")
    print("\nSample transcription:")
    print(SAMPLE_TRANSCRIPTION)

    patient = create_patient_info(name="John Johnson", age=58, gender="male")
    print("Generating cardiology note...")
    start = time.time()
    record = manager.generate_record(
        transcription=SAMPLE_TRANSCRIPTION, patient_info=patient, template="cardiology"
    )
    print(f"✓ Generated in {time.time() - start:.1f}s")
    _print_record(record)


def demo_asr(manager):
    print_header("CardioVoice: MedASR Transcription")
    wav = _default_test_wav()
    if not wav:
        print("  ✗ No bundled MedASR test audio found.")
        return
    if not manager.medasr.load():
        print("  ✗ MedASR failed to load")
        return
    print(f"  Transcribing: {wav}")
    start = time.time()
    result = manager.transcribe(wav)
    print(f"  ✓ Done in {time.time() - start:.1f}s "
          f"({result.audio_duration:.1f}s audio)")
    print(f"\nTranscript:\n  {result.text}")
    return result.text


def demo_full(manager):
    print_header("CardioVoice: Full Pipeline (Audio -> Note)")
    manager.load_all(verbose=True)
    text = demo_asr(manager)
    if not text:
        return
    if not manager.medgemma.is_available():
        print("\n  ⚠ LLM unavailable, skipping note generation")
        return
    patient = create_patient_info(name="Patient", age=0, gender="unknown")
    print("\nGenerating structured note...")
    start = time.time()
    record = manager.generate_record(
        transcription=text, patient_info=patient, template="general"
    )
    print(f"✓ Generated in {time.time() - start:.1f}s")
    _print_record(record)


def main():
    parser = argparse.ArgumentParser(
        description="CardioVoice Quick Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "asr", "llm", "info"],
        default="llm",
        help="Demo mode (default: llm)",
    )
    args = parser.parse_args()

    print("\n  CardioVoice - Real-Time Cardiology Documentation")
    print("  Mic -> MedASR -> MedGemma -> structured note\n")

    manager = UnifiedModelManager()
    try:
        if args.mode == "info":
            demo_info(manager)
        elif args.mode == "llm":
            demo_llm(manager)
        elif args.mode == "asr":
            demo_asr(manager)
        elif args.mode == "full":
            demo_full(manager)
        print_header("Demo Complete")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        print("Troubleshooting: ensure Ollama is running and you're in the "
              "`medgemma` conda env.")
        sys.exit(1)


if __name__ == "__main__":
    main()
