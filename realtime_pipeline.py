#!/usr/bin/env python3
"""
Realtime Pipeline - MedASR + MedGemma Integration
==================================================

Real-time audio processing pipeline that:
1. Receives audio chunks from audio_receiver
2. Transcribes audio using MedASR
3. Generates summary using MedGemma (via Ollama)
4. Displays results in real-time

Usage:
    python realtime_pipeline.py --test
    python realtime_pipeline.py --mode live
"""

import sys
import time
import threading
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from collections import deque

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    from unified_model_manager import UnifiedModelManager, create_patient_info, ModelStatus
except ImportError as e:
    print(f"[Error] Failed to import UnifiedModelManager: {e}")
    print("  Make sure you're running from the med_exper directory")
    sys.exit(1)


@dataclass
class ProcessingResult:
    """Result from processing audio chunk"""
    text: str
    confidence: float
    processing_time: float
    is_final: bool = False


class RealtimePipeline:
    """
    Real-time audio processing pipeline
    """

    def __init__(self, verbose=True):
        """
        Initialize the pipeline

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.manager = None
        self.initialized = False
        self.running = False

        # Audio buffer for accumulating chunks
        self.audio_buffer = bytes()
        self.buffer_lock = threading.Lock()

        # Processing state
        self.session_active = False
        self.session_start = None
        self.full_transcription = ""

        # Results queue for display
        self.results_queue = deque(maxlen=100)

        # Statistics
        self.stats = {
            'chunks_processed': 0,
            'total_audio_seconds': 0,
            'avg_transcribe_time': 0,
            'total_chunks': 0
        }

    def log(self, message):
        """Log message if verbose enabled"""
        if self.verbose:
            print(f"[Pipeline] {message}")

    def initialize(self):
        """Initialize MedASR and MedGemma models"""
        print("\n" + "=" * 60)
        print("  Initializing Real-Time Pipeline")
        print("=" * 60)

        # Check Ollama service
        self.log("Checking Ollama service...")
        try:
            import ollama
            response = ollama.list()
            models = [getattr(m, 'model', str(m)) for m in getattr(response, 'models', [])]
            self.log(f"Available models: {models[:3]}...")
        except Exception as e:
            self.log(f"Ollama check: {e}")

        # Create unified model manager
        print("\n[1/2] Loading MedASR (Medical Speech Recognition)...")
        try:
            self.manager = UnifiedModelManager()
            results = self.manager.load_all(verbose=False)

            if not results.get('medasr'):
                print("  ✗ MedASR failed to load")
                return False

            print("  ✓ MedASR loaded successfully")

        except Exception as e:
            print(f"  ✗ Failed to initialize models: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Check MedGemma
        print("[2/2] Checking MedGemma (Medical Summary)...")
        try:
            medgemma_status = self.manager.medgemma.status
            if medgemma_status == ModelStatus.READY:
                print("  ✓ MedGemma ready")
            else:
                print(f"  ⚠ MedGemma status: {medgemma_status}")
        except Exception as e:
            print(f"  ⚠ MedGemma check: {e}")

        self.initialized = True
        self.running = True

        print("\n" + "-" * 60)
        print("  Pipeline initialized and ready!")
        print("-" * 60)

        return True

    def start_session(self):
        """Start a new recording session"""
        self.session_active = True
        self.session_start = time.time()
        self.audio_buffer = bytes()
        self.full_transcription = ""
        self.stats = {
            'chunks_processed': 0,
            'total_audio_seconds': 0,
            'avg_transcribe_time': 0,
            'total_chunks': 0
        }

        print(f"\n{'='' * 60}")
        print("  Recording Session Started")
        print(f"{'='' * 60}")

    def end_session(self):
        """End current session and generate final summary"""
        if not self.session_active:
            return

        self.session_active = False
        elapsed = time.time() - self.session_start

        print(f"\n{'='' * 60}")
        print("  Recording Session Ended")
        print(f"{'='' * 60}")
        print(f"\n  Session Duration: {elapsed:.1f} seconds")
        print(f"  Chunks Processed: {self.stats['chunks_processed']}")
        print(f"  Total Transcription: {len(self.full_transcription)} characters")

        # Generate final summary if we have transcription
        if self.full_transcription.strip():
            self._generate_summary(self.full_transcription, final=True)

        print(f"\n{'='' * 60}")

    def process_chunk(self, audio_data: bytes):
        """
        Process a single audio chunk

        Args:
            audio_data: Raw PCM audio bytes
        """
        if not self.session_active:
            return

        try:
            # Convert bytes to numpy array (16kHz, 16-bit, mono)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Transcribe using MedASR
            start_time = time.time()

            if self.manager and self.manager.medasr.is_available():
                result = self.manager.medasr.transcribe_array(audio_float, sample_rate=16000)
                transcribe_time = time.time() - start_time

                # Update statistics
                self.stats['chunks_processed'] += 1
                self.stats['total_audio_seconds'] += len(audio_float) / 16000
                self.stats['avg_transcribe_time'] = (
                    (self.stats['avg_transcribe_time'] * (self.stats['chunks_processed'] - 1) +
                     transcribe_time) / self.stats['chunks_processed']
                )

                # Accumulate transcription
                if result.text.strip():
                    self.full_transcription += " " + result.text

                    # Display result
                    self._display_transcription(result, transcribe_time)

                    # Periodically generate summary
                    if self.stats['chunks_processed'] % 50 == 0:
                        self._generate_summary(self.full_transcription, interim=True)

        except Exception as e:
            print(f"[Pipeline] Processing error: {e}")
            import traceback
            traceback.print_exc()

    def _display_transcription(self, result, process_time):
        """Display transcription result"""
        text = result.text.strip()
        if not text:
            return

        confidence = getattr(result, 'confidence', 0) * 100

        print(f"\n[Transcription] (confidence: {confidence:.1f}%, time: {process_time:.2f}s)")
        print(f"  {text}")

    def _generate_summary(self, transcription: str, interim=False, final=False):
        """Generate medical summary using MedGemma"""
        if not transcription.strip():
            return

        try:
            # Check if MedGemma is available
            if not self.manager or not self.manager.medgemma.is_available():
                return

            # Create patient info
            patient = create_patient_info(
                name="Patient",
                age=0,
                gender="unknown",
                medical_history="N/A",
                medications="N/A",
                allergies="N/A"
            )

            # Generate summary
            start_time = time.time()

            record = self.manager.generate_record(
                transcription=transcription,
                patient_info=patient,
                template="general"
            )

            gen_time = time.time() - start_time

            marker = "[SUMMARY-FINAL]" if final else "[SUMMARY-INTERIM]"
            print(f"\n{marker} (generated in {gen_time:.2f}s)")

            # Display key fields
            if record.chief_complaint:
                print(f"  Chief Complaint: {record.chief_complaint[:100]}...")
            if record.assessment:
                print(f"  Assessment: {record.assessment[:100]}...")
            if record.plan:
                print(f"  Plan: {record.plan[:100]}...")

        except Exception as e:
            print(f"[Pipeline] Summary generation error: {e}")

    def shutdown(self):
        """Shutdown the pipeline"""
        self.running = False
        if self.manager:
            self.manager.unload_all()
        print("[Pipeline] Shutdown complete")


def test_pipeline():
    """Test the pipeline with sample audio"""
    print("\n" + "=" * 60)
    print("  Pipeline Test Mode")
    print("=" * 60)

    pipeline = RealtimePipeline(verbose=True)

    print("\nInitializing pipeline...")
    if not pipeline.initialize():
        print("Failed to initialize pipeline")
        return

    # Simulate audio chunks
    print("\nProcessing simulated audio chunks...")
    pipeline.start_session()

    sample_texts = [
        "Good morning, what brings you in today?",
        "I've been having chest pain for the past three days.",
        "It's in the center of my chest, and it radiates to my left arm.",
        "I would rate it about six out of ten.",
    ]

    for i, text in enumerate(sample_texts):
        # Create synthetic audio data (silence)
        silence = np.zeros(16000 * 2, dtype=np.int16)  # 2 seconds of silence
        audio_bytes = silence.tobytes()

        print(f"\n--- Chunk {i+1} ---")
        pipeline.process_chunk(audio_bytes)

        # Simulate transcription
        time.sleep(0.5)

    pipeline.end_session()
    pipeline.shutdown()

    print("\n" + "=" * 60)
    print("  Test Complete")
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time MedASR + MedGemma Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run pipeline test with simulated audio'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'test'],
        default='live',
        help='Run mode (default: live)'
    )

    args = parser.parse_args()

    if args.test or args.mode == 'test':
        test_pipeline()
    else:
        print("\n" + "=" * 60)
        print("  Real-Time Medical Speech Processing Pipeline")
        print("=" * 60)
        print("\n  This module is designed to be used with audio_receiver.py")
        print("  Run: python audio_receiver.py")
        print("\n  The pipeline will automatically process audio chunks")
        print("  and display transcriptions and summaries.")
        print("=" * 60)

        print("\nUsage:")
        print("  1. Start audio_receiver.py: python audio_receiver.py")
        print("  2. The pipeline will initialize automatically")
        print("  3. Start ESP32 recording to see real-time results")


if __name__ == "__main__":
    main()
