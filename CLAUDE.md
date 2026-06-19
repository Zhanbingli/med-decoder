# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CardioVoice is a real-time cardiology outpatient documentation system for the Med-Gemma Impact Challenge. It captures a doctor-patient conversation from a local microphone, transcribes it with MedASR, and generates a structured outpatient record with a local LLM (via Ollama) — all running locally with no cloud data transmission.

## Architecture

Queue-decoupled, hardware-free pipeline. Audio enters from a local USB/built-in microphone (or a WAV file for testing); heavy inference runs on its own worker thread, never on the audio capture thread.

```
AudioSource(frame) -> VADSegmenter -> seg_q -> ASR worker -> transcript
                                                        \-> LLM note (MedGemma)
```

1. **Audio Capture** (`audio_capture.py`) — `MicSource` (sounddevice/PortAudio, resamples device-native rate to 16 kHz via soxr) and `FileSource` (WAV replay for dev/test/demo). Both emit 16 kHz mono float32 frames. `list_input_devices()` enumerates inputs.
2. **VAD Segmenter** (`vad_segmenter.py`) — energy/silence-based endpointing that turns the frame stream into complete utterance segments (no fixed 2s chunking, no model download).
3. **Realtime Pipeline** (`realtime_pipeline.py`) — wires source → segmenter → ASR worker → transcript → note. Modes: `--mic`, `--file [path]`, `--list-devices`.
4. **Unified Model Manager** (`backend/unified_model_manager.py`) — central class managing both models:
   - **MedASR**: HuggingFace Transformers pipeline (`google/medasr`), runs on MPS (Apple Silicon) or CPU.
   - **LLM (MedGemma slot)**: Ollama API. Default model `qwen3.5:9b` (MedGemma GGUF not installed locally). Override with env var `CARDIOVOICE_LLM_MODEL`. Reasoning models need `think=False` (set in the wrapper) or `content` comes back empty.
5. **Demo** (`demo.py`) — no-hardware demo with modes: `llm`, `asr`, `full`, `info`. Uses the real `UnifiedModelManager`.
6. **UI** (`app.py`) — Streamlit MVP (local-first): record from mic or upload audio → transcribe → generate note → edit the 7 fields → save draft / verify. Loads the model once via `@st.cache_resource`; mic recording runs in a background `MicRecorder` thread.
7. **Persistence** (`backend/store.py`) — local SQLite store at `~/.cardiovoice/cardiovoice.db`. `encounters` + `records` with draft/verified status and audit timestamps. Data never leaves the machine. (`schema.sql` is the older Postgres design, not used by the MVP.)

Key data flow: mic/file audio → `audio_capture` → `vad_segmenter` → `UnifiedModelManager` (MedASR transcribe → LLM generate) → structured `OutpatientRecord`.

## Setup

The working conda env is **`medgemma`** (`~/miniconda3/envs/medgemma`); the deps (torch, transformers, ollama, numpy, sounddevice, soxr, soundfile) are installed there.

```bash
conda activate medgemma
pip install -r requirements.txt

# Ollama must be running for note generation
ollama serve
# Uses qwen3.5:9b by default (already pulled). To use a smaller/faster model:
#   export CARDIOVOICE_LLM_MODEL=qwen3.5:2b

# HuggingFace login may be required for MedASR
huggingface-cli login
```

## Running

```bash
# Demo (no hardware needed)
python demo.py --mode llm     # note generation from a sample transcript
python demo.py --mode asr     # transcribe the bundled MedASR test audio
python demo.py --mode full    # audio -> transcript -> note

# Realtime pipeline (CLI)
python realtime_pipeline.py --list-devices
python realtime_pipeline.py --mic                # live microphone
python realtime_pipeline.py --file               # replay bundled test wav
python realtime_pipeline.py --file path/to.wav   # replay a specific file

# UI (doctor-facing MVP)
streamlit run app.py
```

## Key Technical Details

- Apple Silicon optimized: MPS backend for PyTorch, Metal for Ollama. Use `torch.mps.empty_cache()` (not cuda).
- Audio is captured at the device's native rate and resampled to 16 kHz mono; segments are cut at speech pauses by the VAD, not on a fixed clock.
- `transcribe_array` accepts float32 in [-1,1] directly (integer input is scaled by 1/32768).
- The LLM default is `qwen3.5:9b` via Ollama (≈30 s/note). Set `CARDIOVOICE_LLM_MODEL=qwen3.5:2b` for faster, lower-quality output. The real MedGemma GGUF is not downloaded.
- Note parsing is regex-based section extraction; a future improvement is JSON-schema structured output.
- Comments/docstrings in `backend/unified_model_manager.py` are in Chinese.
- `models load_all()` loads MedASR and the LLM probe in parallel.
