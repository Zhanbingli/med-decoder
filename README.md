# 🫀 CardioVoice — Local-First Cardiology Documentation

<div align="center">

**Turn a doctor–patient conversation into a structured outpatient note — entirely on your own Mac.**

![Models](https://img.shields.io/badge/Models-MedASR%20%2B%20Local%20LLM-0E7C7B)
![Audio](https://img.shields.io/badge/Audio-Local%20Microphone-orange)
![Privacy](https://img.shields.io/badge/Data-100%25%20Local-2E7D32)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow)

</div>

---

CardioVoice records an outpatient visit from a local microphone, transcribes it
with **MedASR**, and drafts a structured cardiology note with a **local LLM**
(via Ollama). The doctor reviews and edits the note, then verifies and saves it.
Everything — audio, transcript, notes — stays on the machine. No cloud.

> **Status: working MVP.** The pipeline (mic → transcript → note → review → save)
> runs end to end. The note model defaults to `qwen3.5:9b` because the MedGemma
> GGUF isn't bundled (swap it via `CARDIOVOICE_LLM_MODEL`). Section parsing is
> regex-based and storage is local SQLite. See [Roadmap](#roadmap) for what's next.

---

## How it works

```
🎤 Microphone ──► VAD segment ──► MedASR ──► transcript ──► Local LLM ──► structured note
   (or upload)    (speech pauses)   (ASR)                    (Ollama)       │
                                                                            ▼
                                                          Doctor edits → verify → SQLite
```

| Stage | Component | Tech |
|-------|-----------|------|
| Capture | `audio_capture.py` | sounddevice (PortAudio), soxr 48k→16k |
| Segment | `vad_segmenter.py` | energy/silence endpointing |
| Transcribe | `backend/unified_model_manager.py` | MedASR (Transformers, MPS) |
| Generate | `backend/unified_model_manager.py` | Local LLM via Ollama |
| Persist | `backend/store.py` | SQLite (`~/.cardiovoice/cardiovoice.db`) |
| UI | `app.py` | Streamlit |

The UI uses a **record-then-transcribe** flow (`st.audio_input`). A separate CLI
(`realtime_pipeline.py`) offers true live, utterance-by-utterance streaming.

---

## Why local-first

The whole point is that **protected health information never leaves the
machine**. There's no server, no cloud API, no telemetry. For a clinical tool
this is the simplest path to a defensible privacy posture, and it's a deliberate
product decision rather than a limitation.

---

## Requirements

- macOS on **Apple Silicon** (uses the MPS backend; CPU also works, slower)
- Python 3.11
- [Ollama](https://ollama.com) with a chat model pulled (default `qwen3.5:9b`)
- A microphone (a USB omnidirectional desk/conference mic works well for
  capturing both doctor and patient)

---

## Setup

```bash
git clone https://github.com/Zhanbingli/med-decoder.git
cd med-decoder

# Conda env (the working env is named `medgemma`)
conda create -n medgemma python=3.11
conda activate medgemma

pip install torch torchvision torchaudio   # Apple Silicon MPS build
pip install -r requirements.txt

# MedASR may require a Hugging Face login
huggingface-cli login

# Pull a local LLM for note generation (default)
ollama pull qwen3.5:9b
```

On first microphone use, macOS will prompt for mic permission.

---

## Run

```bash
# 1) Start Ollama (separate terminal)
ollama serve

# 2) Launch the app
conda activate medgemma
streamlit run app.py
```

Then in the browser: fill patient info → **record** (or upload audio) → the
transcript appears → **generate note** → edit the fields → **verify & save**.
Records land in the **History** tab and in `~/.cardiovoice/cardiovoice.db`.

### Faster / lighter model

```bash
export CARDIOVOICE_LLM_MODEL=qwen3.5:2b   # quicker, lower quality
```

### Command line (no UI)

```bash
python demo.py --mode llm                  # note from a sample transcript
python demo.py --mode full                 # bundled test audio → transcript → note
python realtime_pipeline.py --list-devices
python realtime_pipeline.py --mic          # live streaming transcript (Ctrl+C to stop)
python realtime_pipeline.py --file a.wav   # replay a file through the pipeline
```

### Measuring accuracy (WER)

```bash
python eval_wer.py --demo                   # self-test (synthesizes audio via `say`)
python eval_wer.py --dir data/eval/         # your x.wav + x.txt reference pairs
python eval_wer.py --manifest eval.jsonl    # {"audio":..., "reference":...} per line
python eval_wer.py --dir data/eval/ --compare   # WER with audio preprocessing off vs on
python eval_wer.py --dir data/eval/ --correct   # WER with LLM term correction off vs on
```

Reports per-clip and aggregate WER, a substitution/deletion/insertion
breakdown, the most common confusions, and mean ASR confidence — so accuracy
is tracked with numbers, not guessed.

---

## Repository structure

```
med-decoder/
├── app.py                          # Streamlit UI (the product surface)
├── realtime_pipeline.py            # CLI: mic/file → VAD → MedASR → LLM
├── audio_capture.py                # MicSource + FileSource (16 kHz mono)
├── preprocess.py                   # high-pass + peak-normalize before ASR
├── vad_segmenter.py                # energy-based utterance segmentation
├── lexicons/cardiology.txt         # terms/drugs for LLM transcript correction
├── export.py                       # note export: text / Markdown / PDF
├── demo.py / demo_unified.py       # no-hardware demos
├── eval_wer.py                     # WER measurement harness
├── backend/
│   ├── unified_model_manager.py    # MedASR + Ollama LLM wrappers
│   └── store.py                    # local SQLite persistence
├── .streamlit/config.toml          # UI theme
├── requirements.txt
├── schema.sql                      # legacy Postgres schema (not used by the MVP)
└── CLAUDE.md                       # architecture notes for contributors
```

---

## Roadmap

- [x] **Structured output via JSON** (Ollama JSON mode + Pydantic) instead of regex
- [x] **Hallucination control** — note generation is grounded: exam/ECG/meds not in
  the transcript are left empty, never invented
- [x] **Confidence / uncertainty markers** so the doctor knows what to double-check
- [x] **WER measurement harness** (`eval_wer.py`) to track accuracy
- [x] **LLM post-ASR correction** of medical terms/drugs (conservative, lexicon-guided)
- [x] **Export the note** — text / Markdown / PDF (CJK-capable), plus copy-to-clipboard
- [ ] **Speaker separation** (dual-lavalier or diarization) for doctor vs patient
- [ ] Swap in the real **MedGemma** GGUF when available
- [ ] Multi-specialty templates beyond cardiology

---

## Limitations & disclaimer

This is an MVP and a documentation **aid**, not a medical device. Generated notes
are drafts that **must be reviewed and verified by a clinician** before any
clinical use. Transcription and generation can be wrong. No regulatory clearance
is claimed.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Model terms:
- **MedASR** — [HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms)
- LLM terms follow whichever Ollama model you configure.

---

## Contact

**zhanbingli** · zhanbing2025@gmail.com · GitHub [@Zhanbingli](https://github.com/Zhanbingli)
