# CardioVoice Project Files

This folder contains all files for the CardioVoice system - a real-time cardiology 
outpatient documentation system using MedGemma and MedASR for the Kaggle 
Med-Gemma Impact Challenge.

## Quick Start

1. **Read the main README.md** for complete setup instructions

2. **Technical Report**: See `CARDIOVOICE_Technical_Report.md` for detailed documentation

3. **Demo**: Run `python demo.py` to see the system in action

## File Inventory

### Documentation
- `README.md` - Main project documentation and setup guide
- `CARDIOVOICE_Technical_Report.md` - Comprehensive technical report for Kaggle

### Configuration
- `environment.yml` - Conda environment configuration
- `requirements.txt` - Python dependencies
- `schema.sql` - PostgreSQL database schema

### Demo
- `demo.py` - Demonstration script

### Original Project Files (Pre-existing)
- `README_MEDGEMMA_SETUP.md` - Original MedGemma setup guide
- `ollama_medgemma.py` - Ollama integration example
- `test_medgemma.py` - MedGemma test script
- `learn_gguf.py` - GGUF format example
- `learn_medasr.py` - MedASR example

## For Kaggle Submission

Include these files in your submission:
1. `README.md`
2. `CARDIOVOICE_Technical_Report.md`
3. `environment.yml` or `requirements.txt`
4. `schema.sql`
5. Any code files demonstrating your implementation

## System Components

- **ESP32 Firmware** - In `esp32_firmware/` folder (create this)
- **Backend API** - In `backend/` folder (create this)
- **Frontend UI** - In `ui/` folder (create this)
- **Scripts** - In `scripts/` folder (create this)
