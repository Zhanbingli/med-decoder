# Medical Audio Recording Device - Setup Guide
# =============================================

This guide explains how to set up the ESP32-S3 + INMP441 audio recording system for medical speech-to-text using MedASR and MedGemma.

## System Architecture

```
┌──────────────┐         WiFi (iPhone Hotspot)         ┌──────────────┐
│              │         172.20.10.x                    │              │
│  iPhone      │ ◄───────────────────────────────►     │    Mac       │
│  Hotspot     │                                   │              │
└──────────────┘                                   │ ┌──────────┐ │
                                                    │ │MedASR    │ │
┌──────────────┐      TCP Audio Stream (16kHz)     │ │转写      │ │
│ ESP32-S3     │ ─────────────────────────────────► │ └──────────┘ │
│ + INMP441    │      172.20.10.9:8000              │ ┌──────────┐ │
│ MicroPython  │                                   │ │MedGemma  │ │
└──────────────┘                                   │ │总结      │ │
                                                    │ └──────────┘ │
                                                    │ 实时显示    │
                                                    └─────────────┘
```

## Prerequisites

1. **Hardware**:
   - ESP32-S3-DevKitC-1
   - INMP441 MEMS Microphone
   - Jumper wires
   - USB-C cable

2. **Software**:
   - macOS (Apple Silicon recommended)
   - Conda environment `medgemma`
   - Python 3.11+

3. **WiFi**:
   - iPhone with Personal Hotspot enabled
   - Hotspot name: `iPhone`
   - Hotspot password: `lzb023118@`

## Hardware Connection

| INMP441 Pin | ESP32-S3 Pin | Description |
|-------------|--------------|-------------|
| VDD | 3V3 | Power (3.3V) |
| GND | GND | Ground |
| L/R | GND | Left channel (connect to GND) |
| SCK | GPIO 6 | Serial Clock (BCLK) |
| WS | GPIO 7 | Word Select (LRCL) |
| SD | GPIO 15 | Serial Data (DOUT) |

## Software Installation

### Step 1: Install Additional Dependencies

Activate your conda environment and install required packages:

```bash
# Activate conda environment
conda activate medgemma

# Install mpfshell for ESP32 file upload
pip install mpfshell pyserial

# Verify installation
mpfshell --help
```

### Step 2: Upload ESP32 Firmware

1. **Enter MicroPython DFU Mode**:
   - Hold the **BOOT** button on ESP32-S3
   - While holding BOOT, press **RESET**
   - Release BOOT
   - Mac should mount a drive called `PYBFLASH` or `ESP32`

2. **Upload Files using mpfshell**:

   ```bash
   # Navigate to project directory
   cd /Users/lizhanbing12/med_exper

   # Start mpfshell
   mpfshell

   # In mpfshell, run:
   > open tty.usbmodem5A7A0148011
   > cd /flash
   > put esp32_firmware/boot.py
   > put esp32_firmware/main.py
   > put esp32_firmware/i2s_config.py
   > put esp32_firmware/audio_sender.py
   > put esp32_firmware/led.py
   > ls
   > repl
   ```

   Or use command line:

   ```bash
   mpfshell -c "open tty.usbmodem5A7A0148011; cd /flash; put esp32_firmware/boot.py; put esp32_firmware/main.py; put esp32_firmware/i2s_config.py; put esp32_firmware/audio_sender.py; put esp32_firmware/led.py; repl"
   ```

3. **Reset ESP32**:
   - Press the **RESET** button
   - ESP32 will boot and connect to WiFi

### Step 3: Start Mac Services

**Terminal 1 - Audio Receiver**:

```bash
conda activate medgemma
cd /Users/lizhanbing12/med_exper
python audio_receiver.py
```

Expected output:
```
============================================================
  Audio Receiver - ESP32-S3 Audio Streaming Server
============================================================

  Listening on: 0.0.0.0:8000
  Pipeline: Enabled

  Waiting for ESP32 connection...
------------------------------------------------------------
```

**Terminal 2 - Realtime Pipeline** (optional, for display):

```bash
conda activate medgemma
cd /Users/lizhanbing12/med_exper
python realtime_pipeline.py
```

### Step 4: Start Recording

1. **On ESP32**:
   - Press the **BOOT** button to start recording
   - LED will turn blue (blinking)
   - Press BOOT again to stop

2. **On Mac**:
   - Audio receiver will show connection status
   - Transcription will appear in realtime_pipeline output

## File Structure

```
med_exper/
├── esp32_firmware/
│   ├── boot.py           # WiFi/Peripheral initialization
│   ├── main.py           # Main recording loop
│   ├── i2s_config.py     # INMP441 I2S driver
│   ├── audio_sender.py   # TCP client for audio streaming
│   └── led.py            # RGB LED status indicator
├── audio_receiver.py     # TCP server, receives audio from ESP32
├── realtime_pipeline.py  # MedASR + MedGemma integration
├── backend/
│   └── unified_model_manager.py  # Model management (existing)
└── README_SETUP.md       # This file
```

## LED Status Reference

| LED Color | Pattern | Status |
|-----------|---------|--------|
| Red | Blinking | WiFi connecting |
| Red | Solid | WiFi connected |
| Green | Solid | System ready |
| Blue | Blinking | Recording audio |
| Blue | Solid | Audio streaming |
| Green | Quick blink | Data sent |
| Yellow | Blinking | Error |

## Troubleshooting

### ESP32 Not Connecting to WiFi

1. Check iPhone hotspot is enabled
2. Verify WiFi password: `lzb023118@`
3. Check Mac IP address (may change after iPhone restart):
   ```bash
   ipconfig getifaddr en0
   ```

### No Audio Received

1. Verify ESP32 and Mac are on the same network
2. Check firewall settings (port 8000)
3. Verify ESP32 serial output for connection status

### mpfshell Not Working

1. Check serial port:
   ```bash
   ls /dev/tty.usbmodem*
   ```
2. Try with sudo:
   ```bash
   sudo mpfshell -s /dev/tty.usbmodem5A7A0148011
   ```

### Model Loading Errors

1. Verify Ollama is running:
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Check Hugging Face token:
   ```bash
   conda activate medgemma
   huggingface-cli login
   ```

## Usage Notes

1. **IP Address Changes**: iPhone hotspot IP may change after restart. Update `SERVER_IP` in `esp32_firmware/audio_sender.py` if needed.

2. **Audio Quality**: For best results:
   - Keep microphone close to speaker
   - Minimize background noise
   - Ensure stable WiFi connection

3. **Power**: USB-C power should provide sufficient current. If ESP32 resets, try a different USB port or cable.

## Commands Reference

### ESP32 Control (BOOT Button)
- **Short press**: Toggle recording on/off
- **Long press**: (not implemented)

### Mac Terminal Commands

```bash
# Start audio receiver
python audio_receiver.py

# Start pipeline test mode
python realtime_pipeline.py --test

# Check Ollama models
curl http://localhost:11434/api/tags

# Check MedGemma status
conda activate medgemma
python -c "from backend.unified_model_manager import UnifiedModelManager; m=UnifiedModelManager(); m.load_all(); print(m.get_status())"
```

## Support

For issues:
1. Check ESP32 serial output: `mpfshell -s /dev/tty.usbmodem5A7A0148011 -c "repl"`
2. Verify all files are uploaded: `mpfshell -s /dev/tty.usbmodem5A7A0148011 -c "ls"`
3. Check Mac firewall settings
