"""
INMP441 I2S Microphone Configuration for ESP32-S3
================================================

Pin Configuration (DevKitC-1):
- SCK (Serial Clock)  → GPIO 6
- WS  (Word Select)   → GPIO 7
- SD  (Serial Data)   → GPIO 15
- VDD → 3.3V
- GND → GND
- L/R → GND (Left Channel)

Audio Parameters:
- Sample Rate: 16000 Hz
- Bit Depth: 16-bit
- Channels: Mono (Left)
"""

from machine import Pin, I2S
import os

# I2S Pin Configuration
I2S_BCLK = 1  # Serial Clock (SCK)
I2S_LRCL = 2  # Word Select (WS)
I2S_DIN = 42  # Serial Data (SD)

# Audio Parameters
SAMPLE_RATE = 16000
BIT_DEPTH = 16
CHANNELS = 1
BUFFER_SIZE = 4096

# I2S Configuration Dictionary
I2S_CONFIG = {
    "sck": I2S_BCLK,
    "ws": I2S_LRCL,
    "sd": I2S_DIN,
    "mode": I2S.RX,
    "bits": BIT_DEPTH,
    "rate": SAMPLE_RATE,
    "ibuf": BUFFER_SIZE * 2,
}


def i2s_init(i2s_id=0):
    """
    Initialize I2S interface for INMP441 microphone

    Args:
        i2s_id: I2S peripheral ID (0 or 1)

    Returns:
        I2S object configured for reading
    """
    try:
        i2s = I2S(
            i2s_id,
            sck=Pin(I2S_BCLK, Pin.IN),
            ws=Pin(I2S_LRCL, Pin.IN),
            sd=Pin(I2S_DIN, Pin.IN),
            mode=I2S.RX,
            bits=BIT_DEPTH,
            format=I2S.MONO,
            rate=SAMPLE_RATE,
            ibuf=BUFFER_SIZE * 2,
        )
        print(f"[I2S] Initialized successfully")
        print(f"  - BCLK: GPIO{I2S_BCLK}")
        print(f"  - LRCL: GPIO{I2S_LRCL}")
        print(f"  - DIN:  GPIO{I2S_DIN}")
        print(f"  - Sample Rate: {SAMPLE_RATE} Hz")
        print(f"  - Bit Depth: {BIT_DEPTH}-bit")
        print(f"  - Channels: {CHANNELS}")
        return i2s
    except Exception as e:
        print(f"[I2S] Initialization failed: {e}")
        return None


def i2s_deinit(i2s):
    """Deinitialize I2S interface"""
    if i2s:
        i2s.deinit()
        print("[I2S] Deinitialized")


def read_audio_chunk(i2s, chunk_size=1024):
    """
    Read a chunk of audio data from I2S microphone

    Args:
        i2s: I2S object
        chunk_size: Number of bytes to read

    Returns:
        bytes: Audio data chunk, or None if error
    """
    try:
        audio_data = i2s.read(chunk_size)
        return audio_data
    except Exception as e:
        print(f"[I2S] Read error: {e}")
        return None


def get_audio_info():
    """Return current audio configuration"""
    return {
        "sample_rate": SAMPLE_RATE,
        "bit_depth": BIT_DEPTH,
        "channels": CHANNELS,
        "buffer_size": BUFFER_SIZE,
        "pins": {"bclk": I2S_BCLK, "lrcl": I2S_LRCL, "din": I2S_DIN},
    }
