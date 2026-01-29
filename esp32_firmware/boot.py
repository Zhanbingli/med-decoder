"""
boot.py - ESP32-S3 MicroPython Boot Configuration
=================================================

Initialization sequence:
1. Configure LED indicator
2. Initialize I2S microphone (INMP441)
3. Connect to WiFi (iPhone hotspot)
4. Connect to TCP server (Mac)
5. Wait for user interaction
"""

import network
import time
import machine

# Import our modules
from i2s_config import i2s_init, get_audio_info
from led import init_led, led, STATE_WIFI_CONNECTING, STATE_WIFI_CONNECTED, STATE_ERROR
from audio_sender import init_audio_sender, audio_sender

# WiFi Configuration
WIFI_SSID = "iPhone"
WIFI_PASSWORD = "lzb023118@"

# Server Configuration
SERVER_IP = "172.20.10.9"
SERVER_PORT = 8000


def connect_wifi():
    """
    Connect to WiFi network (iPhone hotspot)

    Returns:
        bool: True if connected successfully
    """
    print("\n" + "=" * 50)
    print("  ESP32-S3 Audio Recorder - Starting...")
    print("=" * 50)

    # Initialize LED for WiFi status
    led.set_state(STATE_WIFI_CONNECTING)

    # Disable access point
    ap = network.WLAN(network.AP_IF)
    ap.active(False)

    # Configure station mode
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    print(f"\n[WiFi] Connecting to '{WIFI_SSID}'...")

    # Connect to WiFi
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    # Wait for connection
    max_wait = 30
    while max_wait > 0:
        if wlan.isconnected():
            break
        print(f"[WiFi] Waiting... ({max_wait}s)")
        time.sleep(1)
        max_wait -= 1

    if wlan.isconnected():
        print(f"\n[WiFi] Connected!")
        print(f"  - IP Address: {wlan.ifconfig()[0]}")
        print(f"  - Subnet Mask: {wlan.ifconfig()[1]}")
        print(f"  - Gateway: {wlan.ifconfig()[2]}")
        print(f"  - DNS: {wlan.ifconfig()[3]}")
        led.set_state(STATE_WIFI_CONNECTED)
        return True
    else:
        print("\n[WiFi] Failed to connect")
        led.set_state(STATE_ERROR)
        return False


def print_system_info():
    """Print system information"""
    print("\n" + "-" * 50)
    print("  System Information")
    print("-" * 50)

    # ESP32-S3 Chip Info
    print(f"\n[System]")
    print(f"  - Chip ID: {machine.unique_id().hex()}")
    print(f"  - Frequency: {machine.freq() / 1000000} MHz")
    print(f"  - Memory: {gc.mem_free()} bytes free")

    # Audio Configuration
    audio_info = get_audio_info()
    print(f"\n[Audio - INMP441]")
    print(f"  - Sample Rate: {audio_info['sample_rate']} Hz")
    print(f"  - Bit Depth: {audio_info['bit_depth']}-bit")
    print(f"  - Channels: {audio_info['channels']}")
    print(f"  - Buffer Size: {audio_info['buffer_size']} bytes")
    print(
        f"  - Pins: BCLK={audio_info['pins']['bclk']}, LRCL={audio_info['pins']['lrcl']}, DIN={audio_info['pins']['din']}"
    )

    # Server Configuration
    print(f"\n[Server]")
    print(f"  - IP: {SERVER_IP}")
    print(f"  - Port: {SERVER_PORT}")

    print("\n" + "-" * 50)


def main():
    """Main boot sequence"""
    try:
        # Step 1: Initialize LED
        print("\n[Init] Initializing LED...")
        init_led()

        # Step 2: Initialize I2S Microphone
        print("\n[Init] Initializing I2S microphone...")
        i2s = i2s_init()
        if i2s is None:
            print("[Error] I2S initialization failed!")
            led.set_state(STATE_ERROR)
            return

        # Step 3: Connect to WiFi
        print("\n[Init] Connecting to WiFi...")
        if not connect_wifi():
            print("[Error] WiFi connection failed!")
            return

        # Step 4: Print system info
        print_system_info()

        # Step 5: Connect to TCP server
        print("\n[Init] Connecting to TCP server...")
        if init_audio_sender() is None:
            print("[Warning] TCP connection failed, continuing anyway...")

        # Step 6: Ready indicator
        led.success_pulse()
        print("\n" + "=" * 50)
        print("  System Ready!")
        print("  Waiting for recording command...")
        print("=" * 50)
        print("\n  To start recording, press the BOOT button")
        print("  Or send 'START' command via TCP")
        print("\n")

        # Store I2S object for main.py to use
        global _i2s
        _i2s = i2s

    except Exception as e:
        print(f"\n[Error] Boot sequence failed: {e}")
        import sys

        sys.print_exception(e)
        led.set_state(STATE_ERROR)


# Run boot sequence
main()
