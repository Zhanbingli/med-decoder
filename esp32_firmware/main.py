"""
main.py - ESP32-S3 Audio Recorder Main Loop
===========================================

Main functionality:
1. Monitor button for recording control
2. Read audio data from INMP441 microphone
3. Send audio data to Mac via TCP
4. Handle connection errors and reconnection

Controls:
- BOOT button: Toggle recording (press to start/stop)
"""

import time
import sys
from machine import Pin

# Import our modules
from i2s_config import i2s_init, read_audio_chunk
from led import (
    led,
    STATE_WIFI_CONNECTED,
    STATE_RECORDING,
    STATE_STREAMING,
    STATE_ERROR,
    init_led,
)
from audio_sender import audio_sender, init_audio_sender, SERVER_IP, SERVER_PORT

# Configuration
AUDIO_CHUNK_SIZE = 1024  # Bytes per read
BUTTON_DEBOUNCE = 50  # ms
SEND_INTERVAL = 0.05  # seconds

# Button pin (GPIO 0 is the BOOT button on DevKitC-1)
BUTTON_PIN = 0


class ButtonHandler:
    """Handle BOOT button for recording control"""

    def __init__(self, pin=BUTTON_PIN):
        self.pin = Pin(pin, Pin.IN, Pin.PULL_UP)
        self.last_state = 1  # 1 = not pressed
        self.last_time = 0
        self.debounce = BUTTON_DEBOUNCE

    def is_pressed(self):
        """Check if button is currently pressed"""
        return self.pin.value() == 0

    def was_pressed(self):
        """
        Check if button was just pressed (rising edge detection with debounce)
        Returns True only on initial press
        """
        current_time = time.ticks_ms()
        current_state = self.pin.value()

        # Detect falling edge (press)
        if current_state == 0 and self.last_state == 1:
            # Debounce check
            if time.ticks_diff(current_time, self.last_time) > self.debounce:
                self.last_state = current_state
                self.last_time = current_time
                return True

        # Update state
        if current_state != self.last_state:
            self.last_state = current_state
            self.last_time = current_time

        return False

    def is_held(self, duration_ms=1000):
        """Check if button has been held for specified duration"""
        if self.is_pressed():
            press_time = time.ticks_ms() - self.last_time
            return press_time >= duration_ms
        return False


def check_server_connection():
    """Ensure TCP connection is active, reconnect if needed"""
    if not audio_sender.connected:
        print("\n[Main] TCP connection lost, reconnecting...")
        led.set_state(STATE_ERROR)
        if init_audio_sender():
            print("[Main] Reconnected successfully")
            return True
        else:
            print("[Main] Reconnection failed")
            return False
    return True


def record_and_send(i2s, sender):
    """
    Main recording loop: read audio and send to server

    Args:
        i2s: Initialized I2S object
        sender: Initialized AudioSender object
    """
    chunk_count = 0
    error_count = 0
    max_errors = 10

    while sender.recording:
        # Check server connection
        if not check_server_connection():
            error_count += 1
            if error_count >= max_errors:
                print("[Main] Too many connection errors, stopping recording")
                sender.stop_recording()
                break
            time.sleep(1)
            continue

        # Read audio data from microphone
        audio_data = read_audio_chunk(i2s, AUDIO_CHUNK_SIZE)

        if audio_data is None:
            error_count += 1
            continue

        # Send audio data to server
        if sender.send_audio(audio_data):
            chunk_count += 1
            error_count = 0  # Reset error count on success

            # Update LED to streaming state briefly
            led.set_state(STATE_STREAMING)

            if chunk_count % 100 == 0:
                print(f"[Main] Recording... {chunk_count} chunks sent")

            # Small delay to control send rate
            time.sleep(SEND_INTERVAL)
        else:
            error_count += 1
            print(f"[Main] Send error #{error_count}")

    return chunk_count


def main():
    """Main application loop"""
    print("\n" + "=" * 50)
    print("  ESP32-S3 Audio Recorder")
    print("=" * 50)

    # Initialize button handler
    button = ButtonHandler()
    print(f"\n[Main] BOOT button initialized on GPIO {BUTTON_PIN}")

    # Recording state
    recording = False

    # Main loop
    print("\n[Main] Starting main loop...")
    print("  Press BOOT button to start/stop recording")
    print("-" * 50)

    while True:
        try:
            # Check for button press
            if button.was_pressed():
                if recording:
                    # Stop recording
                    print("\n[Main] Stopping recording...")
                    audio_sender.stop_recording()
                    recording = False
                    led.set_state(STATE_WIFI_CONNECTED)
                else:
                    # Start recording
                    print("\n[Main] Starting recording...")
                    audio_sender.start_recording()
                    recording = True

            # If recording, read and send audio
            if recording:
                import boot
                # Get global I2S object from boot.py
                if hasattr(boot, "_i2s") and boot._i2s is not None:
                    chunks = record_and_send(boot._i2s, audio_sender)
                    print(f"\n[Main] Recording session ended. Total chunks: {chunks}")
                    recording = False
                else:
                    print("[Main] Error: I2S not initialized")
                    led.set_state(STATE_ERROR)
                    time.sleep(1)

            # Small delay to prevent busy-waiting
            time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user")
            if recording:
                audio_sender.stop_recording()
            break

        except Exception as e:
            print(f"\n[Main] Error: {e}")
            import sys

            sys.print_exception(e)
            led.set_state(STATE_ERROR)
            time.sleep(1)


if __name__ == "__main__":
    main()
