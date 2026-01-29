"""
Audio Sender - TCP Client for ESP32-S3
======================================

Sends audio data from INMP441 microphone to Mac via WiFi (TCP)

Protocol:
- TCP connection to Mac (172.20.10.9:8000)
- Audio format: 16kHz, 16-bit, mono PCM
- Data format: Raw PCM bytes
- Control commands: START, STOP, PING
"""

import socket
import time
import struct
from led import (
    led,
    STATE_WIFI_CONNECTED,
    STATE_RECORDING,
    STATE_STREAMING,
    STATE_SENDING,
    STATE_ERROR,
)

# WiFi and Server Configuration
WIFI_SSID = "iPhone"
WIFI_PASSWORD = "lzb023118@"
SERVER_IP = "172.20.10.9"
SERVER_PORT = 8000

# Audio Configuration
CHUNK_SIZE = 1024  # Bytes per audio chunk
SEND_INTERVAL = 0.05  # Send interval in seconds (20fps)

# Connection Settings
CONNECTION_TIMEOUT = 10  # Seconds
RECONNECT_DELAY = 2  # Seconds between reconnection attempts
MAX_RECONNECT_ATTEMPTS = 5


class AudioSender:
    def __init__(self):
        self.socket = None
        self.connected = False
        self.recording = False
        self.bytes_sent = 0
        self.chunks_sent = 0

    def connect_to_server(self):
        """Establish TCP connection to Mac server"""
        print("[TCP] Connecting to server...")
        print(f"  - Server: {SERVER_IP}:{SERVER_PORT}")

        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(CONNECTION_TIMEOUT)
                self.socket.connect((SERVER_IP, SERVER_PORT))

                self.connected = True
                print(f"[TCP] Connected successfully (attempt {attempt + 1})")

                # Send identification
                self.send_command("HELLO", "ESP32-Audio-Recorder")

                return True

            except Exception as e:
                print(f"[TCP] Connection attempt {attempt + 1} failed: {e}")
                if self.socket:
                    try:
                        self.socket.close()
                    except:
                        pass
                time.sleep(RECONNECT_DELAY)

        print("[TCP] Failed to connect after multiple attempts")
        self.connected = False
        return False

    def disconnect(self):
        """Close TCP connection"""
        if self.socket:
            try:
                self.socket.close()
                print("[TCP] Disconnected")
            except Exception as e:
                print(f"[TCP] Disconnect error: {e}")
        self.connected = False

    def send_audio(self, audio_data):
        """
        Send audio data to server

        Args:
            audio_data: bytes - Raw PCM audio data

        Returns:
            bool: True if successful
        """
        if not self.connected or not self.socket:
            return False

        try:
            # Send audio data length (4 bytes, little endian)
            length_header = struct.pack("<I", len(audio_data))
            self.socket.sendall(length_header)

            # Send audio data
            self.socket.sendall(audio_data)

            self.bytes_sent += len(audio_data)
            self.chunks_sent += 1

            if self.chunks_sent % 20 == 0:
                print(f"[TCP] Sent {self.chunks_sent} chunks, {self.bytes_sent} bytes")

            return True

        except Exception as e:
            print(f"[TCP] Send error: {e}")
            self.connected = False
            return False

    def send_command(self, command, args=""):
        """
        Send control command to server

        Args:
            command: str - Command name (HELLO, START, STOP, PING)
            args: str - Command arguments
        """
        if not self.connected or not self.socket:
            return False

        try:
            msg = f"{command}|{args}\n"
            self.socket.sendall(msg.encode("utf-8"))
            return True
        except Exception as e:
            print(f"[TCP] Command send error: {e}")
            return False

    def start_recording(self):
        """Start recording and inform server"""
        self.recording = True
        self.send_command("START")
        print("[TCP] Recording started")
        led.set_state(STATE_RECORDING)

    def stop_recording(self):
        """Stop recording and inform server"""
        self.recording = False
        self.send_command("STOP")
        print("[TCP] Recording stopped")
        print(f"[TCP] Total sent: {self.chunks_sent} chunks, {self.bytes_sent} bytes")
        led.set_state(STATE_WIFI_CONNECTED)

    def get_statistics(self):
        """Return connection statistics"""
        return {
            "connected": self.connected,
            "recording": self.recording,
            "bytes_sent": self.bytes_sent,
            "chunks_sent": self.chunks_sent,
        }


# Global audio sender instance
audio_sender = AudioSender()


def init_audio_sender():
    """Initialize and connect audio sender"""
    print("[TCP] Initializing audio sender...")
    print(f"  - Server: {SERVER_IP}:{SERVER_PORT}")

    if audio_sender.connect_to_server():
        print("[TCP] Audio sender ready")
        led.set_state(STATE_WIFI_CONNECTED)
        return audio_sender
    else:
        print("[TCP] Audio sender failed to connect")
        led.set_state(STATE_ERROR)
        return None
