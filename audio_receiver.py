#!/usr/bin/env python3
"""
Audio Receiver - TCP Server for ESP32-S3 Audio Streaming
=========================================================

Receives audio data from ESP32-S3 via WiFi (TCP) and forwards to processing pipeline.

Protocol:
- Listen on 0.0.0.0:8000
- Accept connections from ESP32
- Receive audio chunks (16kHz, 16-bit, mono PCM)
- Forward to realtime_pipeline for processing
- Handle control commands (START, STOP)

Usage:
    python audio_receiver.py [--host HOST] [--port PORT] [--verbose]
"""

import socket
import struct
import threading
import time
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Lazy import for pipeline (to avoid circular import issues)
_pipeline_class = None


def get_pipeline_class():
    """Lazy import of RealtimePipeline"""
    global _pipeline_class
    if _pipeline_class is None:
        try:
            from realtime_pipeline import RealtimePipeline

            _pipeline_class = RealtimePipeline
        except ImportError as e:
            print(f"[Warning] Pipeline not available: {e}")
            _pipeline_class = None
    return _pipeline_class


class AudioReceiver:
    """TCP Server for receiving audio from ESP32"""

    def __init__(self, host="0.0.0.0", port=8000, verbose=False):
        """
        Initialize audio receiver

        Args:
            host: Bind address (0.0.0.0 = all interfaces)
            port: TCP port to listen on
            verbose: Enable verbose logging
        """
        self.host = host
        self.port = port
        self.verbose = verbose

        self.socket = None
        self.client_socket = None
        self.running = False
        self.client_connected = False
        self.recording = False

        self.bytes_received = 0
        self.chunks_received = 0
        self.start_time: Optional[float] = time.time()

        self.pipeline = None
        pipeline_class = get_pipeline_class()
        if pipeline_class:
            try:
                self.pipeline = pipeline_class()
            except Exception as e:
                print(f"[Warning] Failed to initialize pipeline: {e}")

        self.lock = threading.Lock()

    def log(self, message):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[Receiver] {message}")

    def start(self):
        """Start the TCP server"""
        print("\n" + "=" * 60)
        print("  Audio Receiver - ESP32-S3 Audio Streaming Server")
        print("=" * 60)
        print(f"\n  Listening on: {self.host}:{self.port}")
        print(f"  Pipeline: {'Enabled' if self.pipeline else 'Disabled'}")
        print("\n  Waiting for ESP32 connection...")
        print("-" * 60)

        try:
            # Create TCP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            self.socket.settimeout(1.0)  # Allow periodic checks

            self.running = True
            self._accept_loop()

        except Exception as e:
            print(f"[Error] Server error: {e}")
        finally:
            self.stop()

    def _accept_loop(self):
        """Accept incoming connections"""
        while self.running:
            try:
                self.client_socket, addr = self.socket.accept()
                self.client_connected = True
                print(f"\n  ✓ ESP32 connected from {addr[0]}:{addr[1]}")
                print("  Recording started. Press Ctrl+C to stop.\n")

                self._handle_client()

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Error] Accept error: {e}")
                break

    def _handle_client(self):
        """Handle client connection"""
        self.bytes_received = 0
        self.chunks_received = 0
        self.start_time = time.time()

        try:
            self.client_socket.settimeout(1.0)

            while self.client_connected and self.running:
                try:
                    # Read 4-byte length header
                    length_bytes = self._read_exact(4)
                    if length_bytes is None:
                        break

                    chunk_length = struct.unpack("<I", length_bytes)[0]

                    # Read audio data
                    audio_data = self._read_exact(chunk_length)
                    if audio_data is None:
                        break

                    # Process audio data
                    self._process_audio(audio_data)

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[Error] Client error: {e}")
                    break

        finally:
            self._disconnect_client()

    def _read_exact(self, n):
        """Read exactly n bytes from socket"""
        data = b""
        while len(data) < n:
            try:
                chunk = self.client_socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except Exception:
                return None
        return data

    def _process_audio(self, audio_data):
        """Process received audio data"""
        with self.lock:
            self.bytes_received += len(audio_data)
            self.chunks_received += 1

            # Forward to pipeline if available
            if self.pipeline and self.recording:
                try:
                    self.pipeline.process_chunk(audio_data)
                except Exception as e:
                    print(f"[Pipeline] Error: {e}")

            # Periodic status update
            if self.chunks_received % 100 == 0:
                elapsed = time.time() - self.start_time
                rate = self.bytes_received / elapsed / 1024
                print(
                    f"  Receiving: {self.chunks_received} chunks, "
                    f"{self.bytes_received / 1024:.1f} KB, "
                    f"{rate:.1f} KB/s"
                )

    def _disconnect_client(self):
        """Handle client disconnection"""
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        self.client_connected = False
        self.recording = False

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n  ESP32 disconnected")
        print(
            f"  Session stats: {self.chunks_received} chunks, "
            f"{self.bytes_received / 1024:.1f} KB in {elapsed:.1f}s"
        )
        print("\n  Waiting for new connection...")

    def stop(self):
        """Stop the server"""
        print("\n" + "-" * 60)
        print("  Stopping server...")
        self.running = False
        self._disconnect_client()

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        if self.pipeline:
            self.pipeline.shutdown()

        print("  Server stopped")

    def handle_command(self, command, args=""):
        """Handle control commands"""
        self.log(f"Command received: {command}")

        if command == "START":
            self.recording = True
            print("\n  [REC] Recording started")
            if self.pipeline:
                self.pipeline.start_session()

        elif command == "STOP":
            if self.recording:
                self.recording = False
                print("\n  [REC] Recording stopped")
                if self.pipeline:
                    self.pipeline.end_session()

        elif command == "HELLO":
            print(f"\n  ESP32 identified: {args}")

        elif command == "PING":
            # Respond to ping
            pass

        else:
            self.log(f"Unknown command: {command}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Audio Receiver for ESP32-S3 Audio Streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", "-H", default="0.0.0.0", help="Bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="TCP port (default: 8000)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Create and start receiver
    receiver = AudioReceiver(host=args.host, port=args.port, verbose=args.verbose)

    try:
        receiver.start()
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
