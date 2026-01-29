"""
LED Status Indicator for ESP32-S3
=================================

Status LED Patterns:
- Red blinking:     WiFi connecting
- Red solid:        WiFi connected, waiting
- Blue blinking:    Recording audio
- Blue solid:       Audio streaming
- Green blinking:   Data sent successfully
- Green solid:      System ready
- Yellow blinking:  Error (retry)
"""

from machine import Pin
import time

# LED Pin Configuration
# DevKitC-1 has a built-in RGB LED on GPIO 48
LED_R = 48  # Red
LED_G = 47  # Green
LED_B = 45  # Blue

# Status States
STATE_WIFI_CONNECTING = "wifi_connecting"
STATE_WIFI_CONNECTED = "wifi_connected"
STATE_READY = "ready"
STATE_RECORDING = "recording"
STATE_STREAMING = "streaming"
STATE_SENDING = "sending"
STATE_ERROR = "error"


class LEDIndicator:
    def __init__(self):
        """Initialize RGB LED"""
        # Configure LED pins as outputs
        self.led_r = Pin(LED_R, Pin.OUT)
        self.led_g = Pin(LED_G, Pin.OUT)
        self.led_b = Pin(LED_B, Pin.OUT)

        # Turn off all LEDs initially
        self.led_r.value(1)  # Common anode, 1 = off
        self.led_g.value(1)
        self.led_b.value(1)

        self.current_state = None
        self.blink_task = None

    def set_color(self, r, g, b):
        """Set LED color (0 = on, 1 = off for common anode)"""
        self.led_r.value(0 if r else 1)
        self.led_g.value(0 if g else 1)
        self.led_b.value(0 if b else 1)

    def red(self):
        """Set LED to red"""
        self.set_color(1, 0, 0)

    def green(self):
        """Set LED to green"""
        self.set_color(0, 1, 0)

    def blue(self):
        """Set LED to blue"""
        self.set_color(0, 0, 1)

    def yellow(self):
        """Set LED to yellow (red + green)"""
        self.set_color(1, 1, 0)

    def cyan(self):
        """Set LED to cyan (green + blue)"""
        self.set_color(0, 1, 1)

    def magenta(self):
        """Set LED to magenta (red + blue)"""
        self.set_color(1, 0, 1)

    def white(self):
        """Set LED to white (all on)"""
        self.set_color(1, 1, 1)

    def off(self):
        """Turn off LED"""
        self.set_color(0, 0, 0)

    def blink(self, color_func, count=3, interval=0.2):
        """Blink LED a specified number of times"""
        for _ in range(count):
            color_func()
            time.sleep(interval)
            self.off()
            time.sleep(interval)

    def set_state(self, state):
        """Set LED state and update display"""
        self.current_state = state

        if state == STATE_WIFI_CONNECTING:
            self.blink(self.red, count=255, interval=0.5)  # Continuous slow blink
        elif state == STATE_WIFI_CONNECTED:
            self.red()
        elif state == STATE_READY:
            self.green()
        elif state == STATE_RECORDING:
            self.blink(self.blue, count=255, interval=0.3)  # Continuous blink
        elif state == STATE_STREAMING:
            self.blue()
        elif state == STATE_SENDING:
            self.blink(self.green, count=1, interval=0.1)
        elif state == STATE_ERROR:
            self.blink(self.yellow, count=255, interval=0.5)  # Error pattern

    def success_pulse(self):
        """Quick success pulse"""
        self.green()
        time.sleep(0.5)
        self.off()

    def error_pulse(self):
        """Quick error pulse"""
        self.yellow()
        time.sleep(0.5)
        self.off()


# Global LED indicator instance
led = LEDIndicator()


def init_led():
    """Initialize LED and show startup pattern"""
    print("[LED] Initializing RGB LED...")
    print(f"  - Red:   GPIO{LED_R}")
    print(f"  - Green: GPIO{LED_G}")
    print(f"  - Blue:  GPIO{LED_B}")

    # Startup pattern
    led.red()
    time.sleep(0.2)
    led.green()
    time.sleep(0.2)
    led.blue()
    time.sleep(0.2)
    led.off()
    print("[LED] Ready")
    return led
