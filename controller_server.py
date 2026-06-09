#!/usr/bin/env python3
"""
controller_server.py — UMA Game PC service
===========================================
Runs on the game PC. Exposes a simple HTTP API for controller
and keyboard input. Receives commands from the laptop agent.

Install:
    pip install flask vgamepad

Run BEFORE launching Witcher 3:
    python3 controller_server.py

Endpoints:
    POST /stick     — move analog stick
    POST /button    — press/release button
    POST /key       — keyboard fallback
    POST /release   — release everything
    GET  /health    — connectivity check
"""

import time, subprocess
from flask import Flask, request, jsonify
import vgamepad as vg

app = Flask(__name__)

# Virtual Xbox 360 controller
pad = vg.VX360Gamepad()
pad.update()
print("Virtual controller created.")
print("Launch Witcher 3 now, then disable Steam Input for W3.")
print("Server starting on 0.0.0.0:5002\n")

# Button name → vgamepad constant
BUTTONS = {
    "a":        vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "b":        vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "x":        vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "y":        vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
    "confirm":  vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
    "cancel":   vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "dodge":    vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
    "attack":   vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
    "start":    vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
    "back":     vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
    "dpad_up":  vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
    "dpad_down":vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
    "dpad_left":vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
    "dpad_right":vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
    "lb":       vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
    "rb":       vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
}

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "controller"})

@app.route('/stick', methods=['POST'])
def stick():
    """
    Move an analog stick.
    Body: {"stick": "left|right", "x": float, "y": float, "duration": float}
    x, y range: -1.0 to 1.0
    """
    d = request.json
    x = float(d.get('x', 0.0))
    y = float(d.get('y', 0.0))
    dur = float(d.get('duration', 0.0))

    if d.get('stick') == 'left':
        pad.left_joystick_float(x_value_float=x, y_value_float=y)
    else:
        pad.right_joystick_float(x_value_float=x, y_value_float=y)
    pad.update()

    if dur > 0:
        time.sleep(dur)
        if d.get('stick') == 'left':
            pad.left_joystick_float(0.0, 0.0)
        else:
            pad.right_joystick_float(0.0, 0.0)
        pad.update()

    return jsonify({"ok": True})

@app.route('/button', methods=['POST'])
def button():
    """
    Press and release a button.
    Body: {"button": "a|b|x|y|confirm|...", "duration": float}
    """
    d   = request.json
    name = d.get('button', 'a').lower()
    dur  = float(d.get('duration', 0.1))

    btn = BUTTONS.get(name)
    if btn is None:
        return jsonify({"error": f"unknown button: {name}"}), 400

    pad.press_button(button=btn)
    pad.update()
    time.sleep(dur)
    pad.release_button(button=btn)
    pad.update()
    time.sleep(0.05)

    return jsonify({"ok": True, "button": name, "duration": dur})

@app.route('/release', methods=['POST'])
def release():
    """Release all inputs — emergency stop."""
    pad.left_joystick_float(0.0, 0.0)
    pad.right_joystick_float(0.0, 0.0)
    pad.reset()
    pad.update()
    return jsonify({"ok": True})

@app.route('/key', methods=['POST'])
def key():
    """
    Keyboard fallback via xdotool (for things controller can't do easily).
    Body: {"key": "space|e|...", "duration": float}
    """
    d   = request.json
    k   = d.get('key', 'space')
    dur = float(d.get('duration', 0.0))

    if dur > 0:
        subprocess.run(["xdotool", "keydown", k], check=False)
        time.sleep(dur)
        subprocess.run(["xdotool", "keyup", k], check=False)
    else:
        subprocess.run(["xdotool", "key", k], check=False)

    return jsonify({"ok": True, "key": k})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False)
