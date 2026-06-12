#!/usr/bin/env python3
"""
controller_server.py — UMA Game PC control service  (v5: set-and-hold + WS)
===========================================================================
Runs on the game PC. Drives a virtual Xbox 360 controller.

What changed from v4:
  - SET-AND-HOLD model. Stick commands set the sticks and return IMMEDIATELY.
    No more sleep-then-zero inside the request. The agent is responsible for
    refreshing the stick vector every loop tick; between ticks the sticks hold.
  - WATCHDOG. If no stick command arrives within WATCHDOG_MS, the sticks are
    zeroed automatically. This is the safety net for the set-and-hold model:
    if the agent crashes or the socket drops, Geralt stops instead of walking
    into a wall forever.
  - WEBSOCKET hot path (/ws). Fire-and-forget stick + button messages over a
    persistent connection — no per-call TCP handshake, no response to wait on.
  - HTTP endpoints retained for debugging (curl-able) and as a fallback,
    including a new set-and-hold /sticks. The legacy blocking /navigate is
    kept only for manual testing.

Install:
    pip install flask flask-sock vgamepad

Run BEFORE launching Witcher 3:
    python3 controller_server.py

Then launch W3 and disable Steam Input for the title.
"""

import time, json, threading, subprocess
from flask import Flask, request, jsonify
from flask_sock import Sock
import vgamepad as vg

app  = Flask(__name__)
sock = Sock(app)

pad = vg.VX360Gamepad()
pad.update()
print("Virtual controller created.")
print("Launch Witcher 3 now, then disable Steam Input for W3.")
print("Server starting on 0.0.0.0:5002  (HTTP + WS /ws)\n")

# ── Safety / watchdog ─────────────────────────────────────────────────────────
WATCHDOG_MS = 800          # zero sticks if no stick cmd within this window.
                           # MUST exceed the agent's worst-case tick time, or the
                           # sticks get zeroed between updates and motion stutters.
                           # (Agent ticks ~300-750 ms; 800 leaves margin.)

_lock        = threading.Lock()
_last_cmd_ts = time.time()
_sticks_live = False       # are the sticks currently non-zero?

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
    "rs":       vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
}

# ── Low-level pad ops (all under _lock) ───────────────────────────────────────

def _apply_sticks(lx, ly, rx, ry):
    """Set both sticks, stamp the time, track live state. Caller holds nothing."""
    global _last_cmd_ts, _sticks_live
    with _lock:
        pad.left_joystick_float(x_value_float=lx, y_value_float=ly)
        pad.right_joystick_float(x_value_float=rx, y_value_float=ry)
        pad.update()
        _last_cmd_ts = time.time()
        _sticks_live = (abs(lx) + abs(ly) + abs(rx) + abs(ry)) > 1e-3

def _zero_sticks():
    global _sticks_live
    with _lock:
        pad.left_joystick_float(0.0, 0.0)
        pad.right_joystick_float(0.0, 0.0)
        pad.update()
        _sticks_live = False

def _press(name, dur):
    btn = BUTTONS.get(name)
    if btn is None:
        return False
    with _lock:
        pad.press_button(button=btn); pad.update()
    time.sleep(dur)
    with _lock:
        pad.release_button(button=btn); pad.update()
    return True

def _full_reset():
    global _sticks_live
    with _lock:
        pad.left_joystick_float(0.0, 0.0)
        pad.right_joystick_float(0.0, 0.0)
        pad.reset()
        pad.update()
        _sticks_live = False

def _watchdog():
    while True:
        time.sleep(0.05)
        with _lock:
            live = _sticks_live
            age  = (time.time() - _last_cmd_ts) * 1000.0
        if live and age > WATCHDOG_MS:
            _zero_sticks()
            print(f"  ⚠  watchdog: no stick cmd for {age:.0f} ms — sticks zeroed")

threading.Thread(target=_watchdog, daemon=True).start()

# ── WebSocket hot path ────────────────────────────────────────────────────────

@sock.route('/ws')
def ws(ws):
    """
    Fire-and-forget control. Messages (JSON text):
      {"type":"sticks","left_x":..,"left_y":..,"right_x":..,"right_y":..}
          Sets sticks, returns nothing. Values already in W3 convention
          (the agent inverts Y before sending). Hold persists until the next
          message or the watchdog fires.
      {"type":"button","button":"x","duration":0.12}
      {"type":"release"}                 full reset
      {"type":"ping"}                    -> {"pong":true}
    """
    while True:
        raw = ws.receive()
        if raw is None:
            break
        try:
            m = json.loads(raw)
        except Exception:
            continue
        t = m.get('type')
        if t == 'sticks':
            _apply_sticks(float(m.get('left_x', 0.0)), float(m.get('left_y', 0.0)),
                          float(m.get('right_x', 0.0)), float(m.get('right_y', 0.0)))
        elif t == 'button':
            _press(m.get('button', 'a').lower(), float(m.get('duration', 0.12)))
        elif t == 'release':
            _full_reset()
        elif t == 'ping':
            ws.send(json.dumps({"pong": True}))
    # client gone — fail safe
    _zero_sticks()

# ── HTTP endpoints (debug / fallback) ─────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "controller", "model": "set-and-hold"})

@app.route('/sticks', methods=['POST'])
def sticks_http():
    """Set-and-hold over HTTP (returns immediately). Mirrors the WS 'sticks' msg."""
    d = request.json or {}
    _apply_sticks(float(d.get('left_x', 0.0)), float(d.get('left_y', 0.0)),
                  float(d.get('right_x', 0.0)), float(d.get('right_y', 0.0)))
    return jsonify({"ok": True})

@app.route('/button', methods=['POST'])
def button_http():
    d   = request.json or {}
    name = d.get('button', 'a').lower()
    dur  = float(d.get('duration', 0.12))
    if not _press(name, dur):
        return jsonify({"error": f"unknown button: {name}"}), 400
    return jsonify({"ok": True, "button": name})

@app.route('/release', methods=['POST'])
def release_http():
    _full_reset()
    return jsonify({"ok": True})

@app.route('/navigate', methods=['POST'])
def navigate_legacy():
    """
    LEGACY blocking move — kept for manual testing only. Holds the sticks for
    `duration` then zeroes. Do NOT use this from the continuous agent loop.
    """
    d        = request.json or {}
    _apply_sticks(float(d.get('left_x', 0.0)),  float(d.get('left_y', 1.0)),
                  float(d.get('right_x', 0.0)), float(d.get('right_y', 0.0)))
    time.sleep(float(d.get('duration', 0.6)))
    _zero_sticks()
    return jsonify({"ok": True})

@app.route('/key', methods=['POST'])
def key_http():
    """Keyboard fallback via xdotool (X11/Linux only)."""
    d   = request.json or {}
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
    # threaded=True so the WS connection, HTTP debug calls, and watchdog coexist
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
