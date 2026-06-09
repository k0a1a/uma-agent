#!/usr/bin/env python3
"""
screen_server.py — UMA Game PC screen capture service
======================================================
Runs on the game PC. Captures the game screen and serves
frames to the laptop agent on demand.

Install:
    pip install flask mss pillow

Run:
    python3 screen_server.py

Endpoints:
    GET  /screenshot          — full screen as PNG
    GET  /region?l=&t=&w=&h= — specific region as PNG
    GET  /health              — connectivity check
"""

import io
import mss
from flask import Flask, request, send_file, jsonify
from PIL import Image

app = Flask(__name__)
sct = mss.MSS()

GAME_MONITOR = 1   # adjust if game is on second monitor

print("Screen server starting on 0.0.0.0:5003\n")

def capture(region=None) -> Image.Image:
    raw = sct.grab(sct.monitors[GAME_MONITOR])
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    if region:
        l, t, w, h = region
        img = img.crop((l, t, l+w, t+h))
    return img

def img_to_response(img: Image.Image, quality: int = 85):
    buf = io.BytesIO()
    # use JPEG for speed, PNG for quality
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

@app.route('/health')
def health():
    mon = sct.monitors[GAME_MONITOR]
    return jsonify({
        "status":   "ok",
        "service":  "screen",
        "monitor":  GAME_MONITOR,
        "size":     f"{mon['width']}x{mon['height']}",
    })

@app.route('/screenshot')
def screenshot():
    """Full screen capture."""
    quality = int(request.args.get('quality', 85))
    img = capture()
    return img_to_response(img, quality)

@app.route('/region')
def region():
    """
    Capture a specific screen region.
    Params: l (left), t (top), w (width), h (height), quality
    Example: /region?l=1666&t=65&w=207&h=200
    """
    try:
        l = int(request.args['l'])
        t = int(request.args['t'])
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return jsonify({"error": "missing l,t,w,h params"}), 400

    quality = int(request.args.get('quality', 90))
    img = capture(region=(l, t, w, h))
    return img_to_response(img, quality)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False)
