#!/usr/bin/env python3
"""
screen_server.py — UMA Game PC screen capture service  (v5: bg capture + WS)
============================================================================
Runs on the game PC. Captures the game screen and serves frames to the agent.

What changed from v4:
  - BACKGROUND CAPTURE THREAD. A dedicated thread grabs the screen continuously
    into a single "latest frame" buffer. Capture is now OFF the request path:
    when the agent asks for a frame it gets the freshest one already in memory,
    instead of waiting for a fresh grab+encode synchronously.
  - WEBSOCKET frame channel (/ws), request-driven: the agent sends a 1-byte
    request, the server replies with the latest frame as JPEG bytes. Request-
    driven (not free-running push) so frames never pile up in the socket buffer
    and go stale — the agent always pulls the newest available frame.
  - mss instance is created INSIDE the capture thread. mss is not thread-safe;
    sharing one instance across threads corrupts captures.
  - HTTP /screenshot, /region, /health retained for debugging.

Install:
    pip install flask flask-sock mss pillow

Run:
    python3 screen_server.py

Env:
    GAME_MONITOR   monitor index (default 1)
    CAPTURE_FPS    background capture cap (default 30) — trades CPU for freshness
"""

import io, os, time, threading
import mss
from flask import Flask, request, send_file, jsonify
from flask_sock import Sock
from PIL import Image

app  = Flask(__name__)
sock = Sock(app)

GAME_MONITOR = int(os.environ.get("GAME_MONITOR", "1"))
CAPTURE_FPS  = float(os.environ.get("CAPTURE_FPS", "30"))

print(f"Screen server starting on 0.0.0.0:5003  (HTTP + WS /ws)")
print(f"  monitor={GAME_MONITOR}  capture_fps={CAPTURE_FPS}\n")

# ── Background capture → single latest-frame buffer ───────────────────────────

_frame_lock = threading.Lock()
_latest_rgb = None          # PIL.Image (RGB)
_latest_ts  = 0.0
_mon_size   = (0, 0)

def _capture_loop():
    global _latest_rgb, _latest_ts, _mon_size
    local_sct = mss.mss()                       # thread-local — do NOT share
    mon = local_sct.monitors[GAME_MONITOR]
    _mon_size = (mon["width"], mon["height"])
    period = 1.0 / CAPTURE_FPS if CAPTURE_FPS > 0 else 0.0
    while True:
        t0  = time.time()
        raw = local_sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        with _frame_lock:
            _latest_rgb = img
            _latest_ts  = time.time()
        dt = time.time() - t0
        if period:
            time.sleep(max(0.0, period - dt))

threading.Thread(target=_capture_loop, daemon=True).start()

def _get_latest():
    with _frame_lock:
        return _latest_rgb, _latest_ts

def _encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

# ── WebSocket frame channel ───────────────────────────────────────────────────

@sock.route('/ws')
def ws(ws):
    """
    Request-driven frames. The agent sends any short message to request the
    latest frame; the server replies with JPEG bytes (binary). Optional text
    'q=NN' sets JPEG quality for that request (default 85).
    """
    while True:
        req = ws.receive()
        if req is None:
            break
        quality = 85
        if isinstance(req, str) and req.startswith('q='):
            try:
                quality = max(40, min(95, int(req[2:])))
            except ValueError:
                pass
        img, _ts = _get_latest()
        if img is None:
            ws.send(b'')                      # not ready yet
            continue
        ws.send(_encode_jpeg(img, quality))   # binary frame

# ── HTTP endpoints (debug / fallback) ─────────────────────────────────────────

@app.route('/health')
def health():
    img, ts = _get_latest()
    age_ms = int((time.time() - ts) * 1000) if img is not None else -1
    return jsonify({
        "status":     "ok",
        "service":    "screen",
        "monitor":    GAME_MONITOR,
        "size":       f"{_mon_size[0]}x{_mon_size[1]}",
        "frame_age_ms": age_ms,
        "capture_fps":  CAPTURE_FPS,
    })

@app.route('/screenshot')
def screenshot():
    quality = int(request.args.get('quality', 85))
    img, _ = _get_latest()
    if img is None:
        return jsonify({"error": "no frame yet"}), 503
    return send_file(io.BytesIO(_encode_jpeg(img, quality)), mimetype="image/jpeg")

@app.route('/region')
def region():
    try:
        l = int(request.args['l']); t = int(request.args['t'])
        w = int(request.args['w']); h = int(request.args['h'])
    except (KeyError, ValueError):
        return jsonify({"error": "missing l,t,w,h params"}), 400
    quality = int(request.args.get('quality', 90))
    img, _ = _get_latest()
    if img is None:
        return jsonify({"error": "no frame yet"}), 503
    crop = img.crop((l, t, l + w, t + h))
    return send_file(io.BytesIO(_encode_jpeg(crop, quality)), mimetype="image/jpeg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
