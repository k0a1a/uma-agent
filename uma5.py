#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v5.0
===================================
Plays the Witcher 3 through vision only.

What changed from v4 (the four things that compound):
  1. CONTINUOUS LOOP. The control servers are now set-and-hold; this loop runs
     at TARGET_HZ, re-sending the stick vector every tick. No more walk-0.5s-
     then-freeze-0.7s stutter. Corrections are small and frequent, not large
     and blind.
  2. ORDERED PATH, not a point cloud. Minimap dots are walked into a polyline
     (nearest-neighbour from centre outward), so we can reason about the path's
     shape instead of chasing the single nearest pixel.
  3. PURE-PURSUIT + HEADING. We aim at a lookahead point a fixed arc-length L
     along the ordered path (kills corner-cutting and the dead-ahead-but-path-
     curves miss), and add a tangent/heading term (stops drift off straights).
  4. ALIGNMENT-GATED SPEED + SMOOTHING. Forward speed falls off as cos²(error)
     and hard-gates near zero on sharp turns, so we rotate-then-go instead of
     plowing through bends. RS is exponentially smoothed to stop snap/oscillation.

Plus: brightness mask keeps tinted dots (v4 fix) and a circularity guard rejects
the yellow destination-ring fragments that the size filter alone can leak.

Transport:
  - screen  : WebSocket, request-driven (freshest frame, capture off the path)
  - control : WebSocket, fire-and-forget set-and-hold
  - OCR/VLM : HTTP via a keep-alive Session (request-response by nature)

Architecture:
  b550.local          — W3 + controller_server.py (5002) + screen_server.py (5003)
  beacon.x.k0a1a.net  — LM Studio Qwen2.5-VL-7B (1234) + ocr_server.py (5001)
  laptop              — this script

Install:
    pip install opencv-python numpy pillow requests openai scikit-image websocket-client

Usage:
    python3 uma5.py
"""

import os, sys, time, base64, io, math, json, random
from collections import deque
from typing import Optional, Tuple, List

import cv2
import numpy as np
import requests
import websocket                         # websocket-client (sync)
from PIL import Image, ImageEnhance
from openai import OpenAI
from skimage.metrics import structural_similarity as ssim

# ── Service endpoints ─────────────────────────────────────────────────────────

GAME_PC = os.environ.get("GAME_PC", "b365.local")
BEACON  = os.environ.get("BEACON",  "beacon.x.k0a1a.net")

SCREEN_WS_URL  = f"ws://{GAME_PC}:5003/ws"
CONTROL_WS_URL = f"ws://{GAME_PC}:5002/ws"
SCREEN_HTTP    = f"http://{GAME_PC}:5003"
CONTROL_HTTP   = f"http://{GAME_PC}:5002"
LM_STUDIO_URL  = f"http://{BEACON}:1234/v1"
OCR_URL        = f"http://{BEACON}:5001/ocr"
LM_MODEL       = os.environ.get("LM_MODEL", "local-model")

http = requests.Session()                # keep-alive for OCR/VLM

# ── Loop / motion config ──────────────────────────────────────────────────────

TARGET_HZ      = 6.0          # control loop rate
TICK           = 1.0 / TARGET_HZ
JPEG_QUALITY   = 80           # screen frame quality requested over WS

# ── Minimap dot detection ─────────────────────────────────────────────────────

DOT_BRIGHTNESS  = 160         # min R and G for path dots (white AND yellow pass)
DOT_EXCLUDE_R   = 25          # blank Geralt's arrow at centre
DOT_BORDER_PAD  = 8           # ignore bright rim of the minimap
DOT_AREA_MIN    = 1
DOT_AREA_MAX    = 18
DOT_MIN_EXTENT  = 0.45        # area / bbox — round blobs pass, thin arcs fail
DOT_MAX_ASPECT  = 3.0         # bbox aspect — rejects elongated ring fragments
DOT_GUARD_AREA  = 4           # only apply extent/aspect guard at/above this area
PATH_MAX_GAP    = 26          # px; a bigger hop ends the polyline (separate cluster)

# ── Steering law ──────────────────────────────────────────────────────────────

LOOKAHEAD_L    = 55.0         # px arc-length to the pursuit point (corner knob)
K_CROSS        = 1.0          # weight on pursuit (where the lookahead sits L/R)
K_HEAD         = 0.55         # weight on path tangent near Geralt (anti-drift)
STEER_CLAMP    = 90.0         # deg, clamp on blended steer angle
STEER_FULL_DEG = 60.0         # deg that maps RS to full deflection
RS_EMA_ALPHA   = 0.5          # right-stick smoothing (1 = no smoothing)
V_BASE         = 1.0          # base forward speed
V_CRAWL        = 0.18         # forward speed during a hard turn
HARD_TURN_DEG  = 55.0         # above this heading error, crawl
TANGENT_K      = 4            # dots used to estimate local path tangent

# ── Stuck detection ───────────────────────────────────────────────────────────

SSIM_THRESHOLD = 0.97
STUCK_TICKS    = 3

# ── Combat detection ──────────────────────────────────────────────────────────
# An enemy HP bar is a wide, thin, CONTIGUOUS horizontal red run. Scenery red
# (roofs, banners, sky) is scattered or broken into short spans, so we test for
# a contiguous span rather than a raw pixel count.
COMBAT_MIN_SPAN = 120         # px; min contiguous horizontal red run = HP bar
COMBAT_CONFIRM  = 2           # consecutive raw-COMBAT frames before acting

# ── HUD regions (left, top, width, height) — 1920×1080, Next Gen W3, B550 ─────

MINIMAP_REGION  = (1660,  36, 213, 215)
QUEST_REGION    = (1220, 225, 340, 100)
SUBTITLE_REGION = ( 450, 655, 570,  55)
CHOICE_REGION   = ( 865, 500, 360, 125)
INTERACT_REGION = (  55, 350, 230, 120)
ENEMY_HP_REGION = ( 650,  44, 580,  24)

# ══════════════════════════════════════════════════════════════════════════════
# Transport: WebSocket clients with lazy reconnect
# ══════════════════════════════════════════════════════════════════════════════

_screen_ws:  Optional[websocket.WebSocket] = None
_control_ws: Optional[websocket.WebSocket] = None

def _connect_screen():
    global _screen_ws
    _screen_ws = websocket.create_connection(SCREEN_WS_URL, timeout=5)

def _connect_control():
    global _control_ws
    _control_ws = websocket.create_connection(CONTROL_WS_URL, timeout=5)

def get_screen() -> Optional[Image.Image]:
    """Pull the freshest frame over WS. Reconnect once on failure."""
    global _screen_ws
    for attempt in (1, 2):
        try:
            if _screen_ws is None:
                _connect_screen()
            _screen_ws.send(f"q={JPEG_QUALITY}")
            data = _screen_ws.recv()
            if isinstance(data, str):
                data = data.encode("latin-1")
            if not data:
                return None
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            print(f"  ⚠  screen ws ({attempt}): {e}")
            try: _screen_ws.close()
            except Exception: pass
            _screen_ws = None
            time.sleep(0.3)
    return None

def _control_send(msg: dict):
    global _control_ws
    for attempt in (1, 2):
        try:
            if _control_ws is None:
                _connect_control()
            _control_ws.send(json.dumps(msg))
            return
        except Exception as e:
            print(f"  ⚠  control ws ({attempt}): {e}")
            try: _control_ws.close()
            except Exception: pass
            _control_ws = None
            time.sleep(0.2)

def move(rs_x: float, ls_y: float):
    """
    Set-and-hold. Right stick X = camera rotation, left stick Y = forward.
    W3 inverts Y (negative = forward), so we negate here. Left stick X stays 0
    (no strafe). Returns immediately; the loop refreshes this every tick.
    """
    _control_send({"type": "sticks",
                   "left_x": 0.0, "left_y": -ls_y,
                   "right_x": float(np.clip(rs_x, -1, 1)), "right_y": 0.0})

def button(name: str, duration: float = 0.12):
    _control_send({"type": "button", "button": name, "duration": duration})

def release():
    _control_send({"type": "release"})

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def crop(screen: Image.Image, region: tuple) -> Image.Image:
    l, t, w, h = region
    return screen.crop((l, t, l + w, t + h))

# ══════════════════════════════════════════════════════════════════════════════
# Minimap → ordered path → steering
# ══════════════════════════════════════════════════════════════════════════════

def extract_dots(screen: Image.Image):
    """
    Returns (dots, centre, mask) where dots is a list of (x,y) centroids in
    minimap-local pixels. Brightness mask (keeps tinted dots) + size filter +
    circularity guard (rejects yellow-ring fragments).
    """
    mm   = np.array(crop(screen, MINIMAP_REGION))
    h, w = mm.shape[:2]
    cx, cy = w // 2, h // 2

    r = mm[:, :, 0].astype(int); g = mm[:, :, 1].astype(int)
    mask = ((r > DOT_BRIGHTNESS) & (g > DOT_BRIGHTNESS)).astype(np.uint8) * 255

    cv2.circle(mask, (cx, cy), DOT_EXCLUDE_R, 0, -1)
    border = np.zeros_like(mask)
    cv2.circle(border, (cx, cy), min(cx, cy) - DOT_BORDER_PAD, 255, -1)
    mask = cv2.bitwise_and(mask, border)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    dots = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (DOT_AREA_MIN <= area <= DOT_AREA_MAX):
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]; bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= DOT_GUARD_AREA:                      # judge only sizeable blobs
            extent = area / float(bw * bh) if bw * bh else 0.0
            aspect = max(bw, bh) / float(min(bw, bh)) if min(bw, bh) else 99.0
            if extent < DOT_MIN_EXTENT or aspect > DOT_MAX_ASPECT:
                continue
        dots.append((float(cents[i][0]), float(cents[i][1])))
        keep[labels == i] = 255

    Image.fromarray(mm).save("minimap_raw.png")
    Image.fromarray(keep).save("minimap_mask.png")
    return dots, (cx, cy), keep

def order_path(dots: List[Tuple[float, float]], origin: Tuple[float, float]):
    """Greedy nearest-neighbour walk from origin outward; stop on a large gap."""
    if not dots:
        return []
    remaining = dots[:]
    path, cur = [], origin
    while remaining:
        d2 = [(p[0] - cur[0])**2 + (p[1] - cur[1])**2 for p in remaining]
        j  = int(np.argmin(d2))
        if path and d2[j] > PATH_MAX_GAP**2:            # left the contiguous path
            break
        cur = remaining.pop(j)
        path.append(cur)
    return path

def lookahead_point(path, origin, L):
    """First point at cumulative arc-length >= L; else the far end of the path."""
    if not path:
        return None
    acc, prev = 0.0, origin
    for p in path:
        acc += math.hypot(p[0] - prev[0], p[1] - prev[1])
        if acc >= L:
            return p
        prev = p
    return path[-1]

def _ang(origin, p):
    """Angle (deg) from minimap-up to p, +ve = right. up = Geralt-forward."""
    return math.degrees(math.atan2(p[0] - origin[0], -(p[1] - origin[1])))

def read_minimap(screen: Image.Image):
    """
    Returns (steer_deg, pursuit_deg, n_dots, source) or (None, None, 0, 'none').
    steer_deg blends pursuit (where the lookahead sits) with the local path
    tangent (which way the path is heading near Geralt).
    """
    dots, origin, _ = extract_dots(screen)
    path = order_path(dots, origin)

    if len(path) >= 2:
        look    = lookahead_point(path, origin, LOOKAHEAD_L)
        pursuit = _ang(origin, look)
        k       = min(TANGENT_K, len(path) - 1)
        tangent = _ang(path[0], path[k])               # local path direction
        steer   = float(np.clip(K_CROSS * pursuit + K_HEAD * tangent,
                                -STEER_CLAMP, STEER_CLAMP))
        return steer, pursuit, len(path), "path"

    if len(path) == 1:                                 # single dot — pure pursuit
        pursuit = _ang(origin, path[0])
        return pursuit, pursuit, 1, "dot"

    # Fall back to yellow quest marker centroid
    mm  = np.array(crop(screen, MINIMAP_REGION))
    h, w = mm.shape[:2]; cx, cy = w // 2, h // 2
    hsv = cv2.cvtColor(mm, cv2.COLOR_RGB2HSV)
    ym  = cv2.inRange(hsv, np.array([15, 150, 150]), np.array([35, 255, 255]))
    cv2.circle(ym, (cx, cy), DOT_EXCLUDE_R, 0, -1)
    yys, yxs = np.where(ym > 0)
    if len(yxs) >= 3:
        pursuit = _ang((cx, cy), (float(yxs.mean()), float(yys.mean())))
        return pursuit, pursuit, 0, "marker"

    return None, None, 0, "none"

def steer_to_sticks(steer_deg: float, pursuit_deg: float) -> Tuple[float, float]:
    """
    RS from the blended steer angle (linear, clamped). LS gated on alignment:
    cos²(pursuit) so speed falls off into turns, with a hard crawl past
    HARD_TURN_DEG. Result: rotate-then-go on sharp bends, full speed on straights.
    """
    rs_x = float(np.clip(steer_deg / STEER_FULL_DEG, -1.0, 1.0))
    align = math.cos(math.radians(pursuit_deg))
    ls_y  = V_BASE * max(0.0, align) ** 2
    if abs(pursuit_deg) > HARD_TURN_DEG:
        ls_y = min(ls_y, V_CRAWL)
    return rs_x, ls_y

# ══════════════════════════════════════════════════════════════════════════════
# Mode detection  (OCR only when pixel heuristics already fire)
# ══════════════════════════════════════════════════════════════════════════════

def px(img, x, y):
    return img.getpixel((x, y))[:3]

def _max_true_run(b: np.ndarray) -> int:
    """Longest contiguous run of True in a 1D boolean array."""
    best = run = 0
    for v in b:
        run = run + 1 if v else 0
        if run > best:
            best = run
    return best

def detect_mode(screen: Image.Image) -> str:
    arr = np.array(screen)

    pts = [(960, 540), (480, 270), (1440, 270), (480, 810), (1440, 810)]
    if all(sum(px(screen, x, y)) / 3 < 20 for x, y in pts):
        return "LOADING"

    if (sum(px(screen, 960, 40)) / 3   < 15 and
        sum(px(screen, 960, 1040)) / 3 < 15 and
        sum(px(screen, 960, 540)) / 3  > 30):
        return "CUTSCENE"

    # CHOICES — cheap pixel pre-check first; OCR only to confirm (keeps the hot
    # loop OCR-free during normal walking).
    l, t, w, h = CHOICE_REGION
    cg = cv2.cvtColor(arr[t:t + h, l:l + w], cv2.COLOR_RGB2GRAY)
    if ((cg > 160).sum() / cg.size > 0.04 and
        ((cg > 15) & (cg < 90)).sum() / cg.size > 0.15 and
        cg.mean() < 85):
        if len(ocr(crop(screen, CHOICE_REGION))) > 10:
            return "CHOICES"

    l, t, w, h = ENEMY_HP_REGION
    hsv = cv2.cvtColor(arr[t:t + h, l:l + w], cv2.COLOR_RGB2HSV)
    red = cv2.bitwise_or(                                   # red wraps the hue circle
        cv2.inRange(hsv, np.array([0,   150, 90]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 150, 90]), np.array([180, 255, 255])))
    active = (red > 0).sum(axis=0) >= 2                     # columns holding a thin bar
    if _max_true_run(active) >= COMBAT_MIN_SPAN:            # one wide contiguous run
        return "COMBAT"

    return "EXPLORATION"

# ══════════════════════════════════════════════════════════════════════════════
# OCR + VLM (HTTP, Session keep-alive)
# ══════════════════════════════════════════════════════════════════════════════

def ocr(img: Image.Image) -> str:
    img = ImageEnhance.Contrast(img.convert("L")).enhance(2.5).convert("RGB")
    try:
        r = http.post(OCR_URL, json={"image": to_b64(img)}, timeout=5)
        return r.json().get("text", "").strip()
    except Exception as e:
        print(f"  ⚠  OCR: {e}")
        return ""

llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

def ask(prompt: str, system: str, image: Optional[Image.Image] = None,
        temperature: float = 0.3, max_tokens: int = 64) -> str:
    content = []
    if image is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{to_b64(image)}"}})
    content.append({"type": "text", "text": prompt})
    try:
        r = llm.chat.completions.create(
            model=LM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": content}],
            max_tokens=max_tokens, temperature=temperature)
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠  LLM: {e}")
        return ""

CHOICES_SYSTEM = (
    "You are Geralt of Rivia in Witcher 3. Follow the main story quest.\n"
    "Dialogue choices are visible. Choose what best advances the quest.\n"
    "Reply ONLY:\nDIRECTION: <up|down|confirm>\nREASON: one sentence\n\n"
    "up = previous choice, down = next choice, confirm = select (A)"
)

def parse_direction(response: str) -> str:
    for line in response.strip().splitlines():
        if "DIRECTION:" in line.upper():
            return line.split(":", 1)[1].strip().lower()
    return "confirm"

def navigate_choice(direction: str):
    if direction == "up":
        move(0.0, 1.0)
    elif direction == "down":
        move(0.0, -1.0)
    else:
        button("a")
    time.sleep(0.2)
    release()

# ══════════════════════════════════════════════════════════════════════════════
# Stuck detection
# ══════════════════════════════════════════════════════════════════════════════

class StuckDetector:
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.still  = 0
        self.last   = 1.0

    def record(self, screen: Image.Image):
        grey = np.array(screen)[200:800, 300:1620].mean(axis=2).astype(np.float32)
        self.frames.append(grey)

    def is_stuck(self) -> bool:
        if len(self.frames) < 2:
            return False
        self.last  = float(ssim(self.frames[-2], self.frames[-1], data_range=255))
        self.still = self.still + 1 if self.last > SSIM_THRESHOLD else 0
        return self.still >= STUCK_TICKS

    def recover(self):
        self.still = 0
        self.frames.clear()
        steer = random.choice([-1.0, 1.0])
        print(f"  ⚠  stuck (ssim={self.last:.3f}) — rotate + nudge")
        move(rs_x=steer, ls_y=0.0); time.sleep(0.5)
        move(rs_x=0.0,   ls_y=1.0); time.sleep(0.5)
        release()

    def reset(self):
        self.still = 0
        self.frames.clear()

# ══════════════════════════════════════════════════════════════════════════════
# Startup checks
# ══════════════════════════════════════════════════════════════════════════════

def check_services() -> bool:
    ok = True
    for url, name in [(f"{SCREEN_HTTP}/health",  "screen      b550:5003"),
                      (f"{CONTROL_HTTP}/health", "controller  b550:5002"),
                      (OCR_URL.replace('/ocr', '/health'), "ocr   beacon:5001")]:
        try:
            http.get(url, timeout=3); print(f"  ✓  {name}")
        except Exception as e:
            print(f"  ✗  {name}: {e}"); ok = False
    try:
        ids = [m.id for m in llm.models.list().data]
        print(f"  ✓  lm studio beacon:1234  {ids}")
    except Exception as e:
        print(f"  ✗  lm studio: {e}"); ok = False
    # WS handshakes
    for fn, name in [(_connect_screen, "screen ws"), (_connect_control, "control ws")]:
        try:
            fn(); print(f"  ✓  {name} connected")
        except Exception as e:
            print(f"  ✗  {name}: {e}"); ok = False
    return ok

# ══════════════════════════════════════════════════════════════════════════════
# Main loop — continuous, set-and-hold
# ══════════════════════════════════════════════════════════════════════════════

def run():
    sd = StuckDetector()

    print("UMA v5.0 — checking services...")
    if not check_services():
        print("\nOne or more services unreachable.")
        sys.exit(1)
    print(f"\nAll services OK.  Starting in 5s — W3 must be running on {GAME_PC}...\n")
    for i in range(5, 0, -1):
        print(f"  {i}", end="\r", flush=True); time.sleep(1)
    print("UMA v5.0 running.  Ctrl-C to stop.\n")

    tick          = 0
    last_mode     = ""
    post_load     = False
    rs_smooth     = 0.0
    recent_pursuit = deque(maxlen=5)
    combat_streak = 0

    while True:
        tick += 1
        t0 = time.time()

        screen = get_screen()
        if screen is None:
            print(f"[{tick}]  ⏳ no frame from screen server — check "
                  f"{SCREEN_HTTP}/health (frame_age_ms; -1 = capture thread dead)")
            time.sleep(0.5)
            continue

        raw_mode = detect_mode(screen)

        # COMBAT debounce — a phantom single-frame HP bar (scenery red sweeping
        # the strip) must not trigger the attack sequence.
        combat_streak = combat_streak + 1 if raw_mode == "COMBAT" else 0
        if raw_mode == "COMBAT" and combat_streak < COMBAT_CONFIRM:
            mode = "EXPLORATION"
            print(f"\n[{tick}]  EXPLORATION  (combat? {combat_streak}/{COMBAT_CONFIRM} — ignoring)")
        else:
            mode = raw_mode
            print(f"\n[{tick}]  {mode}")

        # Stop driving the moment we leave exploration
        if mode != "EXPLORATION" and last_mode == "EXPLORATION":
            release(); rs_smooth = 0.0

        # ── Loading ────────────────────────────────────────────────────────
        if mode == "LOADING":
            release()
            print("  ⏳ loading...")
            time.sleep(4)
            for _ in range(15):
                s = get_screen()
                if s is not None and detect_mode(s) != "LOADING":
                    break
                time.sleep(2)
            post_load, last_mode = True, "LOADING"
            continue

        if post_load and last_mode == "LOADING":
            print("  ✅ loaded — pausing 2s")
            time.sleep(2); sd.reset(); post_load = False

        # ── Cutscene ───────────────────────────────────────────────────────
        if mode == "CUTSCENE":
            print("  🎬 waiting"); last_mode = mode; time.sleep(2); continue

        # ── Choices ────────────────────────────────────────────────────────
        if mode == "CHOICES":
            quest   = ocr(crop(screen, QUEST_REGION))
            ch_img  = crop(screen, CHOICE_REGION)
            ch_txt  = ocr(ch_img)
            resp = ask(f'Quest: "{quest}"\nChoices: "{ch_txt}"\nWhat do you do?',
                       CHOICES_SYSTEM, image=ch_img, temperature=0.3, max_tokens=64)
            print(f"  🧠 {resp}")
            navigate_choice(parse_direction(resp))
            last_mode = mode
            time.sleep(max(0, TICK - (time.time() - t0)))
            continue

        # ── Combat ─────────────────────────────────────────────────────────
        if mode == "COMBAT":
            button("x"); time.sleep(0.25)
            button("x"); time.sleep(0.25)
            button("b")
            print("  ⚔  attack attack dodge")
            last_mode = mode
            time.sleep(max(0, TICK - (time.time() - t0)))
            continue

        # ── Exploration ────────────────────────────────────────────────────
        # Reorient camera when returning from another mode
        if last_mode not in ("EXPLORATION", ""):
            steer, pursuit, _, _ = read_minimap(screen)
            if pursuit is not None and abs(pursuit) > 30:
                print(f"  📷 reorient ({pursuit:+.1f}°)")
                move(rs_x=math.sin(math.radians(pursuit)), ls_y=0.0)
                time.sleep(0.3); release()

        last_mode = mode

        sd.record(screen)
        if sd.is_stuck():
            sd.recover(); rs_smooth = 0.0
            time.sleep(max(0, TICK - (time.time() - t0)))
            continue

        interact = ocr(crop(screen, INTERACT_REGION))
        if interact.strip():
            print(f"  💬 '{interact}' → A")
            release(); button("a"); time.sleep(0.4)
            continue

        steer, pursuit, n_dots, source = read_minimap(screen)

        if steer is not None:
            recent_pursuit.append(abs(pursuit))
            # Path consistently behind → sin() stalls at ±180; force a turn out.
            if len(recent_pursuit) == 5 and all(a > 140 for a in recent_pursuit):
                print("  📷 path behind — pivoting")
                move(rs_x=1.0, ls_y=0.0); time.sleep(0.6); release()
                recent_pursuit.clear(); rs_smooth = 0.0
                continue
            rs_raw, ls_y = steer_to_sticks(steer, pursuit)
            rs_smooth = RS_EMA_ALPHA * rs_raw + (1 - RS_EMA_ALPHA) * rs_smooth
            print(f"  🗺  steer {steer:+.1f}° pursuit {pursuit:+.1f}° "
                  f"[{source} n={n_dots}]  →  RS {rs_smooth:+.2f}  LS {ls_y:.2f}")
            move(rs_x=rs_smooth, ls_y=ls_y)
        else:
            print("  🗺  no path — forward")
            move(rs_x=0.0, ls_y=1.0)

        elapsed = time.time() - t0
        print(f"  ⏱  {int(elapsed * 1000)}ms")
        time.sleep(max(0, TICK - elapsed))

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        release()
        print("\n\nUMA stopped. Controller released.")
