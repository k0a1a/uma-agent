#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v4.0
===================================
A highly experimental algorithmic agent capable of navigating
and interacting with the Witcher 3 world, perceiving it
exclusively through vision — the same way a human player would.

Navigation principle (v4):
  - Minimap rotates with Geralt. Top = his forward direction always.
  - Find nearest white path dot via pixel detection.
  - Calculate angle from centre arrow to dot.
  - RS = sin(angle)  — proportional camera rotation
  - LS = cos(angle)  — proportional forward movement
  - Both sticks fire simultaneously every tick.
  - Angle converges to 0 as Geralt aligns with path.
  - No oscillation. No fixed steer values. No word translation.

Input:
  - Left stick Y only  : forward / backward
  - Right stick X only : camera rotation left / right
  - No keyboard. No left stick X. No diagonal movement.

Architecture:
  b550.local          — Witcher 3 + controller_server.py + screen_server.py
  beacon.x.k0a1a.net  — LM Studio (Qwen2.5-VL-7B) + ocr_server.py
  laptop              — this script

Usage:
    python3 uma4.py
"""

import os, sys, time, base64, io, math, random
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance
from openai import OpenAI
from skimage.metrics import structural_similarity as ssim

# ── Service endpoints ─────────────────────────────────────────────────────────

GAME_PC = os.environ.get("GAME_PC", "b550.local")
BEACON  = os.environ.get("BEACON",  "beacon.x.k0a1a.net")

CONTROLLER_URL = f"http://{GAME_PC}:5002"
SCREEN_URL     = f"http://{GAME_PC}:5003"
LM_STUDIO_URL  = f"http://{BEACON}:1234/v1"
OCR_URL        = f"http://{BEACON}:5001/ocr"
LM_MODEL       = os.environ.get("LM_MODEL", "local-model")

# ── Config ────────────────────────────────────────────────────────────────────

TICK_INTERVAL  = 1.2    # seconds per tick
MOVE_DURATION  = 0.5    # seconds to hold sticks per tick

# Minimap dot detection thresholds (tuned for B550, Next Gen W3)
DOT_BRIGHTNESS  = 160   # minimum brightness for path dots
DOT_CHANNEL_MAX = 30    # max channel difference (keeps dots neutral white)
DOT_EXCLUDE_R   = 25    # radius around centre to blank Geralt's arrow
DOT_BORDER_PAD  = 8     # pixels inside minimap edge to ignore (rim is also white)

# Steering — minimum LS forward even during sharp turns
MIN_FORWARD     = 0.3

# SSIM stuck detection
SSIM_THRESHOLD  = 0.97
STUCK_TICKS     = 3

# HUD regions (left, top, width, height) — 1920×1080, Next Gen W3, B550
MINIMAP_REGION  = (1660,  36, 213, 215)
QUEST_REGION    = (1220, 225, 340, 100)
SUBTITLE_REGION = ( 450, 655, 570,  55)
CHOICE_REGION   = ( 865, 500, 360, 125)
INTERACT_REGION = (  55, 350, 230, 120)
ENEMY_HP_REGION = ( 650,  44, 580,  24)

# ── Service check ─────────────────────────────────────────────────────────────

def check_services() -> bool:
    ok = True
    for url, name in [
        (f"{CONTROLLER_URL}/health", "controller  b550:5002"),
        (f"{SCREEN_URL}/health",     "screen      b550:5003"),
        (f"{OCR_URL.replace('/ocr','/health')}", "ocr   beacon:5001"),
    ]:
        try:
            requests.get(url, timeout=3)
            print(f"  ✓  {name}")
        except Exception as e:
            print(f"  ✗  {name}: {e}")
            ok = False
    try:
        ids = [m.id for m in OpenAI(base_url=LM_STUDIO_URL,
               api_key="lm-studio").models.list().data]
        print(f"  ✓  lm studio beacon:1234  {ids}")
    except Exception as e:
        print(f"  ✗  lm studio: {e}")
        ok = False
    return ok

print("UMA v4.0 — checking services...")
if not check_services():
    print("\nOne or more services unreachable.")
    sys.exit(1)
print("\nAll services OK.\n")
llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ── Screen capture ────────────────────────────────────────────────────────────

def get_screen() -> Image.Image:
    r = requests.get(f"{SCREEN_URL}/screenshot", timeout=5)
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def crop(screen: Image.Image, region: tuple) -> Image.Image:
    l, t, w, h = region
    return screen.crop((l, t, l+w, t+h))

# ── Minimap navigation (pixel-based, no model) ───────────────────────────────

def read_minimap(screen: Image.Image) -> Optional[float]:
    """
    Find the nearest white path dot on the minimap and return the angle
    to it from Geralt's forward direction.

    The minimap rotates with Geralt — top is always his forward.
    The white arrow at centre is his position.

    Returns:
      angle in degrees:
        0   = path dot is directly ahead
        +90 = path dot is to the right  (positive = rightward)
        -90 = path dot is to the left   (negative = leftward)
        180 = path dot is directly behind
      None if no dots found.
    """
    mm  = np.array(crop(screen, MINIMAP_REGION))
    h, w = mm.shape[:2]
    cx, cy = w // 2, h // 2

    # Isolate white path dots:
    # - High brightness in all channels
    # - Neutral (not tinted yellow, green, or brown)
    r = mm[:,:,0].astype(int)
    g = mm[:,:,1].astype(int)
    b = mm[:,:,2].astype(int)

    mask = (
        (r > DOT_BRIGHTNESS) &
        (g > DOT_BRIGHTNESS) &
        (b > DOT_BRIGHTNESS) &
        (np.abs(r - g) < DOT_CHANNEL_MAX) &
        (np.abs(g - b) < DOT_CHANNEL_MAX) &
        (np.abs(r - b) < DOT_CHANNEL_MAX)
    ).astype(np.uint8) * 255

    # Remove Geralt's arrow at centre
    cv2.circle(mask, (cx, cy), DOT_EXCLUDE_R, 0, -1)

    # Remove minimap border rim (also white/gold)
    border = np.zeros_like(mask)
    cv2.circle(border, (cx, cy), min(cx, cy) - DOT_BORDER_PAD, 255, -1)
    mask = cv2.bitwise_and(mask, border)

    # Remove single-pixel noise
    kernel = np.ones((2, 2), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(mask > 0)
    if len(xs) < 3:
        return None

    # Find nearest dot outside the exclusion zone
    distances  = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    far_enough = distances > DOT_EXCLUDE_R

    if far_enough.sum() < 3:
        return None

    idx    = distances[far_enough].argmin()
    tx, ty = int(xs[far_enough][idx]), int(ys[far_enough][idx])

    # Angle from Geralt's forward (up = 0°, clockwise positive)
    # atan2(x_offset, -y_offset) gives 0 at top, +90 at right
    angle = math.degrees(math.atan2(tx - cx, -(ty - cy)))

    # Save debug image
    debug = mm.copy()
    debug[mask > 0] = [255, 50, 50]          # mark detected dots red
    cv2.circle(debug, (tx, ty), 6, (0,255,0), 2)   # green circle on target
    cv2.circle(debug, (cx, cy), 3, (0,0,255), -1)  # blue dot at centre
    Image.fromarray(debug).save("minimap_debug.png")

    return angle

def angle_to_sticks(angle: float) -> Tuple[float, float]:
    """
    Convert minimap angle to controller stick values.

    RS x = sin(angle)          — how much to rotate camera
    LS y = max(MIN, cos(angle)) — how fast to move forward

    At 0°  (ahead):  RS=0.00, LS=1.00  full speed, no turn
    At 30° (slight right): RS=0.50, LS=0.87  gentle curve
    At 60° (right):  RS=0.87, LS=0.50  turning while moving
    At 90° (hard right): RS=1.00, LS=0.30  sharp turn, crawling
    At 180° (behind): RS=0.00, LS=0.30  this case → turn right by convention
    """
    rad  = math.radians(angle)
    rs_x = float(np.clip(math.sin(rad), -1.0, 1.0))
    ls_y = float(np.clip(math.cos(rad), MIN_FORWARD, 1.0))
    return rs_x, ls_y

# ── Mode detection ────────────────────────────────────────────────────────────

def px(img, x, y):
    return img.getpixel((x, y))[:3]

def detect_mode(screen: Image.Image) -> str:
    arr = np.array(screen)

    # LOADING
    pts = [(960,540),(480,270),(1440,270),(480,810),(1440,810)]
    if all(sum(px(screen,x,y))/3 < 20 for x,y in pts):
        return "LOADING"

    # CUTSCENE
    if (sum(px(screen,960,40))/3   < 15 and
        sum(px(screen,960,1040))/3 < 15 and
        sum(px(screen,960,540))/3  > 30):
        return "CUTSCENE"

    # CHOICES — dark box with bright text
    l,t,w,h = CHOICE_REGION
    cg = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2GRAY)
    if ((cg>160).sum()/cg.size > 0.04 and
        ((cg>15)&(cg<90)).sum()/cg.size > 0.15 and
        cg.mean() < 85):
        return "CHOICES"

    # DIALOGUE
    l,t,w,h = SUBTITLE_REGION
    sg = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2GRAY)
    if (sg>160).sum() > 200 and sg.mean() < 130:
        return "DIALOGUE"

    # COMBAT
    l,t,w,h = ENEMY_HP_REGION
    hh = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2HSV)
    rm = cv2.inRange(hh, np.array([0,140,100]), np.array([12,255,255]))
    if rm.sum()//255 > 60:
        return "COMBAT"

    return "EXPLORATION"

# ── OCR ───────────────────────────────────────────────────────────────────────

def ocr(img: Image.Image) -> str:
    img = ImageEnhance.Contrast(img.convert("L")).enhance(2.5).convert("RGB")
    try:
        r = requests.post(OCR_URL, json={"image": to_b64(img)}, timeout=5)
        return r.json().get("text", "").strip()
    except Exception as e:
        print(f"  ⚠  OCR: {e}")
        return ""

# ── LLM (dialogue and choices only) ──────────────────────────────────────────

def ask(prompt: str, system: str,
        image: Optional[Image.Image] = None,
        temperature: float = 0.3,
        max_tokens: int = 64) -> str:
    content = []
    if image is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{to_b64(image)}"}})
    content.append({"type": "text", "text": prompt})
    try:
        r = llm.chat.completions.create(
            model       = LM_MODEL,
            messages    = [{"role":"system","content":system},
                           {"role":"user",  "content":content}],
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠  LLM: {e}")
        return ""

# ── Controller ────────────────────────────────────────────────────────────────

def _post(path: str, data: dict):
    try:
        requests.post(f"{CONTROLLER_URL}{path}", json=data, timeout=5)
    except Exception as e:
        print(f"  ⚠  controller {path}: {e}")

def move(rs_x: float, ls_y: float, duration: float = MOVE_DURATION):
    """
    Set both sticks simultaneously and hold for duration.
      rs_x : right stick X  — camera rotation (-1=left, +1=right)
      ls_y : left stick Y   — movement (-1=back, +1=forward)
    Left stick X is always 0 — no diagonal, no strafe.
    """
    _post("/navigate", {
        "left_x":   0.0,    # never use left stick X
        "left_y":   ls_y,
        "right_x":  rs_x,
        "right_y":  0.0,
        "duration": duration,
    })

def button(name: str, duration: float = 0.15):
    _post("/button", {"button": name, "duration": duration})

def release():
    _post("/release", {})

# ── Stuck detection ───────────────────────────────────────────────────────────

class StuckDetector:
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.still  = 0
        self.last   = 1.0

    def record(self, screen: Image.Image):
        arr  = np.array(screen)
        grey = arr[200:800, 300:1620].mean(axis=2).astype(np.float32)
        self.frames.append(grey)

    def is_stuck(self) -> bool:
        if len(self.frames) < 2:
            return False
        self.last = float(ssim(self.frames[-2], self.frames[-1],
                               data_range=255))
        self.still = self.still + 1 if self.last > SSIM_THRESHOLD else 0
        return self.still >= STUCK_TICKS

    def recover(self):
        self.still = 0
        self.frames.clear()
        # Back up while rotating camera to clear obstacle
        steer = random.choice([-0.7, 0.7])
        print(f"  ⚠  stuck (ssim={self.last:.3f}) — back + steer {steer:+.1f}")
        move(rs_x=steer, ls_y=-1.0, duration=0.8)
        time.sleep(0.1)

    def reset(self):
        self.still = 0
        self.frames.clear()

# ── Dialogue / choices ────────────────────────────────────────────────────────

CHOICES_SYSTEM = (
    "You are Geralt of Rivia in Witcher 3. Follow the main story quest.\n"
    "Dialogue choices are visible. Choose what best advances the quest.\n"
    "Reply ONLY:\n"
    "DIRECTION: <up|down|confirm>\n"
    "REASON: one sentence\n\n"
    "up      = move to previous choice (left stick up)\n"
    "down    = move to next choice (left stick down)\n"
    "confirm = select current choice (A button)"
)

def parse_direction(response: str) -> str:
    for line in response.strip().splitlines():
        if "DIRECTION:" in line.upper():
            return line.split(":",1)[1].strip().lower()
    return "confirm"

def navigate_choice(direction: str):
    if direction == "up":
        move(rs_x=0.0, ls_y=1.0, duration=0.25)
    elif direction == "down":
        move(rs_x=0.0, ls_y=-1.0, duration=0.25)
    elif direction == "confirm":
        button("a")
    time.sleep(0.2)

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    sd = StuckDetector()

    print("""
                             \u2592\u2592\u2592\u2592\u2592\u2592\u2592\u2592\u2592
                     \u2592\u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593\u2592
              \u2592\u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593\u2592
           \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593\u2592
         \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593
        \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592      \u2592\u2592\u2592\u2592
       \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2591\u2592\u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2593
      \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
    \u2592\u2592  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592
  \u2592\u2588\u2588\u2588  \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
   \u2593\u2588\u2588\u2588\u2593 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
        \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593\u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
          \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592      \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592
            \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592  \u2592\u2593\u2593\u2592   \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
             \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2593 \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
             \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588 \u2592\u2588\u2588\u2588\u2588\u2588\u2593 \u2588\u2588\u2588\u2588\u2588\u2588\u2592  \u2592\u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
                \u2593\u2588\u2588\u2588\u2588\u2588\u2592 \u2593\u2588\u2588\u2588\u2588\u2592 \u2588\u2588\u2588\u2588\u2592       \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592
                 \u2588\u2588\u2588\u2588\u2588\u2588   \u2592\u2592  \u2593\u2588\u2588\u2588   \u2592\u2593\u2593\u2593\u2592  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
                  \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2593\u2593\u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
                   \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588 \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593
                    \u2592\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592 \u2592\u2588\u2588\u2588\u2588\u2588\u2592 \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2593
                      \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592  \u2592\u2593\u2592  \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2592
                        \u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588
                          \u2592\u2593\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2593\u2592
                                 \u2592\u2592\u2592\u2592
""")
    print(f"Starting in 5s — W3 must be running on {GAME_PC}...\n")
    for i in range(5, 0, -1):
        print(f"  {i}", end="\r", flush=True)
        time.sleep(1)
    print("UMA v4.0 running.  Ctrl-C to stop.\n")

    tick      = 0
    last_mode = ""
    post_load = False

    while True:
        tick += 1
        t0 = time.time()

        try:
            screen = get_screen()
        except Exception as e:
            print(f"\n[{tick}]  ⚠  screen: {e}")
            time.sleep(2)
            continue

        mode = detect_mode(screen)
        print(f"\n[{tick}]  {mode}")

        # ── Loading ────────────────────────────────────────────────────────
        if mode == "LOADING":
            print("  ⏳ loading...")
            time.sleep(4)
            for _ in range(15):
                try:
                    if detect_mode(get_screen()) != "LOADING":
                        break
                except:
                    pass
                time.sleep(2)
            post_load = True
            last_mode = "LOADING"
            continue

        if post_load and last_mode == "LOADING":
            print("  ✅ loaded — pausing 3s")
            time.sleep(3)
            sd.reset()
            post_load = False

        last_mode = mode

        # ── Cutscene ───────────────────────────────────────────────────────
        if mode == "CUTSCENE":
            print("  🎬 waiting")
            time.sleep(2)
            continue

        # ── Dialogue ───────────────────────────────────────────────────────
        if mode == "DIALOGUE":
            sub = ocr(crop(screen, SUBTITLE_REGION))
            print(f"  💬 '{sub}'")
            button("a")
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Choices ────────────────────────────────────────────────────────
        if mode == "CHOICES":
            quest       = ocr(crop(screen, QUEST_REGION))
            choices_img = crop(screen, CHOICE_REGION)
            choices_txt = ocr(choices_img)
            resp = ask(
                f'Quest: "{quest}"\nChoices: "{choices_txt}"\nWhat do you do?',
                CHOICES_SYSTEM,
                image=choices_img,
                temperature=0.3,
                max_tokens=64
            )
            print(f"  🧠 {resp}")
            navigate_choice(parse_direction(resp))
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Combat ─────────────────────────────────────────────────────────
        if mode == "COMBAT":
            button("x"); time.sleep(0.3)
            button("x"); time.sleep(0.3)
            button("b")
            print("  ⚔  attack attack dodge")
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Exploration ────────────────────────────────────────────────────
        if mode == "EXPLORATION":

            # Stuck check
            sd.record(screen)
            if sd.is_stuck():
                sd.recover()
                time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
                continue

            # Interact prompt
            interact = ocr(crop(screen, INTERACT_REGION))
            if interact.strip():
                print(f"  💬 '{interact}' → A")
                button("a")
                time.sleep(0.5)
                continue

            # Read minimap angle
            angle = read_minimap(screen)

            if angle is not None:
                rs_x, ls_y = angle_to_sticks(angle)
                print(f"  🗺  {angle:+.1f}°  →  RS {rs_x:+.2f}  LS {ls_y:.2f}")
            else:
                rs_x, ls_y = 0.0, 1.0
                print(f"  🗺  no path — forward")

            move(rs_x=rs_x, ls_y=ls_y, duration=MOVE_DURATION)

            elapsed = time.time() - t0
            print(f"  ⏱  {int(elapsed*1000)}ms")
            time.sleep(max(0, TICK_INTERVAL - elapsed))

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        release()
        print("\n\nUMA stopped. Controller released.")
