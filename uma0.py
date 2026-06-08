#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v1.1
===================================
Witcher 3 perceptual agent.
- Mouse rotation (proportional, smooth)
- Path following (white dots) with marker fallback
- Stuck/oscillation detection + recovery

Platform:  Linux, X11, Openbox, Nvidia, single 1080p display
Inference: Qwen2-VL via LM Studio (remote machine)

Usage:
    export W3_WINDOW_ID=$(xdotool search --name "Witcher" | head -1)
    export LM_STUDIO_URL="http://<beacon-ip>:1234/v1"
    export LM_STUDIO_MODEL="qwen/qwen3-vl-4b"
    python3 uma0.py
"""

import os, sys, time, base64, subprocess, random
import numpy as np
from io import BytesIO
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import requests
import mss, cv2
from PIL import Image, ImageEnhance
from openai import OpenAI
from skimage.metrics import structural_similarity as ssim

# ── Config ────────────────────────────────────────────────────────────────────

LM_STUDIO_URL   = os.environ.get("LM_STUDIO_URL",   "http://beacon:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "local-model")
BEACON_OCR_URL  = os.environ.get("BEACON_OCR_URL",  "http://192.168.12.232:5001")
W3_WINDOW_ID    = os.environ.get("W3_WINDOW_ID",    "")

GAME_W, GAME_H = 1920, 1080
TICK_INTERVAL  = 2.5
STARTUP_DELAY  = 6
MAX_HISTORY    = 20

NAV_TEMPERATURE      = 0.2
DIALOGUE_TEMPERATURE = 0.5
MAX_TOKENS_NAV       = 256
MAX_TOKENS_DIALOGUE  = 512

# Mouse sensitivity — pixels per degree of angle offset
# Increase if turning is too slow, decrease if too jerky
MOUSE_SENSITIVITY = 0.8
MAX_ROTATE_PX     = 60    # cap per tick — no spinning

# Forward arc for path dot selection — tune based on behaviour
# Narrow (45°) = strict forward preference, may miss sharp corners
# Wide   (75°) = more flexible, may cut corners
# Start at 60°
FORWARD_ARC_DEG = 60

# ── HUD regions (1920x1080) ───────────────────────────────────────────────────

MINIMAP_REGION  = (1666,  65, 207, 200)
QUEST_REGION    = (1195, 218, 265, 130)
SUBTITLE_REGION = ( 420, 648, 610,  78)
CHOICE_REGION   = ( 855, 496, 250, 130)
INTERACT_REGION = (  55, 350, 230, 120)
ENEMY_HP_REGION = ( 650,  44, 580,  24)

# ── Init ──────────────────────────────────────────────────────────────────────

print("Connecting to LM Studio...")
llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
try:
    models = llm.models.list()
    print(f"Connected. Models: {[m.id for m in models.data]}")
except Exception as e:
    print(f"LM Studio error: {e}")
    sys.exit(1)

print(f"OCR server: {BEACON_OCR_URL}\n")

sct = mss.MSS()

# ── Screen capture ────────────────────────────────────────────────────────────

_screen: Optional[Image.Image] = None

def grab_screen() -> Image.Image:
    global _screen
    raw     = sct.grab(sct.monitors[1])
    _screen = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return _screen

def crop(region: tuple) -> Image.Image:
    if _screen is None: grab_screen()
    l, t, w, h = region
    return _screen.crop((l, t, l+w, t+h))

def to_np(img): return np.array(img)
def to_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def px(x, y):
    if _screen is None: grab_screen()
    return _screen.getpixel((x, y))[:3]

# ── Mode detection ────────────────────────────────────────────────────────────

@dataclass
class ScreenMode:
    mode:       str
    confidence: str  = "high"
    details:    dict = field(default_factory=dict)

def detect_mode() -> ScreenMode:
    sample_points = [
        (960, 540),
        (480, 270), (1440, 270),
        (480, 810), (1440, 810),
    ]
    brightnesses = [sum(px(x, y))/3 for x, y in sample_points]
    if all(b < 20 for b in brightnesses):
        return ScreenMode("LOADING",
                          details={"brightnesses": [round(b,1) for b in brightnesses]})

    top_b = sum(px(960,  40))/3
    bot_b = sum(px(960,1040))/3
    ctr_b = sum(px(960, 540))/3
    if top_b < 15 and bot_b < 15 and ctr_b > 30:
        return ScreenMode("CUTSCENE",
                          details={"top": round(top_b,1), "bot": round(bot_b,1),
                                   "ctr": round(ctr_b,1)})

    pts  = [(480,300),(480,540),(480,780),(960,300),(960,540),(960,780)]
    brts = [sum(px(x,y))/3 for x,y in pts]
    var, mean = float(np.var(brts)), float(np.mean(brts))
    mm_c = sum(px(1769, 165))/3
    if var < 150 and 65 < mean < 185 and mm_c < 55:
        return ScreenMode("MENU", details={"var":round(var,1),"mean":round(mean,1)})

    cg = cv2.cvtColor(to_np(crop(CHOICE_REGION)), cv2.COLOR_RGB2GRAY)
    if (cg>160).sum()/cg.size > 0.012 and ((cg>15)&(cg<90)).sum()/cg.size > 0.08:
        return ScreenMode("CHOICES")

    sg = cv2.cvtColor(to_np(crop(SUBTITLE_REGION)), cv2.COLOR_RGB2GRAY)
    if int((sg>160).sum()) > 200 and float(sg.mean()) < 130:
        return ScreenMode("DIALOGUE")

    hh = cv2.cvtColor(to_np(crop(ENEMY_HP_REGION)), cv2.COLOR_RGB2HSV)
    rm = cv2.inRange(hh, np.array([0,140,100]), np.array([12,255,255]))
    if rm.sum()//255 > 60:
        return ScreenMode("COMBAT")

    return ScreenMode("EXPLORATION")

# ── Navigation ────────────────────────────────────────────────────────────────

def _angle_to_dir(angle):
    if   angle < 25 or angle > 335: return "forward",      "none"
    elif angle < 70:                 return "forward-right", "right"
    elif angle < 120:                return "right",         "right"
    elif angle < 160:                return "back-right",    "right"
    elif angle < 200:                return "behind",        "right"
    elif angle < 250:                return "back-left",     "left"
    elif angle < 290:                return "left",          "left"
    else:                            return "forward-left",  "left"

_path_call_count = 0

def read_path_direction() -> dict:
    global _path_call_count
    _path_call_count += 1

    mm = to_np(crop(MINIMAP_REGION))
    h, w = mm.shape[:2]
    cx, cy = w//2, h//2

    r = mm[:,:,0].astype(int)
    g = mm[:,:,1].astype(int)
    b = mm[:,:,2].astype(int)

    white_mask = (
        (r > 160) & (g > 160) & (b > 160) &
        (np.abs(r-g) < 30) & (np.abs(g-b) < 30) & (np.abs(r-b) < 30)
    ).astype(np.uint8) * 255

    # Exclusion zone — Geralt's arrow
    cv2.circle(white_mask, (cx,cy), 25, 0, -1)

    border = np.zeros_like(white_mask)
    cv2.circle(border, (cx,cy), min(cx,cy)-8, 255, -1)
    white_mask = cv2.bitwise_and(white_mask, border)
    white_mask = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8)
    )

    ys, xs = np.where(white_mask > 0)
    if len(xs) < 3:
        return {"found": False, "source": "path", "path_px": 0}

    distances = np.sqrt((xs-cx)**2 + (ys-cy)**2)
    angles_from_fwd = np.abs(
        np.degrees(np.arctan2(xs-cx, -(ys-cy))) % 360
    )
    angles_from_fwd = np.minimum(angles_from_fwd, 360 - angles_from_fwd)

    forward_arc = angles_from_fwd < FORWARD_ARC_DEG
    if forward_arc.sum() > 3:
        fwd_distances = distances.copy()
        fwd_distances[~forward_arc] = np.inf
        idx = fwd_distances.argmin()
    else:
        wider_arc = angles_from_fwd < (FORWARD_ARC_DEG + 15)
        if wider_arc.sum() > 0:
            wd = distances.copy()
            wd[~wider_arc] = np.inf
            idx = wd.argmin()
        else:
            idx = distances.argmin()

    tx, ty = int(xs[idx]), int(ys[idx])

    angle           = float(np.degrees(np.arctan2(tx-cx, -(ty-cy))) % 360)
    direction, turn = _angle_to_dir(angle)

    if _path_call_count % 10 == 0:
        debug = mm.copy()
        debug[white_mask > 0] = [255, 0, 0]
        cv2.circle(debug, (tx, ty), 6, (0,255,0), 2)
        cv2.circle(debug, (cx, cy), 35, (0,0,255), 1)
        Image.fromarray(debug).save("nav_debug.png")

    return {
        "found":     True,
        "source":    "path",
        "angle":     round(angle, 1),
        "direction": direction,
        "turn":      turn,
        "path_px":   int(len(xs)),
        "target":    (tx, ty),
    }

def read_marker_direction() -> dict:
    """Fallback: yellow quest marker direction."""
    mm  = to_np(crop(MINIMAP_REGION))
    hsv = cv2.cvtColor(mm, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([15,100,120]), np.array([45,255,255]))
    h, w = mask.shape
    cx, cy = w//2, h//2
    cv2.circle(mask, (cx,cy), 20, 0, -1)
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx,cy), min(cx,cy)-10, 255, -1)
    mask = cv2.bitwise_and(mask, inner)
    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return {"found": False, "source": "marker"}
    mx, my = float(xs.mean()), float(ys.mean())
    angle           = float(np.degrees(np.arctan2(mx-cx, -(my-cy))) % 360)
    direction, turn = _angle_to_dir(angle)
    return {"found": True, "source": "marker", "angle": round(angle,1),
            "direction": direction, "turn": turn, "yellow_px": int(len(xs))}

def read_minimap() -> dict:
    path = read_path_direction()
    if path["found"] and path["path_px"] > 2:
        return path
    return read_marker_direction()

# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr(region):
    img  = crop(region)
    img  = ImageEnhance.Contrast(img.convert("L")).enhance(2.5).convert("RGB")
    b64  = to_b64(img)
    resp = requests.post(f"{BEACON_OCR_URL}/ocr", json={"image": b64})
    return resp.json().get("text", "")

def read_subtitle()  -> str: return _ocr(SUBTITLE_REGION)
def read_choices()   -> str: return _ocr(SUBTITLE_REGION) + " | " + _ocr(CHOICE_REGION)
def read_quest()     -> str: return _ocr(QUEST_REGION)
def read_interact()  -> str: return _ocr(INTERACT_REGION)

# ── Input ─────────────────────────────────────────────────────────────────────

KEYMAP = {
    "confirm":  "e",   "continue": "space", "skip":    "space",
    "up":       "Up",  "down":     "Down",  "forward": "w",
    "back":     "s",   "left":     "a",     "right":   "d",
    "sprint":   "shift",
}

def _xdo(*args):
    subprocess.run(["xdotool", *args], check=False)

def _activate():
    if W3_WINDOW_ID:
        _xdo("windowactivate", "--sync", W3_WINDOW_ID)

def press(key: str):
    k = KEYMAP.get(key.lower(), key)
    _activate()
    if W3_WINDOW_ID:
        _xdo("key", "--window", W3_WINDOW_ID, "--clearmodifiers", k)
    else:
        _xdo("key", k)
    time.sleep(0.3)

def hold(key: str, duration: float, sprint: bool = False):
    k        = KEYMAP.get(key.lower(), key)
    duration = float(np.clip(duration, 0.1, 5.0))
    _activate()
    if sprint:
        _xdo("keydown", "--window", W3_WINDOW_ID, "--clearmodifiers", "shift")
    if W3_WINDOW_ID:
        _xdo("keydown", "--window", W3_WINDOW_ID, "--clearmodifiers", k)
        time.sleep(duration)
        _xdo("keyup",   "--window", W3_WINDOW_ID, k)
    else:
        _xdo("keydown", k); time.sleep(duration); _xdo("keyup", k)
    if sprint:
        _xdo("keyup", "--window", W3_WINDOW_ID, "shift")
    time.sleep(0.1)

def rotate_toward(nav: dict):
    if not nav.get("found"):
        return
    angle  = nav["angle"]
    offset = angle if angle <= 180 else angle - 360

    if abs(offset) < 8:
        return   # already aligned

    pixels = int(offset * MOUSE_SENSITIVITY)
    pixels = int(np.clip(pixels, -MAX_ROTATE_PX, MAX_ROTATE_PX))

    print(f"  🖱  rotate {pixels:+d}px  (offset {offset:+.1f}°)")
    _activate()
    subprocess.run(
        ["xdotool", "mousemove_relative", "--", str(pixels), "0"],
        check=False
    )
    time.sleep(0.2)  # give game time to update camera

# ── Navigation state machine ──────────────────────────────────────────────────

nav_state = {"phase": "read", "target_angle": None, "forward_count": 0}

def navigation_step():
    nav = read_minimap()

    if not nav.get("found"):
        hold("forward", 0.5)
        return

    angle  = nav["angle"]
    offset = angle if angle <= 180 else angle - 360

    if abs(offset) < 15:
        # Already aligned — just walk forward
        hold("forward", 1.5)
        nav_state["forward_count"] += 1
    else:
        # Rotate ONCE proportionally, then walk forward immediately
        # Don't re-read minimap until after the forward move
        pixels = int(np.clip(offset * 1.5, -100, 100))
        print(f"  🖱  rotate {pixels:+d}px")
        _activate()
        subprocess.run(["xdotool", "mousemove_relative", "--", str(pixels), "0"])
        time.sleep(0.3)   # let camera settle AND minimap redraw
        hold("forward", 1.0)   # always move after rotating

def mouse_rotate(pixels: int):
    """Raw pixel rotation — positive=right, negative=left."""
    _activate()
    subprocess.run(["xdotool", "mousemove_relative", "--", str(pixels), "0"], check=False)
    time.sleep(0.2)

def rotate_and_move(nav: dict):
    if nav.get("found"):
        angle  = nav["angle"]
        offset = angle if angle <= 180 else angle - 360
        if abs(offset) > 10:
            pixels = int(np.clip(offset * MOUSE_SENSITIVITY, -MAX_ROTATE_PX, MAX_ROTATE_PX))
            print(f"  🖱  rotate {pixels:+d}px  (offset {offset:+.1f}°)")
            _activate()
            subprocess.run(["xdotool", "mousemove_relative", "--", str(pixels), "0"], check=False)
            time.sleep(0.3)
    else:
        print("  🗺  no nav signal — walking forward blind")

    # always walk forward regardless
    hold("forward", 1.0)
    print("  🚶 forward 1.0s")

# ── Stuck detection ───────────────────────────────────────────────────────────

class StuckDetector:
    """
    Detects oscillation (left-right-left-right) and no-forward-progress.
    Issues recovery actions without involving the LLM.
    """
    RECOVERY_SEQUENCE = [
        {"action": "move", "key": "forward", "duration": 1.2, "sprint": False,
         "reason": "recovery: push forward"},
        {"action": "move", "key": "back",    "duration": 0.7, "sprint": False,
         "reason": "recovery: back up"},
        {"action": "move", "key": "forward", "duration": 1.5, "sprint": True,
         "reason": "recovery: sprint through"},
        {"action": "move", "key": "back",    "duration": 0.5, "sprint": False,
         "reason": "recovery: back up again"},
    ]

    def __init__(self, window: int = 8):
        self.actions        = deque(maxlen=window)
        self.recovery_idx   = 0
        self.frame_buffer   = deque(maxlen=4)
        self.still_count    = 0
        self.ssim_threshold = 0.985
        self.last_angles    = deque(maxlen=4)

    def record(self, action: str, key: str):
        self.actions.append((action, key))

    def record_frame(self, screen: Image.Image):
        arr = np.array(screen.convert("L").resize((64, 36)))
        self.frame_buffer.append(arr)

    def update(self) -> bool:
        if len(self.frame_buffer) < 2:
            return False
        f1    = self.frame_buffer[-2]
        f2    = self.frame_buffer[-1]
        score = ssim(f1, f2, data_range=255)
        print(f"  📊 ssim={score:.4f}  still={self.still_count}")
        if score > self.ssim_threshold:
            self.still_count += 1
        else:
            self.still_count = 0
        return self.scene_static()

    def scene_static(self) -> bool:
        return self.still_count >= 3

    def record_nav(self, angle: float):
        self.last_angles.append(round(angle, 1))

    def angle_locked(self) -> bool:
        if len(self.last_angles) < 4:
            return False
        return len(set(self.last_angles)) == 1

    def is_stuck(self) -> bool:
        return self.scene_static() or self.angle_locked()

    def next_recovery(self) -> dict:
        if self.angle_locked():
            mouse_rotate(random.choice([-90, 90]))
            time.sleep(0.3)
        action = self.RECOVERY_SEQUENCE[self.recovery_idx % len(self.RECOVERY_SEQUENCE)]
        self.recovery_idx += 1
        self.actions.clear()
        self.last_angles.clear()
        return action

    def reset(self):
        self.recovery_idx = 0
        self.actions.clear()
        self.last_angles.clear()
        self.still_count = 0

stuck = StuckDetector()

# ── Execute action ────────────────────────────────────────────────────────────

def execute_action(parsed: dict) -> str:
    action   = parsed.get("action", "wait")
    key      = parsed.get("key", "space")
    duration = float(np.clip(parsed.get("duration", 0.5), 0.1, 5.0))
    sprint   = parsed.get("sprint", False)
    reason   = parsed.get("reason", "")

    stuck.record(action, key)

    if action == "press":
        press(key)
        print(f"  ⌨  {key}  —  {reason}")
        return f"Pressed {key}."

    elif action == "move":
        direction = key
        print(f"  🚶 {direction} {duration}s"
              f"{'  sprint' if sprint else ''}  —  {reason}")
        hold(direction, duration, sprint=sprint)
        return f"Moved {direction} for {duration}s."

    elif action == "rotate":
        # model explicitly requests a rotation
        nav = read_minimap()
        rotate_toward(nav)
        hold("forward", duration)
        return "Rotated and moved forward."

    elif action == "wait":
        secs = float(np.clip(duration, 0.5, 12.0))
        print(f"  ⏳ {secs}s  —  {reason}")
        time.sleep(secs)
        return f"Waited {secs}s."

    else:
        time.sleep(1.0)
        return "Unknown action."

# ── LLM ───────────────────────────────────────────────────────────────────────

def ask_llm(prompt, system, image=None, temperature=0.3, max_tokens=256) -> str:
    content = []
    if image is not None:
        b64 = to_b64(image)
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    try:
        resp = llm.chat.completions.create(
            model       = LM_STUDIO_MODEL,
            messages    = [{"role": "system", "content": system},
                           {"role": "user",   "content": content}],
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠  LLM error: {e}")
        return "ACTION: wait\nKEY: space\nDURATION: 2\nREASON: LLM error"

def parse_action(response: str) -> dict:
    lines = {}
    for line in response.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            lines[k.strip().upper()] = v.strip()
    action   = lines.get("ACTION",   "wait").lower()
    key      = lines.get("KEY",      "space").lower()
    sprint   = "sprint" in lines.get("KEY", "").lower() or \
               "sprint" in lines.get("REASON","").lower()
    try:
        duration = float(lines.get("DURATION", "0.5").split()[0])
    except:
        duration = 0.5
    return {"action": action, "key": key, "duration": duration,
            "sprint": sprint, "reason": lines.get("REASON", ""), "raw": response}

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_BASE = """You are UMA, an AI agent playing The Witcher 3 as Geralt.
Respond in EXACTLY this format:
ACTION: <move|wait>
KEY: <forward|back>
DURATION: <seconds>
REASON: <one sentence>
In exploration mode only forward and back are valid directions.
Camera steering is handled automatically."""

MODE_CONTEXT = {
    "LOADING":  "Loading screen. Wait.",
    "CUTSCENE": "Cutscene playing. Wait.",
    "MENU":     "Menu open. Navigate with up/down, confirm with e.",
    "DIALOGUE": "NPC speaking. Press space to advance. Wait if still animating.",
    "CHOICES":  ("Choices visible. Read them carefully. "
                 "Choose as Geralt would — pragmatic, lore-consistent. "
                 "Navigate with up/down, confirm with e."),
    "COMBAT":   "Combat active. Press e to attack, space to dodge. Pattern: attack attack dodge.",
    "EXPLORATION": (
        "Geralt is free to move. Follow the quest.\n"
        "Camera rotation is handled automatically before each tick.\n"
        "You only need to decide: move forward, move back, or wait.\n"
        "NEVER use left or right — steering is done by camera rotation.\n"
        "If the path is ahead: ACTION: move / KEY: forward / DURATION: 1.5\n"
        "If blocked or unsure: ACTION: move / KEY: back / DURATION: 0.5\n"
        "Then wait for next tick to re-evaluate."
    ),
}

def system_for(mode):
    return f"{SYSTEM_BASE}\n\nState: {mode}\n{MODE_CONTEXT.get(mode, '')}"

# ── Context ───────────────────────────────────────────────────────────────────

def build_context(mode: ScreenMode, same_count: int) -> tuple:
    lines = [f"Mode: {mode.mode}", f"Unchanged: {same_count} tick(s)"]
    image = _screen

    if mode.mode == "DIALOGUE":
        lines.append(f'Subtitle: "{read_subtitle()}"')
        lines.append("Press space to advance.")

    elif mode.mode == "CHOICES":
        lines.append(f'Choices: "{read_choices()}"')
        lines.append("Choose and confirm.")

    elif mode.mode == "EXPLORATION":
        nav      = read_minimap()
        quest    = read_quest()
        interact = read_interact()
        lines.append(f"Minimap nav: {nav}")
        if quest:    lines.append(f'Quest: "{quest}"')
        if interact: lines.append(f'Interact: "{interact}" — press e')
        lines.append("Follow the path direction. Move forward after aligning.")
        image = crop(MINIMAP_REGION)

    elif mode.mode in ("LOADING", "CUTSCENE"):
        image = None

    return "\n".join(lines), image

# ── Main loop ─────────────────────────────────────────────────────────────────

def find_w3():
    result = subprocess.run(
        ["xdotool", "search", "--name", "Witcher"],
        capture_output=True, text=True
    )
    ids = result.stdout.strip().split("\n")
    return ids[0].strip() if ids and ids[0].strip() else ""

def run():
    global W3_WINDOW_ID
    if not W3_WINDOW_ID:
        W3_WINDOW_ID = find_w3()
        if W3_WINDOW_ID:
            print(f"Found W3 window: {W3_WINDOW_ID}")
        else:
            print("W3 window not found — input goes to focused window")

    print(f"\nStarting in {STARTUP_DELAY}s — switch to Witcher 3...\n")
    for i in range(STARTUP_DELAY, 0, -1):
        print(f"  {i}", end="\r", flush=True)
        time.sleep(1)
    print("UMA running.  Ctrl-C to stop.\n")

    last_ctx   = ""
    same_count = 0
    tick       = 0
    death_count = 0
    last_mode   = "EXPLORATION"

    while True:
        tick += 1
        t0 = time.time()

        grab_screen()
        mode = detect_mode()
        ctx, image = build_context(mode, same_count)

        if ctx != last_ctx:
            last_ctx   = ctx
            same_count = 0
        else:
            same_count += 1

        print(f"\n[{tick}]  {mode.mode}  "
              f"{'NEW' if same_count == 0 else f'same×{same_count}'}")

        # ── Death / load-transition detection ─────────────────────────
        if mode.mode == "LOADING" and last_mode != "LOADING":
            death_count += 1
            print(f"  💀 Death #{death_count} — waiting for load")

        if mode.mode != "LOADING" and last_mode == "LOADING":
            print("  ✅ Loaded — pausing 3s before moving")
            time.sleep(3)
            mouse_rotate(90)
            time.sleep(0.5)
            mouse_rotate(-45)
            time.sleep(0.5)
            stuck.reset()

        last_mode = mode.mode

        # ── Exploration: fully autonomous ─────────────────────────────
        if mode.mode == "EXPLORATION":
            nav = read_minimap()
            print(f"  🗺  {nav}")
            if nav.get("found"):
                stuck.record_nav(nav["angle"])
            rotate_and_move(nav)

            grab_screen()
            stuck.record_frame(_screen)
            if stuck.update():
                recovery = stuck.next_recovery()
                print(f"  ⚠  {recovery['reason']}")
                mouse_rotate(random.choice([-40, 40]))
                time.sleep(0.2)
                execute_action(recovery)

            elapsed = time.time() - t0
            print(f"  ⏱  {int(elapsed*1000)}ms  (no LLM)")
            time.sleep(max(0, TICK_INTERVAL - elapsed))
            continue

        # ── Autonomous: loading / cutscene / menu ─────────────────────
        if mode.mode == "LOADING":
            print("  ⏳ loading — waiting 5s")
            time.sleep(5)
            for _ in range(10):
                grab_screen()
                m = detect_mode()
                if m.mode != "LOADING":
                    break
                time.sleep(2)
            continue

        if mode.mode == "CUTSCENE":
            print("  🎬 cutscene — waiting 2s")
            time.sleep(2)
            continue

        if mode.mode == "MENU":
            print("  📋 menu — pressing Escape")
            press("menu")
            time.sleep(1)
            continue

        # ── LLM: dialogue / choices / combat ──────────────────────────
        if mode.mode not in ("DIALOGUE", "CHOICES", "COMBAT"):
            time.sleep(1)
            continue

        temperature = DIALOGUE_TEMPERATURE if mode.mode in ("CHOICES", "DIALOGUE") else NAV_TEMPERATURE
        max_tokens  = MAX_TOKENS_DIALOGUE  if mode.mode in ("CHOICES", "DIALOGUE") else MAX_TOKENS_NAV

        t_llm    = time.time()
        response = ask_llm(ctx + "\n\nWhat do you do?",
                           system_for(mode.mode),
                           image=image,
                           temperature=temperature,
                           max_tokens=max_tokens)
        llm_ms   = int((time.time() - t_llm) * 1000)

        print(f"  🧠 [{llm_ms}ms] {response}")

        parsed = parse_action(response)
        execute_action(parsed)

        elapsed = time.time() - t0
        print(f"  ⏱  {int(elapsed*1000)}ms  (llm {llm_ms}ms)")
        time.sleep(max(0, TICK_INTERVAL - elapsed))

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nUMA stopped.")
