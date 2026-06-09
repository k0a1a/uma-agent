#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v1.1
===================================
A highly experimental algorithmic agent capable of navigating
and interacting with the Witcher 3 world, perceiving it
exclusively through vision — the same way a human player would.

Changes from v1.0:
  - Minimap navigation via vision model (no pixel heuristics)
  - Improved mode detection (multi-point sampling, no false LOADING)
  - Mouse rotation for smooth steering
  - SSIM-based stuck detection (scene not moving = wall)
  - Death loop prevention (post-load pause)
  - Guaranteed key release on exit

Platform:  Linux, X11, Openbox, Nvidia, single 1080p display
Inference: Qwen2-VL via LM Studio (local network)

Usage:
    export W3_WINDOW_ID=$(xdotool search --name "Witcher" | head -1)
    export LM_STUDIO_URL="http://<beacon-ip>:1234/v1"
    export LM_STUDIO_MODEL="qwen/qwen3-vl-4b"
    python3 uma1.py
"""

import os, sys, time, base64, subprocess, random, atexit, signal
import numpy as np
from io import BytesIO
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import mss, cv2, requests
from PIL import Image, ImageEnhance
from openai import OpenAI
from skimage.metrics import structural_similarity as ssim

# ── Config ────────────────────────────────────────────────────────────────────

LM_STUDIO_URL   = os.environ.get("LM_STUDIO_URL",   "http://beacon:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "qwen2.5-vl-7b-instruct")
OCR_URL         = os.environ.get("OCR_URL",         "http://beacon:5001/ocr")
W3_WINDOW_ID    = os.environ.get("W3_WINDOW_ID",    "")

GAME_W, GAME_H = 1920, 1080
TICK_INTERVAL  = 2.0
STARTUP_DELAY  = 6

# Mouse steering
MOUSE_SENSITIVITY = 1.2    # pixels per degree
MAX_ROTATE_PX     = 80     # cap per tick

# Stuck detection
SSIM_WALL_THRESHOLD  = 0.97   # high = scene not changing = wall
SSIM_FALL_THRESHOLD  = 0.25   # low = scene changed drastically = fall/death
STUCK_TICKS_REQUIRED = 3      # consecutive still frames before recovery

# ── HUD regions (1920x1080, Next Gen W3) ──────────────────────────────────────

MINIMAP_REGION  = (1666,  65, 207, 200)
QUEST_REGION    = (1195, 218, 265, 130)
SUBTITLE_REGION = ( 420, 648, 610,  78)
CHOICE_REGION   = ( 855, 496, 250, 130)
INTERACT_REGION = (  55, 350, 230, 120)
ENEMY_HP_REGION = ( 650,  44, 580,  24)

# Minimap centre pixel (for menu detection — absent in menus)
MINIMAP_CENTRE_PX = (1769, 165)

# ── Init ──────────────────────────────────────────────────────────────────────

print("Connecting to LM Studio...")
llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
try:
    models = llm.models.list()
    print(f"Connected. Models: {[m.id for m in models.data]}\n")
except Exception as e:
    print(f"LM Studio connection failed: {e}")
    print(f"Make sure LM Studio is running at {LM_STUDIO_URL}")
    sys.exit(1)

print(f"OCR service: {OCR_URL}")
try:
    r = requests.get(OCR_URL.replace("/ocr", "/health"), timeout=3)
    print("OCR service reachable.\n")
except Exception:
    print("Warning: OCR service not responding — will retry at runtime.\n")

sct = mss.MSS()

# ── Key release on exit ───────────────────────────────────────────────────────

ALL_KEYS = ["w", "a", "s", "d", "shift", "space", "e"]

def release_all_keys():
    """Send keyup multiple times to ensure release."""
    keys = ["w", "a", "s", "d", "shift", "space", "e", "Up", "Down"]
    env  = {**os.environ}
    for _ in range(3):   # send 3 times to be sure
        for k in keys:
            subprocess.run(
                ["xdotool", "keyup", "--window", W3_WINDOW_ID, k]
                if W3_WINDOW_ID else
                ["xdotool", "keyup", k],
                check=False, capture_output=True, env=env
            )
        time.sleep(0.1)
    print("\nKeys released.")

atexit.register(release_all_keys)
signal.signal(signal.SIGTERM, lambda s, f: (release_all_keys(), sys.exit(0)))

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

def to_b64(img: Image.Image) -> str:
    """Encode PIL image to base64 PNG string."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def to_b64_small(img: Image.Image, max_size: int = 640) -> str:
    """Resize large images before encoding."""
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def px(x: int, y: int) -> tuple:
    if _screen is None: grab_screen()
    return _screen.getpixel((x, y))[:3]

# ── Mode detection ────────────────────────────────────────────────────────────

@dataclass
class ScreenMode:
    mode:       str
    confidence: str  = "high"
    details:    dict = field(default_factory=dict)

def detect_mode() -> ScreenMode:
    """
    Fast pixel-based mode classification.
    Uses multiple sample points to avoid false positives from
    dark foliage, night scenes, or single dark pixels.
    """

    # LOADING — ALL 5 sample points must be near-black
    # Prevents dark forest/night scenes from falsely triggering
    load_points = [(960,540), (480,270), (1440,270), (480,810), (1440,810)]
    brightnesses = [sum(px(x,y))/3 for x,y in load_points]
    if all(b < 20 for b in brightnesses):
        return ScreenMode("LOADING",
                          details={"min_b": round(min(brightnesses),1)})

    # CUTSCENE — black letterbox bars top AND bottom, but bright centre
    top_b = sum(px(960,  40)) / 3
    bot_b = sum(px(960, 1040)) / 3
    ctr_b = sum(px(960,  540)) / 3
    if top_b < 15 and bot_b < 15 and ctr_b > 30:
        return ScreenMode("CUTSCENE",
                          details={"top": round(top_b,1), "bot": round(bot_b,1)})

    # MENU — uniform mid-tone, minimap centre absent
    pts  = [(480,300),(480,540),(480,780),(960,300),(960,540),(960,780)]
    brts = [sum(px(x,y))/3 for x,y in pts]
    var, mean = float(np.var(brts)), float(np.mean(brts))
    mm_c = sum(px(*MINIMAP_CENTRE_PX)) / 3
    if var < 150 and 65 < mean < 185 and mm_c < 55:
        return ScreenMode("MENU",
                          details={"var": round(var,1), "mean": round(mean,1)})

    # CHOICES — warm white text on dark semi-transparent box
    cg           = cv2.cvtColor(to_np(crop(CHOICE_REGION)), cv2.COLOR_RGB2GRAY)
    bright_ratio = float((cg > 160).sum()) / cg.size
    dark_ratio   = float(((cg > 15) & (cg < 90)).sum()) / cg.size
    if bright_ratio > 0.012 and dark_ratio > 0.08:
        return ScreenMode("CHOICES",
                          details={"bright": round(bright_ratio,4)})

    # DIALOGUE — subtitle bar has bright text
    sg         = cv2.cvtColor(to_np(crop(SUBTITLE_REGION)), cv2.COLOR_RGB2GRAY)
    sub_bright = int((sg > 160).sum())
    sub_mean   = float(sg.mean())
    if sub_bright > 200 and sub_mean < 130:
        return ScreenMode("DIALOGUE",
                          details={"bright_px": sub_bright,
                                   "mean":      round(sub_mean,1)})

    # COMBAT — red enemy health bars
    hh = cv2.cvtColor(to_np(crop(ENEMY_HP_REGION)), cv2.COLOR_RGB2HSV)
    rm = cv2.inRange(hh, np.array([0,140,100]), np.array([12,255,255]))
    if int(rm.sum()//255) > 60:
        return ScreenMode("COMBAT",
                          details={"red_px": int(rm.sum()//255)})

    return ScreenMode("EXPLORATION")

# ── Model inference ───────────────────────────────────────────────────────────

def ask_llm(
    prompt:      str,
    system:      str,
    image:       Optional[Image.Image] = None,
    temperature: float = 0.3,
    max_tokens:  int   = 64,
    use_small:   bool  = False,
) -> str:
    content = []
    if image is not None:
        b64 = to_b64_small(image) if use_small else to_b64(image)
        print(f"  🖼  sending image: {len(b64)} chars")
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})
    print(f"  📤  content blocks: {len(content)}")
    try:
        resp = llm.chat.completions.create(
            model       = LM_STUDIO_MODEL,
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠  LLM error: {e}")
        return ""

# ── Minimap navigation (model-based) ─────────────────────────────────────────


MINIMAP_SYSTEM = (
    "You are a navigation assistant for Witcher 3. "
    "Reply with exactly one word only. No explanation."
)

MINIMAP_PROMPT = """This is a Witcher 3 minimap screenshot.

Key facts:
- The small WHITE ARROW in the centre of the circle is Geralt's position
- The TOP of the image is Geralt's current FORWARD direction
- The BOTTOM of the image is BEHIND Geralt
- LEFT of centre = Geralt's LEFT
- RIGHT of centre = Geralt's RIGHT

Find the WHITE DOTTED PATH (small white dots forming a trail).
Determine where the NEAREST white dot is relative to the white arrow at centre.

Reply with ONE word only:
forward  = dots are above the centre arrow (ahead of Geralt)
left     = dots are to the left of the centre arrow
right    = dots are to the right of the centre arrow
behind   = dots are below the centre arrow (behind Geralt)
none     = no white dots visible
"""

DIRECTION_TO_OFFSET = {
    "forward":  0,
    "left":    -45,
    "right":    45,
    "behind":   120,
    "none":     0,
}

def read_minimap() -> dict:
    """
    Ask the vision model to read the minimap and return a direction.
    Model sees the minimap crop directly — no pixel heuristics.
    """
    minimap = crop(MINIMAP_REGION)

    t        = time.time()
    response = ask_llm(
        prompt      = MINIMAP_PROMPT,
        system      = MINIMAP_SYSTEM,
        image       = minimap,
        temperature = 0.1,
        max_tokens  = 10,
    )
    ms = int((time.time() - t) * 1000)

    # Parse — take first word, normalise
    word = response.strip().lower().split()[0] if response.strip() else "none"
    if word not in DIRECTION_TO_OFFSET:
        word = "forward"   # safe default

    return {
        "found":     word != "none",
        "direction": word,
        "offset":    DIRECTION_TO_OFFSET[word],
        "source":    "model",
        "ms":        ms,
        "raw":       response.strip(),
    }

# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr(region: tuple) -> str:
    """Send region crop to remote OCR service on beacon."""
    img = crop(region)
    img = ImageEnhance.Contrast(img.convert("L")).enhance(2.5).convert("RGB")
    b64 = to_b64(img)
    try:
        resp = requests.post(OCR_URL, json={"image": b64}, timeout=5)
        return resp.json().get("text", "").strip()
    except Exception as e:
        print(f"  ⚠  OCR error: {e}")
        return ""

def read_subtitle()  -> str: return _ocr(SUBTITLE_REGION)
def read_choices()   -> str: return _ocr(SUBTITLE_REGION) + " | " + _ocr(CHOICE_REGION)
def read_quest()     -> str: return _ocr(QUEST_REGION)
def read_interact()  -> str: return _ocr(INTERACT_REGION)

# ── Input ─────────────────────────────────────────────────────────────────────

KEYMAP = {
    "confirm":   "e",
    "continue":  "space",
    "skip":      "space",
    "up":        "w",
    "down":      "s",
    "left":      "a",
    "right":     "d",
    "forward":   "w",
    "back":      "s",
    "backward":  "s",
    "sprint":    "shift",
    "w": "w", "s": "s", "a": "a", "d": "d",
    "e": "e", "space": "space",
}

def _xdo(*args):
    subprocess.run(["xdotool", *args], check=False, capture_output=True)

def w3_is_focused() -> bool:
    if not W3_WINDOW_ID:
        return True   # no window ID set — assume focused
    result = subprocess.run(
        ["xdotool", "getactivewindow"],
        capture_output=True, text=True
    )
    return result.stdout.strip() == W3_WINDOW_ID

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
    print(f"  ⌨  holding '{k}' for {duration}s")
    _activate()
    try:
        if sprint and W3_WINDOW_ID:
            _xdo("keydown", "--window", W3_WINDOW_ID, "shift")
        if W3_WINDOW_ID:
            _xdo("keydown", "--window", W3_WINDOW_ID, "--clearmodifiers", k)
        else:
            _xdo("keydown", k)
        time.sleep(duration)
    finally:
        if W3_WINDOW_ID:
            _xdo("keyup", "--window", W3_WINDOW_ID, k)
            if sprint:
                _xdo("keyup", "--window", W3_WINDOW_ID, "shift")
        else:
            _xdo("keyup", k)
        time.sleep(0.2)   # was 0.1 — give W3 more time to process keyup

def mouse_rotate(offset_deg: float):
    if abs(offset_deg) < 5:
        return
    pixels = int(np.clip(offset_deg * MOUSE_SENSITIVITY,
                         -MAX_ROTATE_PX, MAX_ROTATE_PX))
    print(f"  🖱  rotate {pixels:+d}px  ({offset_deg:+.0f}°)")
    if W3_WINDOW_ID:
        subprocess.run(["xdotool", "windowactivate", "--sync", W3_WINDOW_ID],
                       check=False, capture_output=True)
        time.sleep(0.1)   # extra pause after activate
    subprocess.run(
        ["xdotool", "mousemove_relative", "--", str(pixels), "0"],
        check=False, capture_output=True
    )
    time.sleep(0.25)

def rotate_and_move(nav: dict):
    if nav.get("offset", 0) != 0:
        pixels = int(np.clip(nav["offset"] * MOUSE_SENSITIVITY,
                             -MAX_ROTATE_PX, MAX_ROTATE_PX))
        print(f"  🖱  rotate {pixels:+d}px  ({nav['offset']:+.0f}°)")
        if W3_WINDOW_ID:
            subprocess.run(["xdotool", "windowactivate",
                            "--sync", W3_WINDOW_ID], check=False)
        time.sleep(0.3)   # let window activate fully
        subprocess.run(["xdotool", "mousemove_relative",
                        "--", str(pixels), "0"], check=False)
        time.sleep(0.4)   # let camera settle before keypress

    # then move forward
    hold("forward", 0.4)

# ── Stuck detection ───────────────────────────────────────────────────────────

class StuckDetector:
    """
    Detects two stuck conditions:
    1. Wall: scene not changing despite forward movement (high SSIM)
    2. Fall/death: scene changed drastically (very low SSIM)
    """

    RECOVERY_SEQUENCE = [
        {"key": "back",    "duration": 0.8, "sprint": False,
         "reason": "recovery: back away"},
        {"key": "back",    "duration": 0.5, "sprint": False,
         "reason": "recovery: back more"},
        {"key": "forward", "duration": 1.5, "sprint": True,
         "reason": "recovery: sprint through"},
        {"key": "back",    "duration": 1.0, "sprint": False,
         "reason": "recovery: back and re-evaluate"},
    ]

    def __init__(self):
        self.frames       = deque(maxlen=4)
        self.still_count  = 0
        self.recovery_idx = 0
        self.last_score   = 1.0

    def record(self, screen: Image.Image):
        arr  = to_np(screen)
        # crop centre area — excludes HUD which always changes
        grey = arr[200:800, 300:1620].mean(axis=2).astype(np.float32)
        self.frames.append(grey)

    def score(self) -> float:
        if len(self.frames) < 2:
            return 1.0
        s = ssim(self.frames[-2], self.frames[-1], data_range=255)
        self.last_score = float(s)
        return self.last_score

    def is_wall(self) -> bool:
        s = self.score()
        if s > SSIM_WALL_THRESHOLD:
            self.still_count += 1
        else:
            self.still_count = 0
        return self.still_count >= STUCK_TICKS_REQUIRED

    def is_falling(self) -> bool:
        return self.score() < SSIM_FALL_THRESHOLD

    def next_recovery(self) -> dict:
        r = self.RECOVERY_SEQUENCE[self.recovery_idx % len(self.RECOVERY_SEQUENCE)]
        self.recovery_idx += 1
        self.still_count  = 0
        self.frames.clear()
        return r

    def reset(self):
        self.recovery_idx = 0
        self.still_count  = 0
        self.frames.clear()

stuck = StuckDetector()

# ── Dialogue decisions ────────────────────────────────────────────────────────

DIALOGUE_SYSTEM = """You are playing Witcher 3. An NPC is speaking.
Reply ONLY in this exact format:
ACTION: press
KEY: space
DURATION: 1
REASON: one sentence

Always use space to advance dialogue."""

CHOICES_SYSTEM = """You are Geralt of Rivia in Witcher 3. Stay focused on your mission.
Pick the dialogue choice that best advances the quest objective provided.
Do not explore or describe the scene.
Reply ONLY in this exact format:
ACTION: press
KEY: <w|s|e>
DURATION: 1
REASON: one sentence explaining which choice advances the quest"""

def parse_action(response: str) -> dict:
    lines = {}
    for line in response.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            lines[k.strip().upper()] = v.strip()
    key      = lines.get("KEY", "space").lower()
    reason   = lines.get("REASON", "")
    try:
        duration = float(lines.get("DURATION", "1").split()[0])
    except:
        duration = 1.0
    return {"key": key, "duration": duration, "reason": reason}

# ── System prompts for non-navigation modes ───────────────────────────────────

MODE_SYSTEM = {
    "LOADING":  "Wait for loading screen to finish.",
    "CUTSCENE": "A cutscene is playing. Do not act.",
    "MENU":     "A menu is open. Navigate and close it.",
    "DIALOGUE": DIALOGUE_SYSTEM,
    "CHOICES":  CHOICES_SYSTEM,
    "COMBAT":   """You are playing Witcher 3. Combat is active.
Respond in EXACTLY this format:
ACTION: press
KEY: <e for attack|space for dodge>
DURATION: 1
REASON: <one sentence>""",
}

# ── Main loop ─────────────────────────────────────────────────────────────────

def find_w3() -> str:
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
        print(f"W3 window: {W3_WINDOW_ID or 'not found — using focused window'}")

    print(f"\nStarting in {STARTUP_DELAY}s — switch to Witcher 3...\n")
    for i in range(STARTUP_DELAY, 0, -1):
        print(f"  {i}", end="\r", flush=True)
        time.sleep(1)
    print("UMA running.  Ctrl-C to stop.\n")

    tick      = 0
    last_mode = "EXPLORATION"
    post_load = False

    while True:
        tick += 1

        if not w3_is_focused():
            print(f"[{tick}]  W3 not focused — skipping")
            time.sleep(1)
            continue

        t0 = time.time()

        grab_screen()
        # save immediately — game is still focused
        _screen.save("tick_screen.png")

        mode = detect_mode()

        print(f"\n[{tick}]  {mode.mode}  {mode.details}")

        # ── Loading / death handling ───────────────────────────────────────
        if mode.mode == "LOADING":
            print("  ⏳ loading — waiting...")
            time.sleep(3)
            # wait until actually out of loading screen
            for _ in range(15):
                grab_screen()
                if detect_mode().mode != "LOADING":
                    break
                time.sleep(2)
            post_load = True
            last_mode = "LOADING"
            continue

        # Post-load pause — don't rush into danger after respawn
        if post_load and last_mode == "LOADING":
            print("  ✅ Loaded — pausing 3s, surveying area")
            time.sleep(3)
            mouse_rotate(random.choice([-45, 45]))
            time.sleep(0.5)
            stuck.reset()
            post_load = False

        last_mode = mode.mode

        # ── Exploration: model reads minimap, Python steers ────────────────
        if mode.mode == "EXPLORATION":

            # Stuck check before acting
            stuck.record(_screen)
            if stuck.is_falling():
                print(f"  ⚠  Rapid scene change (ssim={stuck.last_score:.2f}) — stopping")
                time.sleep(2)
                continue

            if stuck.is_wall():
                r = stuck.next_recovery()
                print(f"  ⚠  Wall detected (ssim={stuck.last_score:.2f}) — {r['reason']}")
                mouse_rotate(random.choice([-60, 60]))
                time.sleep(0.3)
                hold(r["key"], r["duration"], sprint=r.get("sprint", False))
                print(f"  🔄 {r['key']} {r['duration']}s")
                time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
                continue

            # release any stuck keys before acting
            if W3_WINDOW_ID:
                subprocess.run(["xdotool", "keyup", "--window", W3_WINDOW_ID, "w"],
                               check=False, capture_output=True)
            time.sleep(0.1)

            nav = read_minimap()
            print(f"  🗺  raw='{nav['raw']}'  direction={nav['direction']}  ({nav['ms']}ms)")

            rotate_and_move(nav)

            # Check interact prompt
            interact = read_interact()
            if interact.strip():
                print(f"  💬 interact: '{interact}' → pressing E")
                press("confirm")

            elapsed = time.time() - t0
            print(f"  ⏱  {int(elapsed*1000)}ms  (nav {nav['ms']}ms)")
            time.sleep(max(0, TICK_INTERVAL - elapsed))
            continue

        # ── Simple modes: no LLM needed ──────────────────────────────────
        if mode.mode == "DIALOGUE":
            subtitle = read_subtitle()
            print(f"  💬 '{subtitle}'")
            press("space")
            continue

        if mode.mode == "CUTSCENE":
            time.sleep(2)
            continue

        # ── All other modes: ask LLM ──────────────────────────────────────
        quest_context = read_quest()

        def with_quest(prompt: str) -> str:
            if quest_context:
                return f'Current quest objective: "{quest_context}"\n\n{prompt}'
            return prompt

        system    = MODE_SYSTEM.get(mode.mode, DIALOGUE_SYSTEM)
        use_small = False

        if mode.mode == "CHOICES":
            choices = read_choices()
            prompt  = with_quest(
                f'Dialogue choices: "{choices}"\n'
                f'Pick the option that best advances the current quest objective.\n'
                f'Navigate with w (up) or s (down), confirm with e.\n'
                f'Reply with the key to press next.'
            )
            image   = crop(CHOICE_REGION)   # small crop, not full screen

        elif mode.mode == "COMBAT":
            prompt    = "Combat is active. Attack with E or dodge with Space."
            image     = _screen
            use_small = True

        elif mode.mode == "MENU":
            prompt    = "A menu is open. Close or navigate it."
            image     = _screen
            use_small = True

        else:
            prompt = "What do you see? What do you do?"
            image  = _screen

        t_llm    = time.time()
        response = ask_llm(prompt, system, image=image,
                           temperature=0.4, max_tokens=128, use_small=use_small)
        llm_ms   = int((time.time() - t_llm) * 1000)

        print(f"  🧠 [{llm_ms}ms] {response}")

        if response:
            parsed = parse_action(response)
            press(parsed["key"])
            print(f"  ⌨  {parsed['key']}  —  {parsed['reason']}")

        elapsed = time.time() - t0
        print(f"  ⏱  {int(elapsed*1000)}ms  (llm {llm_ms}ms)")
        time.sleep(max(0, TICK_INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nUMA stopped.")
