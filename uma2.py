#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v2.0
===================================
A highly experimental algorithmic agent capable of navigating
and interacting with the Witcher 3 world, perceiving it
exclusively through vision — the same way a human player would.

Architecture:
  Game PC  — Witcher 3 + controller_server.py + screen_server.py
  Beacon   — LM Studio (Qwen2-VL) + ocr_server.py
  Laptop   — this script (orchestrator, no GPU needed)

Services:
  GAME_PC:  controller input (port 5002) + screen capture (port 5003)
  BEACON:   LM Studio (port 1234) + OCR (port 5001)

Usage:
    export GAME_PC="192.168.x.x"
    export BEACON="192.168.x.x"
    python3 uma2.py

Authors:
    Danja Vasiliev / Tactical Tech
    in collaboration with Claude (Anthropic)
"""

import os, sys, time, base64, io, random
import numpy as np
import cv2
from typing import Optional

import requests
from PIL import Image
from openai import OpenAI

# ── Service endpoints ─────────────────────────────────────────────────────────

GAME_PC = os.environ.get("GAME_PC", "b550.local")
BEACON  = os.environ.get("BEACON",  "beacon.x.k0a1a.net")

CONTROLLER_URL = f"http://{GAME_PC}:5002"
SCREEN_URL     = f"http://{GAME_PC}:5003"
LM_STUDIO_URL  = f"http://{BEACON}:1234/v1"
OCR_URL        = f"http://{BEACON}:5001/ocr"
LM_MODEL       = os.environ.get("LM_MODEL", "local-model")

# ── Game config ───────────────────────────────────────────────────────────────

GAME_W, GAME_H = 1920, 1080
TICK_INTERVAL  = 1.5   # seconds between ticks

# HUD regions (left, top, width, height)
MINIMAP_REGION    = (1660,  36, 213, 215)
MINIMAP_CENTRE_PX = (1766, 143)
QUEST_REGION      = (1220, 225, 340, 100)

SUBTITLE_REGION = ( 450, 655, 570,  55)
CHOICE_REGION   = ( 865, 500, 360, 125)

INTERACT_REGION   = (  55, 350, 230, 120)
ENEMY_HP_REGION   = ( 650,  44, 580,  24)

# Controller — W3 button mapping
CONFIRM  = "a"      # interact / select
CANCEL   = "b"      # cancel / dodge
ATTACK   = "x"      # fast attack
DODGE    = "b"
ADV_DLG  = "a"      # advance dialogue (A button in W3)

# ── Init ──────────────────────────────────────────────────────────────────────

def check_services():
    """Verify all remote services are reachable before starting."""
    services = [
        (f"{CONTROLLER_URL}/health", "Controller server (game PC)"),
        (f"{SCREEN_URL}/health",     "Screen server (game PC)"),
        (f"{OCR_URL.replace('/ocr', '/health')}", "OCR server (beacon)"),
    ]
    ok = True
    for url, name in services:
        try:
            r = requests.get(url, timeout=3)
            print(f"  ✓  {name}: {r.json()}")
        except Exception as e:
            print(f"  ✗  {name}: {e}")
            ok = False

    # LM Studio
    try:
        llm    = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        models = llm.models.list()
        print(f"  ✓  LM Studio (beacon): {[m.id for m in models.data]}")
    except Exception as e:
        print(f"  ✗  LM Studio (beacon): {e}")
        ok = False

    return ok

print("UMA v2.0 — checking services...")
if not check_services():
    print("\nSome services unreachable. Check IPs and that servers are running.")
    sys.exit(1)

print("\nAll services OK.\n")
llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ── Screen capture (from game PC via HTTP) ────────────────────────────────────

def get_screen() -> Image.Image:
    """Fetch full screenshot from game PC."""
    resp = requests.get(f"{SCREEN_URL}/screenshot", timeout=5)
    return Image.open(io.BytesIO(resp.content))

def get_region(region: tuple) -> Image.Image:
    """Fetch a specific screen region from game PC."""
    l, t, w, h = region
    resp = requests.get(
        f"{SCREEN_URL}/region",
        params={"l": l, "t": t, "w": w, "h": h, "quality": 92},
        timeout=5
    )
    return Image.open(io.BytesIO(resp.content))

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── Mode detection ────────────────────────────────────────────────────────────

def px(img: Image.Image, x: int, y: int) -> tuple:
    return img.getpixel((x, y))[:3]

def detect_mode(screen: Image.Image) -> str:
    """
    Fast pixel-based mode detection on the fetched screen image.
    Returns mode string.
    """
    arr = np.array(screen)

    # LOADING — all 5 points near black
    pts  = [(960,540),(480,270),(1440,270),(480,810),(1440,810)]
    brts = [sum(px(screen,x,y))/3 for x,y in pts]
    if all(b < 20 for b in brts):
        return "LOADING"

    # CUTSCENE — black bars top and bottom, bright centre
    top_b = sum(px(screen, 960,  40))/3
    bot_b = sum(px(screen, 960, 1040))/3
    ctr_b = sum(px(screen, 960,  540))/3
    if top_b < 15 and bot_b < 15 and ctr_b > 30:
        return "CUTSCENE"

    # CHOICES — bright text in choice region
    l, t, w, h = CHOICE_REGION
    cg = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2GRAY)
    bright_ratio    = float((cg > 160).sum()) / cg.size
    dark_ratio      = float(((cg > 15) & (cg < 90)).sum()) / cg.size
    mean_brightness = float(cg.mean())

    if bright_ratio > 0.04 and dark_ratio > 0.15 and mean_brightness < 85:
        screen.crop((l, t, l+w, t+h)).save("last_choice_crop.png")
        return "CHOICES"

    # DIALOGUE — bright text in subtitle region
    l,t,w,h = SUBTITLE_REGION
    sg = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2GRAY)
    if int((sg > 160).sum()) > 200 and float(sg.mean()) < 130:
        return "DIALOGUE"

    # COMBAT — red health bars
    l,t,w,h = ENEMY_HP_REGION
    hh = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2HSV)
    rm = cv2.inRange(hh, np.array([0,140,100]), np.array([12,255,255]))
    if rm.sum()//255 > 60:
        return "COMBAT"

    return "EXPLORATION"

# ── OCR (from beacon) ─────────────────────────────────────────────────────────

def ocr(img: Image.Image) -> str:
    """Send image to OCR service on beacon."""
    from PIL import ImageEnhance
    img = ImageEnhance.Contrast(img.convert("L")).enhance(2.5).convert("RGB")
    try:
        resp = requests.post(OCR_URL,
                             json={"image": to_b64(img)},
                             timeout=5)
        return resp.json().get("text", "").strip()
    except Exception as e:
        print(f"  ⚠  OCR error: {e}")
        return ""

# ── LLM inference (from beacon) ──────────────────────────────────────────────

def ask(prompt: str, system: str,
        image: Optional[Image.Image] = None,
        temperature: float = 0.3,
        max_tokens: int = 64) -> str:
    content = []
    if image is not None:
        b64 = to_b64(image)
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    try:
        resp = llm.chat.completions.create(
            model       = LM_MODEL,
            messages    = [{"role": "system", "content": system},
                           {"role": "user",   "content": content}],
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠  LLM error: {e}")
        return ""

# ── Controller input (to game PC) ─────────────────────────────────────────────

def stick(which: str, x: float, y: float, duration: float = 0.0):
    """Move analog stick."""
    requests.post(f"{CONTROLLER_URL}/stick",
                  json={"stick": which, "x": x, "y": y, "duration": duration},
                  timeout=3)

def button(name: str, duration: float = 0.15):
    """Press a controller button."""
    requests.post(f"{CONTROLLER_URL}/button",
                  json={"button": name, "duration": duration},
                  timeout=3)

def release():
    """Release all controller inputs."""
    requests.post(f"{CONTROLLER_URL}/release", timeout=3)


# ── Navigation ────────────────────────────────────────────────────────────────

MINIMAP_SYSTEM = (
    "You are a navigation assistant reading a rotating game minimap. "
    "The map rotates with the player — top is always the player's forward direction. "
    "Reply with one word only."
)

MINIMAP_PROMPT = (
    "This is a rotating minimap from Witcher 3. "
    "The WHITE ARROW in the centre of the circle represents Geralt. "
    "The arrow always points UP — up is always Geralt's FORWARD direction. "
    "Find the WHITE DOTTED PATH (small white dots forming a trail). "
    "Where is the NEAREST white dot relative to the centre arrow?\n"
    "- If the dot is ABOVE the arrow: forward\n"
    "- If the dot is LEFT of the arrow: left\n"
    "- If the dot is RIGHT of the arrow: right\n"
    "- If the dot is BELOW the arrow: behind\n"
    "Reply with ONE word only: forward / left / right / behind / none"
)

STEER = {
    "forward":       0.0,
    "forward-left": -0.4,
    "left":         -0.9,
    "behind":       -1.0,
    "forward-right": 0.4,
    "right":         0.9,
    "none":          0.0,
}

# Navigation state
nav_direction   = "forward"
nav_ticks_left  = 0
NAV_COMMIT_TICKS = 4

def query_minimap(screen: Image.Image) -> str:
    """Crop minimap, ask LLM for direction, return direction string."""
    minimap = screen.crop((
        MINIMAP_REGION[0], MINIMAP_REGION[1],
        MINIMAP_REGION[0]+MINIMAP_REGION[2],
        MINIMAP_REGION[1]+MINIMAP_REGION[3]
    ))
    minimap.save("minimap_live.png")

    prompt = MINIMAP_PROMPT + f"\n[t:{int(time.time())}]"
    t   = time.time()
    raw = ask(prompt, MINIMAP_SYSTEM,
              image=minimap, temperature=0.1, max_tokens=5)
    ms  = int((time.time()-t)*1000)
    print(f"  🗺  raw='{raw}'  ({ms}ms)")

    direction = raw.strip().lower().split()[0] if raw.strip() else "forward"
    if direction not in ("forward","forward-left","left","behind","forward-right","right","none"):
        direction = "forward"
    return direction

def navigate(direction: str):
    steer = STEER.get(direction, 0.0)
    requests.post(f"{CONTROLLER_URL}/navigate",
                  json={"left_x":  0.0,
                        "left_y":  1.0,
                        "right_x": steer,
                        "right_y": 0.0,
                        "duration": 0.6},
                  timeout=5)
    print(f"  🕹  forward + steer {steer:+.1f}  [{direction}]")

def navigate_tick(screen: Image.Image):
    global nav_direction, nav_ticks_left

    if nav_ticks_left <= 0:
        nav_direction  = query_minimap(screen)
        nav_ticks_left = NAV_COMMIT_TICKS
        print(f"  🗺  new direction: {nav_direction} (commit {NAV_COMMIT_TICKS} ticks)")
    else:
        nav_ticks_left -= 1
        print(f"  🗺  holding: {nav_direction} ({nav_ticks_left} ticks left)")

    steer = STEER.get(nav_direction, 0.0)
    requests.post(f"{CONTROLLER_URL}/navigate",
                  json={"left_x": 0.0, "left_y": 1.0,
                        "right_x": steer, "right_y": 0.0,
                        "duration": 0.5},
                  timeout=5)
    print(f"  🕹  forward + steer {steer:+.2f}  [{nav_direction}]")

# ── Stuck detection ───────────────────────────────────────────────────────────

class StuckDetector:
    def __init__(self, threshold: float = 0.97, required: int = 3):
        from collections import deque
        from skimage.metrics import structural_similarity as ssim_fn
        self.ssim_fn   = ssim_fn
        self.frames    = deque(maxlen=4)
        self.still     = 0
        self.threshold = threshold
        self.last      = 1.0
        self.rec_idx   = 0

    def record(self, screen: Image.Image):
        arr  = np.array(screen)
        grey = arr[200:800, 300:1620].mean(axis=2).astype(np.float32)
        self.frames.append(grey)

    def is_stuck(self) -> bool:
        if len(self.frames) < 2:
            return False
        self.last = float(self.ssim_fn(self.frames[-2], self.frames[-1],
                                        data_range=255))
        if self.last > self.threshold:
            self.still += 1
        else:
            self.still = 0
        return self.still >= 3

    def recover(self):
        self.still = 0
        self.frames.clear()
        print(f"  ⚠  stuck (ssim={self.last:.2f}) — backing up")
        requests.post(f"{CONTROLLER_URL}/navigate",
                      json={"left_x":  0.0, "left_y":  -1.0,
                            "right_x": random.choice([-0.5, 0.5]),
                            "right_y": 0.0, "duration": 0.8},
                      timeout=5)
        self.rec_idx += 1

stuck = StuckDetector()

# ── Prompts ───────────────────────────────────────────────────────────────────

CHOICES_SYSTEM = """You are Geralt in Witcher 3. Pick the choice that advances the quest.
Reply ONLY:
ACTION: press
BUTTON: <up|down|a>
REASON: one sentence

Use 'up' to go to previous choice, 'down' to go to next choice, 'a' to confirm."""

COMBAT_SYSTEM = """You are Geralt in Witcher 3 combat.
Reply ONLY in this exact format:
ACTION: press
BUTTON: <x|b>
REASON: one sentence

x = fast attack, b = dodge"""

def navigate_choice(direction: str):
    """Navigate dialogue choices with left stick."""
    if direction == "dpad_up":
        stick("left", 0.0, 1.0, duration=0.3)
    elif direction == "dpad_down":
        stick("left", 0.0, -1.0, duration=0.3)
    time.sleep(0.2)

def confirm_choice():
    button("a", duration=0.15)

def parse_button(response: str) -> str:
    for line in response.strip().splitlines():
        if "BUTTON:" in line.upper():
            btn = line.split(":", 1)[1].strip().lower()
            btn = btn.replace("d-pad_", "dpad_").replace("d_pad_", "dpad_")
            return btn
    return "a"

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    stuck = StuckDetector()

    print(f"Starting in 5s...\n")
    for i in range(5, 0, -1):
        print(f"  {i}", end="\r", flush=True)
        time.sleep(1)
    print("UMA running.\n")

    tick      = 0
    last_mode = ""
    post_load = False

    while True:
        tick += 1
        t0 = time.time()

        # Fetch screen from game PC
        try:
            screen = get_screen()
        except Exception as e:
            print(f"[{tick}]  ⚠  screen fetch failed: {e}")
            time.sleep(2)
            continue

        mode = detect_mode(screen)
        print(f"\n[{tick}]  {mode}")

        # ── Loading ────────────────────────────────────────────────────────
        if mode == "LOADING":
            print("  ⏳ loading — waiting...")
            time.sleep(4)
            for _ in range(10):
                try:
                    s = get_screen()
                    if detect_mode(s) != "LOADING":
                        break
                except:
                    pass
                time.sleep(2)
            post_load = True
            last_mode = "LOADING"
            continue

        # Post-load recovery
        if post_load and last_mode == "LOADING":
            print("  ✅ loaded — pausing 3s")
            time.sleep(3)
            stuck = StuckDetector()
            post_load = False

        last_mode = mode

        # ── Cutscene ───────────────────────────────────────────────────────
        if mode == "CUTSCENE":
            print("  🎬 cutscene — waiting")
            time.sleep(2)
            continue

        # ── Dialogue — auto-advance, no LLM needed ─────────────────────────
        if mode == "DIALOGUE":
            sub = ocr(screen.crop((
                SUBTITLE_REGION[0], SUBTITLE_REGION[1],
                SUBTITLE_REGION[0]+SUBTITLE_REGION[2],
                SUBTITLE_REGION[1]+SUBTITLE_REGION[3]
            )))
            print(f"  💬 '{sub}'")
            button(ADV_DLG)
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Choices — LLM decides ──────────────────────────────────────────
        if mode == "CHOICES":
            quest   = ocr(screen.crop((
                QUEST_REGION[0], QUEST_REGION[1],
                QUEST_REGION[0]+QUEST_REGION[2],
                QUEST_REGION[1]+QUEST_REGION[3]
            )))
            choices_img = screen.crop((
                CHOICE_REGION[0], CHOICE_REGION[1],
                CHOICE_REGION[0]+CHOICE_REGION[2],
                CHOICE_REGION[1]+CHOICE_REGION[3]
            ))
            choices_text = ocr(choices_img)
            prompt = (
                f'Quest: "{quest}"\n'
                f'Choices: "{choices_text}"\n'
                f'Pick the option that best advances the quest.'
            )
            t_llm    = time.time()
            response = ask(prompt, CHOICES_SYSTEM,
                          image=choices_img, temperature=0.3, max_tokens=64)
            llm_ms   = int((time.time()-t_llm)*1000)
            print(f"  🧠 [{llm_ms}ms] {response}")
            btn = parse_button(response)
            if btn in ("up", "dpad_up"):
                navigate_choice("dpad_up")
            elif btn in ("down", "dpad_down"):
                navigate_choice("dpad_down")
            elif btn == "a":
                confirm_choice()
            print(f"  🎮  {btn}")
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Combat ─────────────────────────────────────────────────────────
        if mode == "COMBAT":
            t_llm    = time.time()
            response = ask("Combat active. Attack or dodge?", COMBAT_SYSTEM,
                          temperature=0.2, max_tokens=32)
            llm_ms   = int((time.time()-t_llm)*1000)
            btn = parse_button(response)
            print(f"  ⚔  [{llm_ms}ms] {btn}")
            button(btn)
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Exploration ────────────────────────────────────────────────────
        if mode == "EXPLORATION":
            stuck.record(screen)
            if stuck.is_stuck():
                stuck.recover()
                time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
                continue

            navigate_tick(screen)

            # Check interact prompt
            interact_img = screen.crop((
                INTERACT_REGION[0], INTERACT_REGION[1],
                INTERACT_REGION[0]+INTERACT_REGION[2],
                INTERACT_REGION[1]+INTERACT_REGION[3]
            ))
            interact = ocr(interact_img)
            if interact.strip():
                print(f"  💬 interact: '{interact}' → A")
                button(CONFIRM)

            elapsed = time.time() - t0
            print(f"  ⏱  {int(elapsed*1000)}ms")
            time.sleep(max(0, TICK_INTERVAL - elapsed))

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        release()
        print("\n\nUMA stopped. Controller released.")
