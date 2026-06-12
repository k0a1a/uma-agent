#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v3.0
===================================
A highly experimental algorithmic agent capable of navigating
and interacting with the Witcher 3 world, perceiving it
exclusively through vision — the same way a human player would.

What we learned building v1 and v2:
  - Left stick forward/back only. Right stick steering only.
  - Never touch the keyboard. No xdotool.
  - Minimap rotates with Geralt. Top = his forward direction always.
  - Navigate by comparing white arrow (Geralt) to white dotted path.
  - Commit to a direction for several ticks — re-reading every tick causes
    oscillation because the minimap rotates as Geralt turns.
  - Dialogue: A button to advance. Left stick to navigate choices.
  - KV cache bust required — append timestamp to each minimap prompt.
  - SSIM stuck detection works reliably for wall detection.

Architecture:
  b550.local          — Witcher 3 + controller_server.py + screen_server.py
  beacon.x.k0a1a.net  — LM Studio (Qwen2.5-VL-7B) + ocr_server.py
  laptop              — this script (orchestrator, no GPU, no display)

Services:
  b550:5002  controller input  (vgamepad virtual Xbox 360)
  b550:5003  screen capture    (mss HTTP API)
  beacon:1234 LM Studio        (OpenAI-compatible vision API)
  beacon:5001 OCR              (easyocr HTTP service)

Usage:
    python3 uma3.py
    # optional overrides:
    GAME_PC=b550.local BEACON=beacon.x.k0a1a.net python3 uma3.py

Authors:
    Danja Vasiliev / Tactical Tech
    in collaboration with Claude (Anthropic)
"""

import os, sys, time, base64, io, random
from collections import deque
from typing import Optional

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

# ── Game / agent config ───────────────────────────────────────────────────────

TICK_INTERVAL    = 1.5    # seconds between ticks
NAV_COMMIT_TICKS = 4     # re-read minimap every N ticks (prevents oscillation)
STEER_AMOUNT     = 0.35  # right stick x for left/right turns (gentle)
SSIM_THRESHOLD   = 0.97  # scene similarity above this = stuck against wall
STUCK_TICKS      = 3     # consecutive still frames before recovery

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
    checks = [
        (f"{CONTROLLER_URL}/health", "controller (b550:5002)"),
        (f"{SCREEN_URL}/health",     "screen     (b550:5003)"),
        (f"{OCR_URL.replace('/ocr','/health')}", "ocr (beacon:5001)"),
    ]
    for url, name in checks:
        try:
            requests.get(url, timeout=3).json()
            print(f"  ✓  {name}")
        except Exception as e:
            print(f"  ✗  {name}: {e}")
            ok = False
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        ids    = [m.id for m in client.models.list().data]
        print(f"  ✓  LM Studio (beacon:1234): {ids}")
    except Exception as e:
        print(f"  ✗  LM Studio: {e}")
        ok = False
    return ok

print("UMA v3.0 — checking services...")
if not check_services():
    print("\nOne or more services unreachable. Check that all servers are running.")
    sys.exit(1)

print("\nAll services OK.\n")
llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# ── Screen capture ────────────────────────────────────────────────────────────

def get_screen() -> Image.Image:
    resp = requests.get(f"{SCREEN_URL}/screenshot", timeout=5)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def get_region(region: tuple) -> Image.Image:
    l, t, w, h = region
    resp = requests.get(f"{SCREEN_URL}/region",
                        params={"l":l,"t":t,"w":w,"h":h,"quality":92},
                        timeout=5)
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def crop(screen: Image.Image, region: tuple) -> Image.Image:
    l, t, w, h = region
    return screen.crop((l, t, l+w, t+h))

# ── Mode detection ────────────────────────────────────────────────────────────

def px(img: Image.Image, x: int, y: int) -> tuple:
    return img.getpixel((x, y))[:3]

def detect_mode(screen: Image.Image) -> str:
    arr = np.array(screen)

    # LOADING — all 5 sample points near black
    pts  = [(960,540),(480,270),(1440,270),(480,810),(1440,810)]
    if all(sum(px(screen,x,y))/3 < 20 for x,y in pts):
        return "LOADING"

    # CUTSCENE — black letterbox top and bottom, bright centre
    if (sum(px(screen,960,40))/3   < 15 and
        sum(px(screen,960,1040))/3 < 15 and
        sum(px(screen,960,540))/3  > 30):
        return "CUTSCENE"

    # CHOICES — dark semi-transparent box with bright text
    # Mean < 85 distinguishes choice overlay from bright game world
    l,t,w,h = CHOICE_REGION
    cg = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2GRAY)
    if ((cg > 160).sum()/cg.size > 0.04 and
        ((cg>15)&(cg<90)).sum()/cg.size > 0.15 and
        cg.mean() < 85):
        return "CHOICES"

    # DIALOGUE — bright subtitle text on darker background
    l,t,w,h = SUBTITLE_REGION
    sg = cv2.cvtColor(arr[t:t+h, l:l+w], cv2.COLOR_RGB2GRAY)
    if (sg > 160).sum() > 200 and sg.mean() < 130:
        return "DIALOGUE"

    # COMBAT — red enemy health bars at top centre
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

# ── LLM ───────────────────────────────────────────────────────────────────────

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

# ── Controller (all input via vgamepad — no keyboard) ─────────────────────────

def _post(path: str, data: dict):
    try:
        requests.post(f"{CONTROLLER_URL}{path}", json=data, timeout=5)
    except Exception as e:
        print(f"  ⚠  controller: {e}")

def navigate(left_y: float, right_x: float, duration: float = 0.5):
    """
    Single call to move Geralt.
    left_y  : forward/back on left stick  (+1=forward, -1=back)
    right_x : steer on right stick        (-1=left, +1=right)
    Both sticks are set simultaneously and released after duration.
    """
    _post("/navigate", {
        "left_x":  0.0,
        "left_y":  left_y,
        "right_x": right_x,
        "right_y": 0.0,
        "duration": duration,
    })

def button(name: str, duration: float = 0.15):
    _post("/button", {"button": name, "duration": duration})

def release():
    _post("/release", {})

# ── Minimap navigation ────────────────────────────────────────────────────────

MINIMAP_SYSTEM = (
    "You are reading a rotating minimap in Witcher 3. "
    "The map rotates with the player — the top of the minimap is ALWAYS "
    "the player's current forward direction. "
    "Reply with exactly one word."
)

MINIMAP_PROMPT = (
    "Look at this Witcher 3 minimap.\n"
    "The WHITE ARROW at the centre of the circle is Geralt's position.\n"
    "The arrow always points UP, which is Geralt's FORWARD direction.\n"
    "Find the WHITE DOTTED PATH — small white dots forming a trail.\n"
    "Where is the nearest white dot relative to the centre arrow?\n"
    "  forward = dot is above the arrow (ahead of Geralt)\n"
    "  left    = dot is to the left of the arrow\n"
    "  right   = dot is to the right of the arrow\n"
    "  behind  = dot is below the arrow (behind Geralt)\n"
    "  none    = no white dots visible\n"
    "Reply ONE word: forward / left / right / behind / none"
)

# Steer amounts for right stick — gentle to avoid oscillation
STEER = {
    "forward": 0.0,
    "left":   -0.25,   # was -0.35
    "right":  +0.25,   # was +0.35
    "behind":  0.0,    # don't spin — just go forward, re-read next commit
    "none":    0.0,
}

class Navigator:
    """
    Reads minimap every NAV_COMMIT_TICKS ticks.
    Holds the direction between reads to prevent oscillation feedback loop.
    (Minimap rotates with Geralt, so re-reading every tick after a turn
    always shows the path in the opposite direction → oscillation.)
    """
    def __init__(self):
        self.direction  = "forward"
        self.ticks_left = 0

    def tick(self, screen: Image.Image):
        if self.ticks_left <= 0:
            # Time to re-read minimap
            minimap = crop(screen, MINIMAP_REGION)
            minimap.save("minimap_live.png")
            prompt = MINIMAP_PROMPT + f"\n[t:{int(time.time())}]"
            t   = time.time()
            raw = ask(prompt, MINIMAP_SYSTEM, image=minimap,
                      temperature=0.1, max_tokens=5)
            ms  = int((time.time()-t)*1000)
            word = raw.strip().lower().split()[0] if raw.strip() else "forward"
            if word not in STEER:
                word = "forward"
            self.direction  = word
            self.ticks_left = NAV_COMMIT_TICKS
            print(f"  🗺  {self.direction}  ({ms}ms)  [commit {NAV_COMMIT_TICKS}]")
        else:
            self.ticks_left -= 1
            print(f"  🗺  {self.direction}  (held, {self.ticks_left} left)")

        steer = STEER[self.direction]
        navigate(left_y=1.0, right_x=steer, duration=0.5)
        print(f"  🕹  fwd + steer {steer:+.2f}  [{self.direction}]")

    def reset(self):
        self.direction  = "forward"
        self.ticks_left = 0

# ── Stuck detection ───────────────────────────────────────────────────────────

class StuckDetector:
    """
    Detects wall collision by comparing consecutive scene frames with SSIM.
    High SSIM = scene not changing = Geralt is stuck against something.
    """
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
        self.last = float(ssim(self.frames[-2], self.frames[-1], data_range=255))
        if self.last > SSIM_THRESHOLD:
            self.still += 1
        else:
            self.still = 0
        return self.still >= STUCK_TICKS

    def recover(self):
        """Back up with a random steer to get away from the obstacle."""
        self.still  = 0
        self.frames.clear()
        steer = random.choice([-0.5, 0.5])
        print(f"  ⚠  stuck (ssim={self.last:.3f}) — backing up + steer {steer:+.1f}")
        navigate(left_y=-1.0, right_x=steer, duration=0.8)
        time.sleep(0.1)

    def reset(self):
        self.still  = 0
        self.frames.clear()

# ── Dialogue / choices ────────────────────────────────────────────────────────

CHOICES_SYSTEM = (
    "You are Geralt in Witcher 3. You follow the main story quest.\n"
    "Dialogue choices are visible. Pick the one that best advances the quest.\n"
    "Choose as Geralt would: pragmatic, direct, lore-consistent.\n"
    "Reply ONLY in this exact format:\n"
    "ACTION: navigate\n"
    "DIRECTION: <up|down|confirm>\n"
    "REASON: one sentence\n\n"
    "up      = move selection to previous choice\n"
    "down    = move selection to next choice\n"
    "confirm = select the currently highlighted choice"
)

def parse_direction(response: str) -> str:
    for line in response.strip().splitlines():
        if "DIRECTION:" in line.upper():
            return line.split(":",1)[1].strip().lower()
    return "confirm"

def navigate_choice(direction: str):
    """Navigate choices with left stick, confirm with A button."""
    if direction == "up":
        navigate(left_y=1.0, right_x=0.0, duration=0.25)
    elif direction == "down":
        navigate(left_y=-1.0, right_x=0.0, duration=0.25)
    elif direction == "confirm":
        button("a")
    time.sleep(0.2)

# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    nav   = Navigator()
    stuck = StuckDetector()

    print(f"Starting in 5s — make sure W3 is running on {GAME_PC}...\n")
    for i in range(5, 0, -1):
        print(f"  {i}", end="\r", flush=True)
        time.sleep(1)
    print("UMA v3.0 running.  Ctrl-C to stop.\n")

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
            print("  ⏳ waiting for load...")
            time.sleep(4)
            for _ in range(15):
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

        # Post-load: pause then reset nav state
        if post_load and last_mode == "LOADING":
            print("  ✅ loaded — pausing 3s")
            time.sleep(3)
            nav.reset()
            stuck.reset()
            post_load = False

        last_mode = mode

        # ── Cutscene ───────────────────────────────────────────────────────
        if mode == "CUTSCENE":
            print("  🎬 cutscene — waiting 2s")
            time.sleep(2)
            continue

        # ── Dialogue — A to advance, no LLM needed ─────────────────────────
        if mode == "DIALOGUE":
            sub = ocr(crop(screen, SUBTITLE_REGION))
            print(f"  💬 '{sub}'")
            button("a")
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Choices — LLM reads and decides ───────────────────────────────
        if mode == "CHOICES":
            quest       = ocr(crop(screen, QUEST_REGION))
            choices_img = crop(screen, CHOICE_REGION)
            choices_txt = ocr(choices_img)
            prompt = (
                f'Quest objective: "{quest}"\n'
                f'Dialogue choices on screen: "{choices_txt}"\n'
                f'What do you do?'
            )
            t_llm = time.time()
            resp  = ask(prompt, CHOICES_SYSTEM,
                        image=choices_img, temperature=0.3, max_tokens=64)
            ms    = int((time.time()-t_llm)*1000)
            print(f"  🧠 [{ms}ms] {resp}")
            navigate_choice(parse_direction(resp))
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Combat — attack or dodge ───────────────────────────────────────
        if mode == "COMBAT":
            # Simple pattern: attack twice, dodge once
            # TODO v3.1: use vision to detect enemy type and choose sign
            button("x")          # fast attack
            time.sleep(0.3)
            button("x")          # fast attack
            time.sleep(0.3)
            button("b")          # dodge
            print("  ⚔  attack attack dodge")
            time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
            continue

        # ── Exploration ────────────────────────────────────────────────────
        if mode == "EXPLORATION":

            # SSIM stuck check
            stuck.record(screen)
            if stuck.is_stuck():
                stuck.recover()
                nav.reset()   # re-read minimap after recovery
                time.sleep(max(0, TICK_INTERVAL - (time.time()-t0)))
                continue

            # Check interact prompt — press A if visible
            interact = ocr(crop(screen, INTERACT_REGION))
            if interact.strip():
                print(f"  💬 interact: '{interact}' → A")
                button("a")
                time.sleep(0.5)
                nav.reset()
                continue

            # Navigate toward quest objective
            nav.tick(screen)

            elapsed = time.time() - t0
            print(f"  ⏱  {int(elapsed*1000)}ms")
            time.sleep(max(0, TICK_INTERVAL - elapsed))

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        release()
        print("\n\nUMA stopped. Controller released.")
