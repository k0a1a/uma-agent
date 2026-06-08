#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent
============================
A highly experimental algorithmic agent capable of navigating
and interacting with the Witcher 3 world, perceiving it
exclusively through vision — the same way a human player would.

Version:   1.0-local
Model:     Qwen2-VL (vision) via LM Studio local inference
Platform:  Linux, X11, Openbox, Nvidia, single 1080p display
Game:      Witcher 3 Next Gen, Steam/Proton, Borderless Window

Dependencies:
    pip install openai easyocr mss pillow numpy opencv-python

System:
    sudo apt install xdotool

LM Studio setup:
    1. Load a Qwen2-VL model (7B recommended)
    2. Start Local Server (default: http://localhost:1234)
    3. Verify: curl http://localhost:1234/v1/models

Usage:
    export W3_WINDOW_ID=$(xdotool search --name "Witcher" | head -1)
    python3 w3agent.py

Research questions (v1.0):
    - Can a local VLM follow a game narrative from subtitles alone?
    - How does Qwen2-VL scene understanding compare to Claude?
    - Where does local inference latency become a bottleneck?
    - What is the quality floor for lore-aware dialogue decisions?

Authors:
    Danja Vasiliev / Tactical Tech
    in collaboration with Claude (Anthropic)
"""

import os
import sys
import time
import base64
import subprocess
import numpy as np
from io import BytesIO
from dataclasses import dataclass, field
from typing import Optional

import requests
import mss
import cv2
from PIL import Image, ImageEnhance
from openai import OpenAI

# ── LM Studio configuration ───────────────────────────────────────────────────

LM_STUDIO_URL   = os.environ.get("LM_STUDIO_URL", "http://beacon:1234/v1")
BEACON_OCR_URL  = os.environ.get("BEACON_OCR_URL", "http://192.168.12.232:5001")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "local-model")

# Generation parameters
# Low temperature for consistent navigation decisions
# Higher for dialogue choices (more character-appropriate variation)
NAV_TEMPERATURE      = 0.2
DIALOGUE_TEMPERATURE = 0.5
MAX_TOKENS_NAV       = 256
MAX_TOKENS_DIALOGUE  = 512

# ── Game configuration ────────────────────────────────────────────────────────

W3_WINDOW_ID = os.environ.get("W3_WINDOW_ID", "")
GAME_W       = 1920
GAME_H       = 1080

TICK_INTERVAL = 2.5
STARTUP_DELAY = 6
MAX_HISTORY   = 20     # turns before trimming (kept shorter for local model)

# ── HUD regions (1920x1080, Next Gen W3) ──────────────────────────────────────

MINIMAP_REGION  = (1255,  35, 200, 185)
QUEST_REGION    = (1195, 218, 265, 130)
SUBTITLE_REGION = ( 420, 648, 610,  78)
CHOICE_REGION   = ( 855, 496, 250, 130)
INTERACT_REGION = (  55, 350, 230, 120)
ENEMY_HP_REGION = ( 650,  44, 580,  24)

# ── Initialise ────────────────────────────────────────────────────────────────

print("Connecting to LM Studio...")
llm = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# Verify connection
try:
    models = llm.models.list()
    print(f"LM Studio connected. Available models:")
    for m in models.data:
        print(f"  {m.id}")
except Exception as e:
    print(f"Error connecting to LM Studio at {LM_STUDIO_URL}: {e}")
    print("Make sure LM Studio is running with a model loaded and server started.")
    sys.exit(1)

print(f"OCR server: {BEACON_OCR_URL}\n")

sct = mss.mss()

# ── Screen capture ────────────────────────────────────────────────────────────

_screen: Optional[Image.Image] = None

def grab_screen() -> Image.Image:
    global _screen
    raw     = sct.grab(sct.monitors[1])
    _screen = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return _screen

def crop(region: tuple) -> Image.Image:
    if _screen is None:
        grab_screen()
    l, t, w, h = region
    return _screen.crop((l, t, l + w, t + h))

def to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def px(x: int, y: int) -> tuple:
    if _screen is None:
        grab_screen()
    return _screen.getpixel((x, y))[:3]

# ── Mode detection ────────────────────────────────────────────────────────────

@dataclass
class ScreenMode:
    mode:       str
    confidence: str  = "high"
    details:    dict = field(default_factory=dict)

def detect_mode() -> ScreenMode:
    centre = px(960, 540)
    if sum(centre) / 3 < 20:
        return ScreenMode("LOADING",
                          details={"brightness": round(sum(centre)/3, 1)})

    top_b = sum(px(960,  40)) / 3
    bot_b = sum(px(960, 1040)) / 3
    if top_b < 15 and bot_b < 15:
        return ScreenMode("CUTSCENE",
                          details={"top": round(top_b,1), "bot": round(bot_b,1)})

    pts  = [(480,300),(480,540),(480,780),(960,300),(960,540),(960,780)]
    brts = [sum(px(x, y)) / 3 for x, y in pts]
    var, mean = float(np.var(brts)), float(np.mean(brts))
    mm_c = sum(px(1355, 128)) / 3
    if var < 150 and 65 < mean < 185 and mm_c < 55:
        return ScreenMode("MENU",
                          details={"var": round(var,1), "mean": round(mean,1)})

    cg           = cv2.cvtColor(to_np(crop(CHOICE_REGION)), cv2.COLOR_RGB2GRAY)
    bright_ratio = float((cg > 160).sum()) / cg.size
    dark_ratio   = float(((cg > 15) & (cg < 90)).sum()) / cg.size
    if bright_ratio > 0.012 and dark_ratio > 0.08:
        return ScreenMode("CHOICES",
                          details={"bright": round(bright_ratio,4),
                                   "dark":   round(dark_ratio,4)})

    sg         = cv2.cvtColor(to_np(crop(SUBTITLE_REGION)), cv2.COLOR_RGB2GRAY)
    sub_bright = int((sg > 160).sum())
    sub_mean   = float(sg.mean())
    if sub_bright > 200 and sub_mean < 130:
        return ScreenMode("DIALOGUE",
                          details={"bright_px": sub_bright,
                                   "mean":      round(sub_mean,1)})

    hh = cv2.cvtColor(to_np(crop(ENEMY_HP_REGION)), cv2.COLOR_RGB2HSV)
    rm = cv2.inRange(hh,
        np.array([0,  140, 100]),
        np.array([12, 255, 255])
    )
    if int(rm.sum() // 255) > 60:
        return ScreenMode("COMBAT",
                          details={"red_px": int(rm.sum()//255)})

    return ScreenMode("EXPLORATION")

# ── Minimap navigation ────────────────────────────────────────────────────────

def read_minimap() -> dict:
    mm  = to_np(crop(MINIMAP_REGION))
    hsv = cv2.cvtColor(mm, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,
        np.array([15, 100, 120]),
        np.array([45, 255, 255])
    )
    h, w   = mask.shape
    cx, cy = w // 2, h // 2
    cv2.circle(mask, (cx, cy), 20, 0, -1)
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), min(cx, cy) - 10, 255, -1)
    mask = cv2.bitwise_and(mask, inner)

    ys, xs = np.where(mask > 0)
    if len(xs) < 4:
        return {"found": False, "direction": "unknown",
                "turn": "none", "yellow_px": 0}

    mx, my = float(xs.mean()), float(ys.mean())
    dx, dy = mx - cx, my - cy
    angle  = float(np.degrees(np.arctan2(dx, -dy)) % 360)

    if   angle < 25 or angle > 335: direction, turn = "forward",       "none"
    elif angle < 70:                 direction, turn = "forward-right",  "right"
    elif angle < 120:                direction, turn = "right",          "right"
    elif angle < 160:                direction, turn = "back-right",     "right"
    elif angle < 200:                direction, turn = "behind",         "right"
    elif angle < 250:                direction, turn = "back-left",      "left"
    elif angle < 290:                direction, turn = "left",           "left"
    else:                            direction, turn = "forward-left",   "left"

    dist = float(np.sqrt(dx**2 + dy**2))
    return {
        "found":     True,
        "angle":     round(angle, 1),
        "direction": direction,
        "turn":      turn,
        "on_edge":   dist > (min(cx, cy) * 0.75),
        "yellow_px": int(len(xs)),
    }

# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr(region: tuple) -> str:
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
    "confirm":  "e",
    "continue": "space",
    "skip":     "space",
    "up":       "Up",
    "down":     "Down",
    "forward":  "w",
    "back":     "s",
    "left":     "a",
    "right":    "d",
    "sprint":   "shift",
    "jump":     "space",
    "menu":     "Escape",
}

def _xdo(*args):
    subprocess.run(["xdotool", *args], check=False)

def press(key: str):
    k = KEYMAP.get(key.lower(), key)
    if W3_WINDOW_ID:
        _xdo("windowactivate", "--sync", W3_WINDOW_ID)
        _xdo("key", "--window", W3_WINDOW_ID, "--clearmodifiers", k)
    else:
        _xdo("key", k)
    time.sleep(0.3)

def hold(key: str, duration: float):
    k        = KEYMAP.get(key.lower(), key)
    duration = float(np.clip(duration, 0.1, 5.0))
    if W3_WINDOW_ID:
        _xdo("windowactivate", "--sync", W3_WINDOW_ID)
        _xdo("keydown", "--window", W3_WINDOW_ID, "--clearmodifiers", k)
        time.sleep(duration)
        _xdo("keyup",   "--window", W3_WINDOW_ID, k)
    else:
        _xdo("keydown", k)
        time.sleep(duration)
        _xdo("keyup",   k)
    time.sleep(0.1)

def hold_two(key1: str, key2: str, duration: float):
    k1 = KEYMAP.get(key1.lower(), key1)
    k2 = KEYMAP.get(key2.lower(), key2)
    duration = float(np.clip(duration, 0.1, 5.0))
    if W3_WINDOW_ID:
        _xdo("windowactivate", "--sync", W3_WINDOW_ID)
        _xdo("keydown", "--window", W3_WINDOW_ID, "--clearmodifiers", k1)
        _xdo("keydown", "--window", W3_WINDOW_ID, k2)
        time.sleep(duration)
        _xdo("keyup",   "--window", W3_WINDOW_ID, k2)
        _xdo("keyup",   "--window", W3_WINDOW_ID, k1)
    else:
        _xdo("keydown", k1)
        _xdo("keydown", k2)
        time.sleep(duration)
        _xdo("keyup",   k2)
        _xdo("keyup",   k1)
    time.sleep(0.1)

# ── LM Studio inference ───────────────────────────────────────────────────────

def ask_llm(
    prompt:      str,
    system:      str,
    image:       Optional[Image.Image] = None,
    temperature: float = 0.3,
    max_tokens:  int   = 256,
) -> str:
    """
    Send a prompt to Qwen2-VL via LM Studio.
    Optionally includes an image for vision tasks.

    The model receives:
      - system prompt describing its role and current mode
      - optional screenshot as base64 image
      - text prompt with screen context and question

    Returns the model's response as a string.
    """
    content = []

    if image is not None:
        b64 = to_b64(image)
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    content.append({"type": "text", "text": prompt})

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
        print(f"  ⚠  LM Studio error: {e}")
        return "wait"

# ── Decision parsers ──────────────────────────────────────────────────────────

def parse_action(response: str) -> dict:
    """
    Parse the model's text response into a structured action.

    The model is prompted to respond in a simple format:
      ACTION: <action_name>
      KEY: <key_name>
      DURATION: <seconds>
      REASON: <brief explanation>

    Falls back to WAIT if parsing fails.
    """
    lines  = {
        line.split(":")[0].strip().upper(): ":".join(line.split(":")[1:]).strip()
        for line in response.strip().splitlines()
        if ":" in line
    }

    action   = lines.get("ACTION",   "wait").lower().strip()
    key      = lines.get("KEY",      "space").lower().strip()
    duration = float(lines.get("DURATION", "0.5").split()[0])
    reason   = lines.get("REASON",   "")

    return {
        "action":   action,
        "key":      key,
        "duration": duration,
        "reason":   reason,
        "raw":      response,
    }

def execute_action(parsed: dict) -> str:
    """Execute a parsed action. Returns result string."""
    action   = parsed["action"]
    key      = parsed["key"]
    duration = float(np.clip(parsed["duration"], 0.1, 5.0))
    reason   = parsed.get("reason", "")

    if action == "press":
        press(key)
        print(f"  ⌨  {key}  —  {reason}")
        return f"Pressed {key}."

    elif action == "move":
        direction = key  # forward/back/left/right
        sprint    = "sprint" in parsed.get("raw", "").lower()
        print(f"  🚶 {direction} {duration}s"
              f"{'  sprint' if sprint else ''}  —  {reason}")
        if sprint and direction == "forward":
            hold_two("forward", "sprint", duration)
        else:
            hold(direction, duration)
        return f"Moved {direction} for {duration}s."

    elif action == "wait":
        secs = float(np.clip(duration, 0.5, 12.0))
        print(f"  ⏳ {secs}s  —  {reason}")
        time.sleep(secs)
        return f"Waited {secs}s."

    else:
        print(f"  ⚠  Unknown action: {action}, waiting")
        time.sleep(1.0)
        return "Unknown action, waited."

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_BASE = """
You are UMA, an experimental AI agent playing The Witcher 3: Wild Hunt.
You follow the main story quest line.
You perceive the world only through the game screen — no engine access.
You embody Geralt of Rivia: pragmatic, world-weary, morally serious.

You must respond in EXACTLY this format, nothing else:
ACTION: <press|move|wait>
KEY: <key or direction>
DURATION: <seconds, e.g. 0.5>
REASON: <one sentence>

Valid keys for press: e, space, up, down, confirm, continue, skip, menu
Valid directions for move: forward, back, left, right
""".strip()

MODE_CONTEXT = {
    "LOADING":  "The game is loading. Wait for it to finish.",
    "CUTSCENE": "A cutscene is playing. Do not interrupt it.",
    "MENU":     "A menu is open. Navigate with up/down, confirm with e.",
    "DIALOGUE": (
        "An NPC is speaking. No choices visible.\n"
        "Press space/continue to advance dialogue.\n"
        "Wait if the line is still animating."
    ),
    "CHOICES": (
        "Dialogue choices are on screen.\n"
        "Read the choices from the screen text provided.\n"
        "Navigate with up/down keys, confirm with e.\n"
        "Choose as Geralt would: pragmatic, consistent with his character.\n"
        "Think about Witcher 3 lore before deciding."
    ),
    "COMBAT": (
        "Combat is active.\n"
        "Press e for fast attack, space to dodge.\n"
        "Pattern: attack, attack, dodge. Repeat."
    ),
    "EXPLORATION": (
        "Geralt is free to move. Follow the quest objective.\n"
        "The minimap rotates with Geralt — top is his forward direction.\n"
        "Use the minimap direction data provided to navigate.\n"
        "Short left/right moves (0.3-0.5s) rotate Geralt.\n"
        "Forward moves walk toward the objective.\n"
        "If unchanged for 3+ ticks: back up and try a new angle."
    ),
}

def system_for(mode: str) -> str:
    context = MODE_CONTEXT.get(mode, "Observe the screen and act.")
    return f"{SYSTEM_BASE}\n\nCurrent game state: {mode}\n{context}"

# ── Context builder ───────────────────────────────────────────────────────────

def build_prompt(mode: ScreenMode, same_count: int) -> tuple:
    """
    Build the text prompt for this tick.
    Returns (prompt_text, image_to_send).

    Image selection:
    - EXPLORATION: send minimap crop (focused, faster inference)
    - DIALOGUE/CHOICES: send full screen (model reads text from image)
    - COMBAT: send full screen (model sees enemy positions)
    - Other: send full screen
    """
    lines = [
        f"Game state: {mode.mode}",
        f"Screen unchanged for: {same_count} tick(s)",
    ]

    image_to_send = _screen   # default: full screen

    if mode.mode == "DIALOGUE":
        subtitle = read_subtitle()
        lines.append(f'Subtitle text: "{subtitle}"')
        lines.append("The NPC is speaking. What do you do?")

    elif mode.mode == "CHOICES":
        choices = read_choices()
        lines.append(f'Choices on screen: "{choices}"')
        lines.append(
            "Dialogue choices are visible. Think about Witcher 3 lore "
            "and Geralt's character. What do you choose?"
        )

    elif mode.mode == "EXPLORATION":
        nav      = read_minimap()
        quest    = read_quest()
        interact = read_interact()

        lines.append(f"Minimap quest marker: {nav}")
        if quest:
            lines.append(f'Active quest: "{quest}"')
        if interact:
            lines.append(f'Interaction prompt visible: "{interact}"')
        lines.append(
            "Navigate toward the quest objective using the minimap data. "
            "What do you do?"
        )
        # For exploration, minimap crop is more useful than full screen
        image_to_send = crop(MINIMAP_REGION)

    elif mode.mode == "COMBAT":
        lines.append(
            "Enemy health bars detected. Combat is active. "
            "What do you do?"
        )

    elif mode.mode in ("LOADING", "CUTSCENE"):
        lines.append("Wait for this to finish.")
        image_to_send = None   # no need to send image for these

    elif mode.mode == "MENU":
        lines.append("A menu is open. What do you do?")

    prompt = "\n".join(lines)
    return prompt, image_to_send

# ── Main loop ─────────────────────────────────────────────────────────────────

def find_w3_window() -> str:
    result = subprocess.run(
        ["xdotool", "search", "--name", "Witcher"],
        capture_output=True, text=True
    )
    ids = result.stdout.strip().split("\n")
    if ids and ids[0].strip():
        wid = ids[0].strip()
        print(f"Found W3 window: {wid}")
        return wid
    return ""

def run():
    global W3_WINDOW_ID

    if not W3_WINDOW_ID:
        W3_WINDOW_ID = find_w3_window()
        if not W3_WINDOW_ID:
            print("Warning: W3 window not found. Input goes to focused window.\n")

    print(f"Starting in {STARTUP_DELAY}s — switch to Witcher 3...\n")
    for i in range(STARTUP_DELAY, 0, -1):
        print(f"  {i}", end="\r", flush=True)
        time.sleep(1)
    print("Running.  Ctrl-C to stop.\n")

    last_ctx   = ""
    same_count = 0
    tick       = 0

    while True:
        tick += 1
        t_start = time.time()

        # 1. Capture
        grab_screen()

        # 2. Detect mode
        mode = detect_mode()

        # 3. Build prompt + select image
        prompt, image = build_prompt(mode, same_count)

        # 4. Track stability
        if prompt != last_ctx:
            last_ctx   = prompt
            same_count = 0
        else:
            same_count += 1

        print(f"\n[{tick}]  {mode.mode}  "
              f"{'NEW' if same_count == 0 else f'same×{same_count}'}")

        # 5. Ask local model
        temperature = (DIALOGUE_TEMPERATURE
                       if mode.mode in ("CHOICES", "DIALOGUE")
                       else NAV_TEMPERATURE)

        max_tokens  = (MAX_TOKENS_DIALOGUE
                       if mode.mode in ("CHOICES", "DIALOGUE")
                       else MAX_TOKENS_NAV)

        t_llm = time.time()
        response = ask_llm(
            prompt      = prompt,
            system      = system_for(mode.mode),
            image       = image,
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        llm_ms = int((time.time() - t_llm) * 1000)

        print(f"  🧠 [{llm_ms}ms] {response}")

        # 6. Parse and execute
        parsed = parse_action(response)
        execute_action(parsed)

        # 7. Timing report
        total_ms = int((time.time() - t_start) * 1000)
        print(f"  ⏱  tick {total_ms}ms  (llm {llm_ms}ms)")

        time.sleep(max(0, TICK_INTERVAL - (time.time() - t_start)))

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\nUMA stopped.")
