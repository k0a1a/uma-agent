#!/usr/bin/env python3
"""
UMA — calibrate.py
==================
HUD region calibration tool for the UMA Witcher 3 agent.

Platform:  Linux, X11, Openbox, Nvidia, single 1080p display

Usage:
    export DISPLAY=:0
    python3 calibrate.py

Every 8 seconds:
  - Takes a screenshot via mss
  - Detects current game mode
  - Prints all raw sensor values
  - Runs OCR on all text regions
  - Saves crop_*.png for visual inspection

Open crop_minimap.png, crop_subtitle.png etc. in an image viewer
to verify each region is hitting the right part of the screen.
If a region shows game world instead of HUD, adjust the coordinates
in both calibrate.py and uma0.py.

Test each game state between readings:
  Walking (quest active)  → EXPLORATION + minimap direction
  Near NPC               → EXPLORATION + interact prompt
  In dialogue            → DIALOGUE + subtitle text
  Choices visible        → CHOICES + choice text
  Inventory / map open   → MENU
  Cutscene playing       → CUTSCENE
  In combat              → COMBAT
"""

import sys
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import mss
import cv2
import easyocr
from PIL import Image, ImageEnhance

# ── Regions — (left, top, width, height) at 1920x1080 ────────────────────────
# These must match the values in uma0.py.
# Adjust if crop_*.png images show game world instead of HUD elements.

MINIMAP_REGION  = (1666,  65, 207, 200)   # top-right golden circle
QUEST_REGION    = (1195, 218, 265, 130)   # quest title + objectives
SUBTITLE_REGION = ( 420, 648, 610,  78)   # subtitle text
CHOICE_REGION   = ( 855, 496, 250, 130)   # numbered choices
INTERACT_REGION = (  55, 350, 230, 120)   # [E] interact prompt
ENEMY_HP_REGION = ( 650,  44, 580,  24)   # enemy health bars

GAME_W = 1920
GAME_H = 1080

# ── Screen capture ────────────────────────────────────────────────────────────

sct     = mss.MSS()
_screen: Optional[Image.Image] = None

def grab_screen() -> Image.Image:
    global _screen
    raw     = sct.grab(sct.monitors[1])
    _screen = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    return _screen

def crop(region) -> Image.Image:
    if _screen is None:
        grab_screen()
    l, t, w, h = region
    return _screen.crop((l, t, l + w, t + h))

def to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

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
    mm_c = sum(px(1769, 165)) / 3
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

# ── Minimap direction ─────────────────────────────────────────────────────────

def read_minimap() -> dict:
    mm  = to_np(crop(MINIMAP_REGION))
    hsv = cv2.cvtColor(mm, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv,
        np.array([15, 100, 120]),
        np.array([45, 255, 255])
    )

    h, w   = mask.shape
    cx, cy = w // 2, h // 2

    # blank centre dot
    cv2.circle(mask, (cx, cy), 20, 0, -1)

    # blank outer border ring
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), min(cx, cy) - 10, 255, -1)
    mask = cv2.bitwise_and(mask, inner)

    ys, xs = np.where(mask > 0)

    if len(xs) < 4:
        return {
            "found":     False,
            "direction": "unknown",
            "turn":      "none",
            "yellow_px": int(len(xs)),
        }

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

print("Loading OCR model...")
ocr = easyocr.Reader(['en'], gpu=True)
print("OCR ready.\n")

def _ocr(region: tuple) -> str:
    img  = crop(region)
    img  = ImageEnhance.Contrast(img.convert("L")).enhance(2.5).convert("RGB")
    text = ocr.readtext(to_np(img), detail=0, paragraph=True)
    return " ".join(text).strip()

# ── Sensor dump ───────────────────────────────────────────────────────────────

def dump_sensors():
    centre = px(960, 540)
    top_b  = sum(px(960,  40)) / 3
    bot_b  = sum(px(960, 1040)) / 3
    mm_c   = sum(px(1355, 128)) / 3
    pts    = [(480,300),(480,540),(480,780),(960,300),(960,540),(960,780)]
    brts   = [sum(px(x, y)) / 3 for x, y in pts]

    cg = cv2.cvtColor(to_np(crop(CHOICE_REGION)),   cv2.COLOR_RGB2GRAY)
    sg = cv2.cvtColor(to_np(crop(SUBTITLE_REGION)), cv2.COLOR_RGB2GRAY)
    hh = cv2.cvtColor(to_np(crop(ENEMY_HP_REGION)), cv2.COLOR_RGB2HSV)
    rm = cv2.inRange(hh, np.array([0,140,100]), np.array([12,255,255]))

    mm   = to_np(crop(MINIMAP_REGION))
    mhsv = cv2.cvtColor(mm, cv2.COLOR_RGB2HSV)
    ym   = cv2.inRange(mhsv,
               np.array([15, 100, 120]),
               np.array([45, 255, 255]))
    mh, mw = ym.shape
    cv2.circle(ym, (mw//2, mh//2), 20, 0, -1)

    print("  ── Sensors ──────────────────────────────────────────────")
    print(f"  centre brightness:      {sum(centre)/3:.1f}   (LOADING if < 20)")
    print(f"  top / bot letterbox:    {top_b:.1f} / {bot_b:.1f}  (CUTSCENE if both < 15)")
    print(f"  menu var / mean:        {np.var(brts):.1f} / {np.mean(brts):.1f}  (MENU if var<150 and mean 65-185)")
    print(f"  minimap centre px:      {mm_c:.1f}   (MENU extra: absent if < 55)")
    print(f"  choice bright160:       {(cg>160).sum()/cg.size:.4f}  (CHOICES if > 0.012)")
    print(f"  choice dark ratio:      {((cg>15)&(cg<90)).sum()/cg.size:.4f}  (CHOICES if > 0.08)")
    print(f"  subtitle bright160 px:  {(sg>160).sum()}   (DIALOGUE if > 200)")
    print(f"  subtitle mean:          {sg.mean():.1f}  (DIALOGUE if < 130)")
    print(f"  enemy red pixels:       {rm.sum()//255}    (COMBAT if > 60)")
    print(f"  minimap yellow pixels:  {(ym>0).sum()}   (nav marker — want > 4)")

# ── Save crops ────────────────────────────────────────────────────────────────

def save_crops():
    regions = {
        "minimap":  MINIMAP_REGION,
        "quest":    QUEST_REGION,
        "subtitle": SUBTITLE_REGION,
        "choices":  CHOICE_REGION,
        "interact": INTERACT_REGION,
        "enemy_hp": ENEMY_HP_REGION,
    }
    for name, region in regions.items():
        crop(region).save(f"crop_{name}.png")
    print("  Saved: " + "  ".join(f"crop_{n}.png" for n in regions))

# ── Resolution check ──────────────────────────────────────────────────────────

def check_resolution():
    raw = sct.grab(sct.monitors[1])
    w, h = raw.size
    print(f"Screenshot size:  {w} × {h}")
    if w != GAME_W or h != GAME_H:
        print(f"⚠  Expected {GAME_W}×{GAME_H} — coordinates may be wrong")
        print(f"   Adjust GAME_W/GAME_H or check display/scaling settings")
    else:
        print(f"✓  Resolution correct: {GAME_W}×{GAME_H}")
    print()

# ── Minimap HSV visualiser ────────────────────────────────────────────────────

def save_minimap_debug():
    """
    Save additional minimap debug images:
      crop_minimap_hsv_mask.png  — yellow pixels detected by HSV filter
      crop_minimap_annotated.png — original with detected marker circled
    """
    mm   = to_np(crop(MINIMAP_REGION))
    hsv  = cv2.cvtColor(mm, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv,
               np.array([15, 100, 120]),
               np.array([45, 255, 255]))

    h, w   = mask.shape
    cx, cy = w // 2, h // 2
    cv2.circle(mask, (cx, cy), 20, 0, -1)
    inner = np.zeros_like(mask)
    cv2.circle(inner, (cx, cy), min(cx, cy) - 10, 255, -1)
    clean_mask = cv2.bitwise_and(mask, inner)

    # save raw mask
    Image.fromarray(clean_mask).save("crop_minimap_hsv_mask.png")

    # save annotated original
    annotated = mm.copy()
    ys, xs    = np.where(clean_mask > 0)
    if len(xs) > 0:
        mx, my = int(xs.mean()), int(ys.mean())
        cv2.circle(annotated, (mx, my), 8, (255, 0, 0), 2)   # red circle on marker
        cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)  # green dot at centre
    Image.fromarray(annotated).save("crop_minimap_annotated.png")
    print("  Saved: crop_minimap_hsv_mask.png  crop_minimap_annotated.png")

# ── Main loop ─────────────────────────────────────────────────────────────────

check_resolution()

print("Switch to Witcher 3 now.  First reading in 5s.\n")
print("Test each state between readings:")
print("  Walking (quest active)  → EXPLORATION + minimap direction")
print("  Near NPC               → EXPLORATION + interact prompt")
print("  In dialogue            → DIALOGUE + subtitle text")
print("  Choices visible        → CHOICES + choice text")
print("  Inventory open         → MENU")
print("  Cutscene               → CUTSCENE")
print("  In combat              → COMBAT\n")

time.sleep(5)
tick = 0

while True:
    tick += 1
    print(f"\n{'='*62}")
    print(f"  Tick {tick}")
    print(f"{'='*62}")

    grab_screen()

    # Mode
    mode = detect_mode()
    print(f"\n  MODE:      {mode.mode}  |  {mode.details}")

    # Sensors
    dump_sensors()

    # Minimap
    nav = read_minimap()
    print(f"\n  MINIMAP:   {nav}")

    # OCR
    print(f"\n  quest:    \"{_ocr(QUEST_REGION)}\"")
    print(f"  subtitle: \"{_ocr(SUBTITLE_REGION)}\"")
    print(f"  choices:  \"{_ocr(CHOICE_REGION)}\"")
    print(f"  interact: \"{_ocr(INTERACT_REGION)}\"")

    # Crops
    save_crops()
    save_minimap_debug()

    print(f"\n  Next reading in 8s...  Ctrl-C to stop.")
    time.sleep(8)
