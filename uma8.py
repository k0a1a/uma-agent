#!/usr/bin/env python3
"""
UMA — Unknown Morphic Agent  v8.0
===================================
Plays the Witcher 3 through vision only — no engine hooks, no game data. It reads
the screen, infers a heading from the minimap, and drives a virtual controller.

Navigation (NAV_MODE, default "pixel"):
  • Gold destination marker — the largest COMPACT amber blob (rim & arc rejected);
    a reliable far-field bearing to the goal, used as startup/fallback.
  • White dotted trail — the minimap crop is upscaled and dots are detected by
    WHITENESS (neutral + bright), so yellow icons / gold arc / green signposts are
    ignored. The trail ROUTES around obstacles; forward direction is chosen by
    rejecting the 'behind' cluster (where we came from), which keeps hard turns.
  • Priority: follow the near trail; widen to the whole trail near the destination
    (where the gold diamond is replaced by the target icon); gold bearing as backstop.
  • "vlm" mode (Qwen via LM Studio) is kept for comparison but reads this small,
    cluttered minimap unreliably — see project history.

Obstacle avoidance (NEW in v8): the minimap routes around buildings but is BLIND
to small upright obstacles — poles, posts, fence rails, trunks, wall corners. A
VIEWPORT layer reads the near-ground main image, scores per-sector obstacle energy
(vertical-edge density), and DEFLECTS the trail heading toward the clearest gap
before Geralt wedges himself. It's silent when the path ahead is clear; SSIM stuck
recovery remains the backstop for collisions vision can't see.

Steering: LEFT-STICK DIRECTIONAL — the stick vector points where we want to go and
the camera auto-trails (no rate-steering feedback loop). Forward speed falls off
with cos(heading) so we turn-then-go instead of plowing through bends. The viewport
and minimap share this camera frame (0° = forward = image centre column, + = right),
so a "blocked at +X°" reading is directly comparable to the trail heading.

Transport:
  - screen  : WebSocket, request-driven (freshest frame, capture off the path)
  - control : WebSocket, fire-and-forget set-and-hold
  - OCR/VLM : HTTP via a keep-alive Session (request-response by nature)

Architecture:
  b365.local          — W3 + controller_server.py (5002) + screen_server.py (5003)
  beacon.x.k0a1a.net  — LM Studio Qwen2.5-VL (1234) + ocr_server.py (5001)
  laptop              — this script

Install:
    pip install opencv-python numpy pillow requests openai scikit-image websocket-client

Usage:
    python3 uma7.py
"""

import os, sys, time, base64, io, math, json, random, threading
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

MINIMAP_UPSCALE = 2           # upscale the minimap before dot detection (tiny dots)
DOT_EXCLUDE_R   = 25          # blank Geralt's arrow at centre (original-scale px)
DOT_BORDER_PAD  = 8           # ignore bright rim of the minimap
DOT_AREA_MIN    = 1           # areas are in ORIGINAL-scale px (scaled internally)
DOT_AREA_MAX    = 18
DOT_MIN_EXTENT  = 0.45        # area / bbox — round blobs pass, thin arcs fail
DOT_MAX_ASPECT  = 3.0         # bbox aspect — rejects elongated ring fragments
DOT_GUARD_AREA  = 4           # only apply extent/aspect guard at/above this area
# Path dots are WHITE (R≈G≈B, all bright). Whiteness rejects yellow icons (notice
# board, gold arc/diamond) and green signposts — only neutral bright blobs pass.
WHITE_MIN       = 150         # all three channels brighter than this
WHITE_SPREAD    = 45          # max(R,G,B) - min(R,G,B) below this = neutral (white/grey)

# ── Gold destination marker (pixel) ───────────────────────────────────────────
# The amber quest marker (far-field) is big and points at the destination. Taken
# as the largest COMPACT gold blob, so the thin curved quest-area arc and the rim
# are rejected. Used only as a fallback when no white trail is visible.
GOLD_HSV_LO     = (15, 120, 150)   # OpenCV HSV low  (hue 0-180)
GOLD_HSV_HI     = (40, 255, 255)   # OpenCV HSV high
GOLD_RIM_FRAC   = 0.88        # ignore gold outside this fraction of radius (the rim)
GOLD_MIN_PIX    = 25          # min blob area to call it a marker
GOLD_MIN_EXTENT = 0.35        # area / bbox — compact blob; rejects the thin arc
GOLD_MAX_BBOX   = 45          # reject gold spans wider than this (arc / rim fragments)
GOLD_SWITCH_RATIO = 1.5       # temporal coherence (Fix 2): switch off the tracked gold
                              # blob only if another is this× bigger — else hold, so the
                              # destination diamond and a quest icon stop flip-flopping
GOLD_TRUST_MIN_FRAC = 0.15    # within this × minimap-radius the marker is ~on top of us
                              # and its BEARING is noise (wild swings / 180° flips), so we
                              # stop steering by it and let the latch + arrival take over

# ── Steering law — LEFT-STICK DIRECTIONAL + NEAR-FIELD HEADING ────────────────
# Heading is the direction of the IMMEDIATE path dots around Geralt, resolved
# robustly: only dots within R_NEAR count, the through-Geralt forward/backward
# ambiguity is broken by momentum (prefer the cluster near the previous heading),
# and we only trust a frame whose near dots actually agree (concentration >= CONC_MIN).

R_NEAR          = 75      # px; only dots this close to centre drive the heading
CONC_MIN        = 0.50    # min angular concentration (0=scattered, 1=aligned) to trust
MIN_DOTS_FOR_HEADING = 3  # min trail dots to TRUST a trail read (additive to CONC_MIN:
                          # need enough dots AND concentration). Fewer → defer to the
                          # gold bearing / latch instead of committing to a thin trail.
                          # Mainly bites near the destination where dots go sparse.
FWD_CONE_DEG    = 90      # (legacy) near dots within this of the previous heading
BEHIND_CONE     = 70      # reject trail dots within this of 'behind' (the came-from
                          # direction) — keeps forward + around-obstacle turns, drops origin
HEADING_EMA     = 0.40    # vector smoothing on heading (sin/cos, wrap-safe)
SPEED_FLOOR     = 0.50    # keep moving while turning (never pirouette in place)
DEBUG_EVERY     = 3       # write minimap debug PNGs every N ticks (0 = never)
INTERACT_EVERY  = 3       # run the interact-prompt OCR every N ticks (latency)
DIALOGUE_ENABLE = os.environ.get("DIALOGUE", "0") not in ("0", "false", "")
                          # OFF by default for now: suppresses interact-prompt 'A' presses
                          # and CHOICES dialogue selection so navigation testing can't be
                          # derailed by spurious key presses on garbage OCR. DIALOGUE=1 re-enables.
DISTANCE_ENABLE = os.environ.get("DISTANCE", "1") not in ("0", "false", "")
DISTANCE_EVERY  = 5       # OCR the objective-distance counter every N ticks (it changes slowly,
                          # and it's read-only, so this stays cheap and off the per-tick path)
DISTANCE_MAX        = 5000  # m; reject reads above this as OCR garbage
DISTANCE_MAX_JUMP   = 60    # m; a one-off read jumping more than this from last-good is a glitch...
DISTANCE_CONFIRM_TOL = 15   # ...unless a consecutive read confirms it (a real objective change)
DISTANCE_STALE      = 4     # drop last-good after this many failed reads (then return None)

# ── Navigation source ─────────────────────────────────────────────────────────
# "pixel": gold destination marker (primary) + decluttered dotted trail (fallback).
#          Fast, deterministic, no VLM/VPN in the hot path. RECOMMENDED.
# "vlm"  : a background thread asks Qwen which way to go. Kept for comparison, but
#          small local VLMs read this minimap unreliably (see project history).
NAV_MODE     = os.environ.get("NAV_MODE", "pixel")
VLM_PERIOD   = 1.5        # seconds between VLM minimap reads (call may take longer)
VLM_UPSCALE  = 2          # upscale the minimap crop before sending (dots are tiny).
                          # higher = the VLM sees dots better but MORE vision tokens =
                          # slower inference + bigger upload. 1 = native 213px (fastest).
NAV_EMA      = 0.5        # vector smoothing on the committed VLM heading
TURN_TAU     = 0.8        # s; the committed heading decays toward 0 as it ages, so a
                          # stale angle becomes "turn toward the path, then go straight"
                          # instead of circling. A fresh VLM read resets the decay.

# ── Stuck detection ───────────────────────────────────────────────────────────

SSIM_THRESHOLD = 0.97
STUCK_TICKS    = 3
EDGE_BIAS_DEG  = 35       # after a stuck, hold this lateral bias on the heading...
EDGE_BIAS_TICKS = 8       # ...for this many ticks, so Geralt arcs AROUND the obstacle
                          # instead of resuming the heading that drove him into it
JUMP_BUTTON    = "b"      # controller button for Jump. FIRST stuck-recovery action: a
                          # rock / wood pile is often clearable with a hop. VERIFY this
                          # matches the in-game "Jump" prompt; change if it's mapped else.

# ── Gold-bearing latch (near-destination orientation) ─────────────────────────
# Close to the target the gold diamond is replaced by the destination icon and the
# trail dots get sparse/occluded. Rather than chase unreliable dots, keep steering
# toward the LAST gold bearing, decaying it toward straight-ahead so we turn-then-go
# (a held relative bearing would circle). Refreshed whenever the marker reappears.
GOLD_LATCH_TICKS = 30     # keep the latch alive this many ticks after the marker vanishes
GOLD_LATCH_TAU   = 6      # ticks; latched bearing decays to straight over ~this scale

# ── Combat detection ──────────────────────────────────────────────────────────
# An enemy HP bar is a wide, thin, CONTIGUOUS horizontal red run. Scenery red
# (roofs, banners, sky) is scattered or broken into short spans, so we test for
# a contiguous span rather than a raw pixel count.
COMBAT_MIN_SPAN = 120         # px; min contiguous horizontal red run = HP bar
COMBAT_CONFIRM  = 2           # consecutive raw-COMBAT frames before acting

# ── HUD regions (left, top, width, height) — 1920×1080, Next Gen W3, b365 ─────

#MINIMAP_REGION = (1660,  36, 213, 215)    # default HUD minimap size
MINIMAP_REGION  = (1608,  50, 258, 258)    # enlarged HUD minimap (HUD prefs)
QUEST_REGION    = (1220, 225, 340, 100)
SUBTITLE_REGION = ( 450, 655, 570,  55)
CHOICE_REGION   = ( 865, 500, 360, 125)
INTERACT_REGION = (  55, 350, 230, 120)
ENEMY_HP_REGION = ( 650,  44, 580,  24)
# Objective-distance (footprints) counter, below-left of the minimap. Shows a number
# in steps, or the word "NEARBY" when under ~10 steps from the objective. A tight box
# around it, clear of the 'CLEAR' weather box and sun/moon dial above.
DISTANCE_REGION = (1500, 204, 100,  30)    # verified @1920×1080

# ── Auto-scale minimap-pixel constants to the configured minimap size ──────────
# The dot/gold geometry above was tuned at a 213-px minimap. Scaling by the actual
# minimap width keeps it correct when you resize the HUD minimap — change
# MINIMAP_REGION and everything follows. Lengths scale linearly; pixel-area
# thresholds scale with the square. (Whiteness/HSV/fraction params are size-free.)
MINIMAP_REF_PX = 213
_mm_s = MINIMAP_REGION[2] / MINIMAP_REF_PX
DOT_EXCLUDE_R  = max(1, round(DOT_EXCLUDE_R  * _mm_s))
DOT_BORDER_PAD = max(1, round(DOT_BORDER_PAD * _mm_s))
R_NEAR         = max(1, round(R_NEAR         * _mm_s))
DOT_AREA_MIN   = max(1, round(DOT_AREA_MIN   * _mm_s * _mm_s))
DOT_AREA_MAX   = max(2, round(DOT_AREA_MAX   * _mm_s * _mm_s))
GOLD_MIN_PIX   = max(4, round(GOLD_MIN_PIX   * _mm_s * _mm_s))
GOLD_MAX_BBOX  = max(4, round(GOLD_MAX_BBOX  * _mm_s))

# ── Viewport obstacle avoidance (VFH over the near-ground main image) ──────────
# The minimap is blind to small upright obstacles (poles, posts, fence rails,
# trunks, wall corners, rocks); before v8 those only surfaced as an SSIM stuck
# AFTER Geralt wedged himself. This layer reads the main image and deflects the
# trail heading toward the clearest gap BEFORE the collision. Set VIEWPORT_AVOID=0
# to disable and fall back to v7 behaviour (trail only + stuck recovery).
VIEWPORT_AVOID = os.environ.get("VIEWPORT_AVOID", "1") not in ("0", "false", "")

# Near-ground band of the main image — CALIBRATE EMPIRICALLY against
# viewport_raw.png / viewport_annotated.png, exactly as the HUD regions were. It
# must sit BELOW the horizon (distant terrain is not an immediate obstacle), ABOVE
# the bottom edge, clear of the right-side HUD (minimap/quest), and ideally below
# the SUBTITLE band (~y655): ambient banter subtitles are bright text that reads as
# a dead-centre blob. This starting guess SPANS the subtitle band — if false
# "blocked ahead" fires during walking banter, lower the band or shrink its height.
VIEWPORT_REGION  = (360, 520, 1200, 300)   # (l, t, w, h) @1920×1080 — STARTING GUESS

# Geralt's body sits at lower-CENTRE and would read as a permanent obstacle. But an
# obstacle a few metres AHEAD also projects to centre — just HIGHER up the frame
# (further = higher). So we blank only the LOWER centre (his body/feet) and keep the
# upper-centre rows live for head-on detection. The W3 camera is slightly off-centre
# — tune the centre. Blanking the full column would make UMA blind dead-ahead.
BODY_MASK_CENTRE = 0.50     # fraction across the region width where his body sits
BODY_MASK_WIDTH  = 0.16     # fraction of region width to blank
BODY_MASK_BOTTOM = 0.55     # blank the centre only BELOW this row fraction (0=top)

VIEW_FOV_DEG     = 90.0     # angular span the region WIDTH subtends (column→angle).
                            # Widened from 70 → 90: at 70 ~40% of frames fell beyond
                            # ±35° and printed 'wide' (obstacle check skipped) right
                            # when a hard turn needed guidance. Perspective makes the
                            # map approximate; this is the fudge knob.
N_SECTORS        = 24       # angular histogram resolution across VIEW_FOV_DEG
NEAR_ROW_GAMMA   = 1.5      # weight lower (nearer) rows ∝ (row_frac)**this
BLOCK_K          = 1.8      # a sector blocks above median·this (ADAPTIVE: a busy
                            # frame raises its own bar → only RELATIVELY worst block)
BLOCK_FLOOR      = 0.06     # ...but never below this absolute (normalised 0..1), so
                            # a near-empty frame isn't all "blocked" off noise
AVOID_CONE_DEG   = 55.0     # most we'll deflect to dodge (never steer ~backwards)
SLOW_ON_BLOCK    = 0.45     # speed × this while squeezing past / boxed in

# ── Live-lock detection + commit-to-gap recovery (Fix 1) ──────────────────────
# SSIM stuck-recovery only catches a FROZEN frame. A live-lock — Geralt micro-turning
# at an obstacle, nibbling at a gap, frames changing but going nowhere (the 30s barn
# oscillation) — slips past it. We count consecutive BLOCKED ticks; once that exceeds
# LIVELOCK_TICKS we stop nibbling, pick the WIDEST clear gap in the viewport, and
# COMMIT to it — drive straight through at full speed for COMMIT_TICKS, ignoring the
# (oscillating) trail/gold input. If even the widest gap is too narrow it's a true
# box-in and we leave it to SSIM back-off.
LIVELOCK_TICKS   = 10       # consecutive blocked ticks before declaring a live-lock
COMMIT_TICKS     = 6        # drive committed through the chosen gap for this many ticks
COMMIT_SPEED     = 0.95     # full-ish speed through the gap (overrides SLOW_ON_BLOCK)
COMMIT_MIN_GAP   = 8.0      # deg; widest gap must be at least this wide to commit forward
GAP_CLEAR_FRAC   = 0.40     # widest-gap "clear" cut: e_min + this×(e_max−e_min). Range-
                            # relative so it still finds the opening when a wall fills
                            # most of the FOV (where the median-adaptive cut would miss it)

# ── Arrival detection (stop instead of overshooting the destination) ──────────
# The gold marker's RADIAL distance from the minimap centre shrinks as we approach
# (radial = rotation-invariant, like the latch bearing). Very close, the diamond is
# replaced by the destination icon — gold detection returns nothing — and the trail
# dots go sparse. We declare ARRIVAL only after the marker has been CLOSE (entered
# the arrival zone) and THEN reads very-near, or vanishes with no trail, for a few
# ticks. The "must have been close first" gate stops a marker merely occluded
# mid-field (or a far rim-clamped target) from triggering arrival. On arrival we
# release the stick and hold; a NEW far marker OR a returning trail (next objective)
# resumes navigation. Interaction at the spot stays with the interact-prompt path.
ARRIVAL_ENABLE      = os.environ.get("ARRIVAL", "1") not in ("0", "false", "")
ARRIVAL_RADIUS_FRAC = 0.30   # marker within this × minimap-radius = in the arrival zone
ARRIVAL_NEAR_FRAC   = 0.15   # marker this close (still drawn) = arrived outright
ARRIVAL_RESUME_FRAC = 0.55   # a marker beyond this (post-arrival) = new objective → resume
ARRIVAL_CONFIRM     = 4       # consecutive confirming ticks before declaring arrival
ARRIVAL_MAX_DOTS    = 4       # near-dot count at/below which the trail counts as "gone"
# Metre thresholds (preferred when the objective-distance OCR is available — it's the
# game's own ground-truth distance, with none of the rim-clamp / centre-jitter
# pathologies of the gold-pixel geometry). Pixel fracs above are the fallback.
ARRIVAL_DIST_M      = 5       # arrived within this many metres of the objective
ARRIVAL_ZONE_M      = 20      # a numeric read within this arms the arrival (seen_close)
ARRIVAL_RESUME_M    = 30      # beyond this after arriving = a new objective → resume
_MM_R = min(MINIMAP_REGION[2], MINIMAP_REGION[3]) / 2.0   # minimap usable radius (px)
GOLD_TRUST_MIN_DIST = GOLD_TRUST_MIN_FRAC * _MM_R         # px; below this the bearing is noise

# ── Progress detection (viewport-independent stall) ───────────────────────────
# The edge viewport is blind to low-contrast / horizontal-edged walls (a shadowed
# log cabin reads as 'clear'), and because it reads clear neither the live-lock
# counter (keyed on the viewport's blocked flag) nor full-frame SSIM (defeated by
# idle animation + foliage) fires — Geralt can press a wall indefinitely. So we also
# watch the gold marker's RADIAL distance: rotation-invariant (a weave can't fool
# it), and if it stops decreasing while we steer straight at the marker, we're
# blocked regardless of what the viewport thinks. Recovery is reverse+sidestep —
# never the viewport widest-gap, whose histogram is poisoned by the same blind spot.
# Active only on gold/latch steering (no trail): while following a trail the route
# may legitimately curve away from the marker, so we don't second-guess it.
PROGRESS_WINDOW     = 15    # ticks of gold/latch steering examined for closure
PROGRESS_EPS        = 6.0   # px; must close at least this much over the window, else stalled
PROGRESS_RANGE      = 10.0  # px; if the distance also varied less than this it barely moved
                            # (flat = wall press); a detour has a large range and is spared
PROGRESS_CLAMP_FRAC = 0.80  # ignore a rim-clamped far marker (its distance saturates there)

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

def set_sticks(lx: float, ly: float, rx: float = 0.0, ry: float = 0.0):
    """Low-level set-and-hold. Values already in W3 convention; server applies raw."""
    _control_send({"type": "sticks",
                   "left_x":  float(np.clip(lx, -1, 1)), "left_y":  float(np.clip(ly, -1, 1)),
                   "right_x": float(np.clip(rx, -1, 1)), "right_y": float(np.clip(ry, -1, 1))})

def move_dir(heading_deg: float, speed: float = 1.0):
    """
    Directional movement. heading_deg is measured on the minimap (0 = up =
    camera-forward, + = right). The left stick points that way; Geralt turns to
    face it and runs, and the camera auto-trails — no right-stick rate steering,
    so no camera-rotation feedback loop. W3 inverts Y, hence -cos.
    """
    h = math.radians(heading_deg)
    set_sticks(math.sin(h) * speed, -math.cos(h) * speed, 0.0, 0.0)

def button(name: str, duration: float = 0.12):
    _control_send({"type": "button", "button": name, "duration": duration})

def release():
    _control_send({"type": "release"})

def to_b64(img: Image.Image, fmt: str = "PNG", quality: int = 85) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def crop(screen: Image.Image, region: tuple) -> Image.Image:
    l, t, w, h = region
    return screen.crop((l, t, l + w, t + h))

# ══════════════════════════════════════════════════════════════════════════════
# Minimap → ordered path → steering
# ══════════════════════════════════════════════════════════════════════════════

def extract_dots(screen: Image.Image, save: bool = False):
    """
    Returns (dots, centre, mask) with dots as (x,y) centroids in ORIGINAL minimap
    pixels. The crop is upscaled MINIMAP_UPSCALE× so the tiny dots resolve into
    clean round blobs; detection is by WHITENESS (all channels bright + neutral),
    which keeps white path dots and rejects yellow icons / gold arc / green signposts.
    """
    mm   = np.array(crop(screen, MINIMAP_REGION))
    U    = MINIMAP_UPSCALE
    big  = cv2.resize(mm, None, fx=U, fy=U, interpolation=cv2.INTER_LINEAR) if U > 1 else mm
    h, w = big.shape[:2]
    cx, cy = w // 2, h // 2

    r = big[:, :, 0].astype(int); g = big[:, :, 1].astype(int); b = big[:, :, 2].astype(int)
    mn = np.minimum(np.minimum(r, g), b)
    mx = np.maximum(np.maximum(r, g), b)
    mask = ((mn > WHITE_MIN) & ((mx - mn) < WHITE_SPREAD)).astype(np.uint8) * 255

    cv2.circle(mask, (cx, cy), DOT_EXCLUDE_R * U, 0, -1)
    border = np.zeros_like(mask)
    cv2.circle(border, (cx, cy), min(cx, cy) - DOT_BORDER_PAD * U, 255, -1)
    mask = cv2.bitwise_and(mask, border)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    s2 = U * U
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    dots = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (DOT_AREA_MIN * s2 <= area <= DOT_AREA_MAX * s2):
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]; bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= DOT_GUARD_AREA * s2:                 # judge only sizeable blobs
            extent = area / float(bw * bh) if bw * bh else 0.0
            aspect = max(bw, bh) / float(min(bw, bh)) if min(bw, bh) else 99.0
            if extent < DOT_MIN_EXTENT or aspect > DOT_MAX_ASPECT:
                continue
        dots.append((cents[i][0] / U, cents[i][1] / U))   # back to original scale
        keep[labels == i] = 255

    origin = (mm.shape[1] // 2, mm.shape[0] // 2)
    if save:
        Image.fromarray(mm).save("minimap_raw.png")
        small = cv2.resize(keep, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(small).save("minimap_mask.png")
    return dots, origin, keep

def _ang(origin, p):
    """Angle (deg) from minimap-up to p, +ve = right. up = Geralt-forward."""
    return math.degrees(math.atan2(p[0] - origin[0], -(p[1] - origin[1])))

def _ang_diff(a, b):
    """Smallest signed difference a-b in degrees, wrapped to [-180,180]."""
    return ((a - b + 180) % 360) - 180

def _circ_mean(angles):
    """Circular mean (deg) and resultant length R in [0,1] (1 = all aligned)."""
    s = sum(math.sin(math.radians(a)) for a in angles) / len(angles)
    c = sum(math.cos(math.radians(a)) for a in angles) / len(angles)
    return math.degrees(math.atan2(s, c)), math.hypot(s, c)

def _save_minimap_debug(mm_rgb, origin, all_dots, near_dots, heading, src, conc, gold_pt=None):
    """Annotated minimap: all dots (white), near dots (green), gold marker (magenta),
    heading arrow (blue)."""
    img = np.ascontiguousarray(mm_rgb.copy())
    cx, cy = origin
    cv2.circle(img, (cx, cy), R_NEAR, (255, 170, 0), 1)        # near-field ring
    for d in all_dots:
        cv2.circle(img, (int(d[0]), int(d[1])), 3, (255, 255, 255), 1)
    for d in near_dots:
        cv2.circle(img, (int(d[0]), int(d[1])), 3, (0, 255, 0), -1)
    if gold_pt is not None:
        cv2.circle(img, (int(gold_pt[0]), int(gold_pt[1])), 6, (255, 0, 255), 2)  # gold marker
    if heading is not None:
        hr = math.radians(heading)
        ex = int(cx + 45 * math.sin(hr)); ey = int(cy - 45 * math.cos(hr))
        cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 80, 255), 2, tipLength=0.3)
    cv2.circle(img, (cx, cy), 3, (255, 0, 0), -1)              # centre
    cv2.putText(img, f"{src} c={conc:.2f}", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    Image.fromarray(img).save("minimap_annotated.png")

def find_gold_marker(mm_rgb, origin, prev_ang=None):
    """Bearing of the gold destination diamond, or (None, None, 0). Compactness
    rejects the thin quest-area arc and the rim ring.

    TEMPORAL COHERENCE (Fix 2): the minimap often shows TWO amber blobs — the
    destination diamond AND a nearby quest/notice icon. Picking the largest each
    frame flip-flops between them when their areas are close (the +4°/+30° square
    wave that drove the barn oscillation). Instead, among the valid blobs we keep
    the one nearest PREV_ANG (last frame's pick) and only switch to a different blob
    if it's clearly bigger (GOLD_SWITCH_RATIO) — a more convincing marker. With no
    prior we seed on the largest."""
    cx, cy = origin
    hsv  = cv2.cvtColor(mm_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(GOLD_HSV_LO), np.array(GOLD_HSV_HI))
    interior = np.zeros_like(mask)
    cv2.circle(interior, (cx, cy), int(min(cx, cy) * GOLD_RIM_FRAC), 255, -1)  # drop rim
    cv2.circle(interior, (cx, cy), DOT_EXCLUDE_R, 0, -1)                       # drop centre
    mask = cv2.bitwise_and(mask, interior)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    cands = []                                            # (bearing, area, pt)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < GOLD_MIN_PIX:
            continue
        bw = stats[i, cv2.CC_STAT_WIDTH]; bh = stats[i, cv2.CC_STAT_HEIGHT]
        extent = area / float(bw * bh) if bw * bh else 0.0
        if extent < GOLD_MIN_EXTENT or max(bw, bh) > GOLD_MAX_BBOX:   # reject arc / rim
            continue
        pt = (float(cents[i][0]), float(cents[i][1]))
        cands.append((_ang(origin, pt), float(area), pt))

    if not cands:
        return None, None, 0

    largest = max(cands, key=lambda c: c[1])
    if prev_ang is None:
        best = largest                                    # seed on the biggest blob
    else:
        nearest = min(cands, key=lambda c: abs(_ang_diff(c[0], prev_ang)))
        # hold the tracked (nearest) blob unless the largest is clearly bigger
        best = largest if (largest is not nearest and
                           largest[1] > nearest[1] * GOLD_SWITCH_RATIO) else nearest
    return best[0], best[2], best[1]

_gold_latch_ang: Optional[float] = None   # last seen gold bearing (deg, minimap frame)
_gold_latch_age: int = 0                  # ticks since the marker was last seen
_last_gold_dist: Optional[float] = None   # radial px from centre to gold marker (None = absent)

def _trail_dir(pool, origin, prev_heading, gold_ang):
    """Mean direction of a dot pool, rejecting the 'behind' cluster (came-from) so a
    hard turn around an obstacle survives. Returns (heading, conc, kept_dots)."""
    angs = [_ang(origin, d) for d in pool]
    behind = None
    if prev_heading is not None:
        behind = prev_heading + 180.0
    elif gold_ang is not None:
        behind = gold_ang + 180.0          # startup: assume we came from opposite the goal
    if behind is not None:
        keep = [i for i, a in enumerate(angs) if abs(_ang_diff(a, behind)) > BEHIND_CONE]
        use_idx = keep if len(keep) >= 2 else list(range(len(pool)))
    else:
        use_idx = list(range(len(pool)))
    h, c = _circ_mean([angs[i] for i in use_idx])
    return h, c, [pool[i] for i in use_idx]

def path_heading(screen: Image.Image, prev_heading, save: bool = False):
    """
    Combined heading. Priority:
      1. Near dotted TRAIL — routes around obstacles (forward = reject the behind cluster).
      2. Current GOLD marker — direct bearing on open ground / mid-field.
      3. LATCHED gold — near the destination the marker is gone and the dots are
         unreliable: keep steering toward its LAST bearing, decaying to straight, so
         we walk in to the target instead of getting lost chasing dots.
      4. Wider trail / single dot — deep fallbacks. Else none (loop holds).
    Returns (heading_deg, n_near, conc, source). source 'path'|'gold'|'latch' = trust.
    """
    global _gold_latch_ang, _gold_latch_age, _last_gold_dist
    dots, origin, _ = extract_dots(screen, save=False)
    mm_rgb = np.array(crop(screen, MINIMAP_REGION))
    gold_ang, gold_pt, gold_px = find_gold_marker(mm_rgb, origin, prev_ang=_gold_latch_ang)
    _last_gold_dist = (math.hypot(gold_pt[0] - origin[0], gold_pt[1] - origin[1])
                       if gold_pt is not None else None)

    # Distance-gate the BEARING. Within GOLD_TRUST_MIN_DIST of centre the marker is
    # only a few px off-centre, so its ANGLE is noise — tiny centroid jitter swings it
    # wildly and flips it ~180° across centre (the corner-orbit failure). We keep the
    # distance (arrival relies on it) but suppress the bearing so we don't steer by it:
    # the latch (last trustworthy bearing, decaying to straight) walks us in and
    # arrival stops us on the spot.
    gold_trusted = (gold_ang is not None and _last_gold_dist is not None
                    and _last_gold_dist >= GOLD_TRUST_MIN_DIST)
    gold_use = gold_ang if gold_trusted else None

    # Maintain the gold latch: refresh on a TRUSTED sight, otherwise age it out.
    if gold_use is not None:
        _gold_latch_ang, _gold_latch_age = gold_use, 0
    elif _gold_latch_ang is not None:
        _gold_latch_age += 1
        if _gold_latch_age > GOLD_LATCH_TICKS:
            _gold_latch_ang = None

    near, alld = [], []
    for d in dots:
        dd = math.hypot(d[0] - origin[0], d[1] - origin[1])
        if dd <= DOT_EXCLUDE_R:
            continue
        alld.append(d)
        if dd <= R_NEAR:
            near.append(d)

    heading, conc, src = None, 0.0, "none"

    # 1. PRIMARY — the near dotted trail, but only when it's RELIABLE: enough dots
    #    (MIN_DOTS_FOR_HEADING) AND angularly concentrated (CONC_MIN). The trail
    #    routes around obstacles, so a TRUSTED trail outranks the straight gold
    #    bearing. An UNRELIABLE trail (too few or scattered dots — the typical case
    #    NEAR THE DESTINATION, where the gold diamond becomes the target icon and
    #    the dots go sparse) must NOT pre-empt gold/latch, so it defers (Fix 3).
    if len(near) >= MIN_DOTS_FOR_HEADING:
        h_n, c_n, near_kept = _trail_dir(near, origin, prev_heading, gold_use)
        if c_n >= CONC_MIN:
            heading, conc, near, src = h_n, c_n, near_kept, "path"

    # 2–5. Trail not trusted → straight gold bearing, then its near-destination
    #      latch, then a wider whole-map trail read, then whatever scattered dots
    #      remain (untrusted → the loop holds the last heading).
    if src != "path":
        if gold_use is not None:
            heading, conc, src = gold_use, 1.0, "gold"           # direct bearing
        elif _gold_latch_ang is not None:                        # near destination: latch
            decay = math.exp(-_gold_latch_age / GOLD_LATCH_TAU)  # turn-then-straighten
            heading, conc, src = _gold_latch_ang * decay, 1.0, "latch"
        elif len(alld) >= MIN_DOTS_FOR_HEADING:                  # wider trail fallback
            heading, conc, near = _trail_dir(alld, origin, prev_heading, gold_use)
            src = "path" if conc >= CONC_MIN else "spread"
        elif len(near) >= 2:                                     # last resort: hold on it
            heading, conc, near = _trail_dir(near, origin, prev_heading, gold_use)
            src = "spread"
        elif len(near) == 1:
            heading, conc, src = _ang(origin, near[0]), 1.0, "one"

    if save:
        _save_minimap_debug(mm_rgb, origin, dots, near, heading, src, conc, gold_pt)

    return heading, len(near), conc, src

# ══════════════════════════════════════════════════════════════════════════════
# Viewport obstacle avoidance — Vector Field Histogram over the near-ground image
# ══════════════════════════════════════════════════════════════════════════════
# WHY VFH and not a raw potential field (the obvious first idea): a potential field
# sums an attractive goal vector with repulsive obstacle vectors. It is prone to
# (a) freezing in a local minimum BETWEEN two obstacles and (b) oscillation, and in
# dense clutter (forest) every direction repels so it stalls. VFH instead bins
# obstacle energy into angular SECTORS and seeks the clearest GAP nearest the goal
# heading. With an ADAPTIVE threshold it raises its own bar in a busy frame, so it
# degrades toward "trust the trail" rather than "blocked everywhere" — the same
# graceful-failure property we wanted after the minimap-VLM clutter problem.
#
# Frame: the viewport shares the minimap's CAMERA frame — centre column = forward =
# 0°, + = right — because steering is left-stick-directional and the camera trails.
# So a sector angle is directly comparable to the trail heading; no transform.

def _sector_angles():
    """Centre angle (deg, + = right) of each histogram sector across the FOV."""
    half, step = VIEW_FOV_DEG / 2.0, VIEW_FOV_DEG / N_SECTORS
    return [(-half + step * (i + 0.5)) for i in range(N_SECTORS)]

_SECTOR_ANG = _sector_angles()

def viewport_histogram(screen: Image.Image):
    """Per-sector obstacle energy (normalised 0..1) from the near-ground viewport.
    Upright obstacles are vertical lines → strong horizontal gradient (Sobel-x).
    Nearer (lower) rows weigh more; Geralt's body slice is blanked. Returns
    (energy[N_SECTORS], debug-dict)."""
    crop_img = np.array(crop(screen, VIEWPORT_REGION))
    g  = cv2.GaussianBlur(cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    sx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))     # vertical-edge energy
    h, w = sx.shape
    sx *= (np.linspace(0.0, 1.0, h).reshape(-1, 1) ** NEAR_ROW_GAMMA)  # near = urgent
    bc = int(BODY_MASK_CENTRE * w); bw = int(BODY_MASK_WIDTH * w / 2)
    by = int(BODY_MASK_BOTTOM * h)                           # blank centre only below here
    sx[by:, max(0, bc - bw):min(w, bc + bw)] = 0.0           # Geralt's body/feet
    col = sx.sum(axis=0)

    edges  = np.linspace(0, w, N_SECTORS + 1).astype(int)
    energy = np.array([col[edges[i]:edges[i + 1]].mean() if edges[i + 1] > edges[i]
                       else 0.0 for i in range(N_SECTORS)])
    mx = energy.max()
    energy = energy / mx if mx > 1e-6 else energy
    return energy, {"crop": crop_img, "edge": sx, "w": w, "h": h, "bc": bc, "bw": bw, "by": by}

def _ang_to_sector(angle: float):
    """Sector index for an angle, or None if it falls outside the viewport FOV."""
    half = VIEW_FOV_DEG / 2.0
    if angle <= -half or angle >= half:
        return None
    return min(N_SECTORS - 1, int((angle + half) / VIEW_FOV_DEG * N_SECTORS))

def viewport_avoid(screen: Image.Image, desired: float, save: bool = False):
    """Deflect `desired` (a camera-frame heading, deg) toward the nearest clear gap.
    Returns (heading, blocked_ahead, clearance, source):
      source 'clear' — path ahead open, heading unchanged.
      source 'avoid' — desired was blocked; deflected to nearest clear sector.
      source 'boxed' — whole avoidance cone blocked; hold heading, slow, let SSIM act.
      source 'wide'  — desired is outside the FOV (a hard turn); can't judge, pass through."""
    energy, dbg = viewport_histogram(screen)
    thresh  = max(BLOCK_FLOOR, float(np.median(energy)) * BLOCK_K)
    blocked = energy > thresh

    di = _ang_to_sector(desired)
    if di is None:                                           # hard turn beyond the FOV
        if save: _save_viewport_debug(dbg, energy, blocked, desired, desired, "wide")
        return desired, False, 1.0, "wide"

    clearance = 1.0 - float(energy[di])
    if not blocked[di]:                                      # path ahead is open
        if save: _save_viewport_debug(dbg, energy, blocked, desired, desired, "clear")
        return desired, False, clearance, "clear"

    # blocked ahead: search outward for the nearest clear sector inside the cone
    best = None
    for off in range(1, N_SECTORS):
        for s in (di - off, di + off):
            if 0 <= s < N_SECTORS and not blocked[s] \
               and abs(_SECTOR_ANG[s] - desired) <= AVOID_CONE_DEG:
                best = s; break
        if best is not None:
            break

    if best is None:                                         # boxed in
        if save: _save_viewport_debug(dbg, energy, blocked, desired, desired, "boxed")
        return desired, True, clearance, "boxed"

    if save: _save_viewport_debug(dbg, energy, blocked, desired, _SECTOR_ANG[best], "avoid")
    return _SECTOR_ANG[best], True, clearance, "avoid"

def viewport_widest_gap(screen: Image.Image):
    """For live-lock recovery: the centre bearing of the WIDEST contiguous run of
    clear sectors, plus its angular width (deg). Returns (bearing, width_deg). When
    nothing is clear, returns the single least-blocked sector with width 0.

    Uses a RANGE-relative threshold, not viewport_avoid's median-adaptive one: with
    a wall filling most of the FOV the median rides up with the wall and nothing
    reads as blocked. During a live-lock we always want the best opening, so we
    threshold off the energy range (clearest..most-blocked) and take the widest run
    of the relatively-clearest sectors."""
    energy, _ = viewport_histogram(screen)
    e_min, e_max = float(energy.min()), float(energy.max())
    thresh = e_min + GAP_CLEAR_FRAC * (e_max - e_min)
    clear  = energy <= thresh
    step   = VIEW_FOV_DEG / N_SECTORS

    best_lo = best_len = -1
    i = 0
    while i < N_SECTORS:
        if clear[i]:
            j = i
            while j < N_SECTORS and clear[j]:
                j += 1
            if (j - i) > best_len:
                best_len, best_lo = j - i, i
            i = j
        else:
            i += 1

    if best_len <= 0:                                    # nothing clear — least-blocked sector
        s = int(np.argmin(energy))
        return _SECTOR_ANG[s], 0.0
    centre_idx = best_lo + (best_len - 1) / 2.0
    bearing = -VIEW_FOV_DEG / 2.0 + step * (centre_idx + 0.5)
    return bearing, best_len * step

def _save_viewport_debug(dbg, energy, blocked, desired, chosen, src):
    """viewport_annotated.png — sector bars (red=blocked, green=clear) over the crop,
    yellow arrow = desired (trail) heading, magenta = chosen heading after avoid."""
    img = np.ascontiguousarray(dbg["crop"].copy())
    h, w = dbg["h"], dbg["w"]
    bc, bw, by = dbg["bc"], dbg["bw"], dbg["by"]
    cv2.rectangle(img, (bc - bw, by), (bc + bw, h - 1), (90, 90, 90), 1)  # body mask
    edges = np.linspace(0, w, N_SECTORS + 1).astype(int)
    for i in range(N_SECTORS):
        bar = int(energy[i] * (h - 4))
        cv2.rectangle(img, (edges[i] + 1, h - 1 - bar), (edges[i + 1] - 1, h - 1),
                      (255, 60, 60) if blocked[i] else (60, 200, 60), -1)
    def ang_x(a):
        return int(np.clip((a + VIEW_FOV_DEG / 2.0) / VIEW_FOV_DEG * w, 0, w - 1))
    cv2.arrowedLine(img, (ang_x(desired), h - 1), (ang_x(desired), 6), (255, 255, 0), 2, tipLength=0.2)
    cv2.arrowedLine(img, (ang_x(chosen),  h - 1), (ang_x(chosen),  6), (255, 0, 255), 2, tipLength=0.2)
    cv2.putText(img, src, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    Image.fromarray(img).save("viewport_annotated.png")
    Image.fromarray(dbg["edge"].clip(0, 255).astype(np.uint8)).save("viewport_edges.png")
    Image.fromarray(dbg["crop"]).save("viewport_raw.png")

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
        temperature: float = 0.3, max_tokens: int = 64, img_fmt: str = "PNG") -> str:
    content = []
    if image is not None:
        mime = "jpeg" if img_fmt.upper() == "JPEG" else "png"
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{to_b64(image, img_fmt)}"}})
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

# ── VLM navigation (background thread) ────────────────────────────────────────

VLM_NAV_SYSTEM = (
    "You read a Witcher 3 minimap: a circular compass with the player as the white "
    "arrowhead at the EXACT CENTER, always facing UP (12 o'clock).\n"
    "A large GOLD/AMBER diamond marker on the map marks the quest destination, and "
    "a trail of small white dots leads from the player toward it.\n"
    "IGNORE everything else: green leaf icons (herbs), green diamond signposts, "
    "buildings, roads, terrain.\n"
    "Report the CLOCK DIRECTION from the centre to the GOLD destination marker. "
    "If no gold marker is visible, use the white dotted trail instead. "
    "12 = up/ahead, 3 = right, 6 = down/behind, 9 = left.\n"
    "Answer with ONLY two lines:\nCLOCK: <1-12>\nCONF: <high|low>"
)

_nav_lock = threading.Lock()
_committed_heading: Optional[float] = None    # degrees, minimap frame
_nav_conf = "low"
_nav_age  = 0.0

def _minimap_http() -> Optional[Image.Image]:
    """Fetch just the minimap region over HTTP (independent of the loop's WS)."""
    l, t, w, h = MINIMAP_REGION
    try:
        r = http.get(f"{SCREEN_HTTP}/region",
                     params={"l": l, "t": t, "w": w, "h": h, "quality": 92}, timeout=5)
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print(f"  ⚠  nav minimap fetch: {e}")
        return None

def _parse_clock(resp: str):
    """Return (heading_deg in [-180,180], conf) from a CLOCK/CONF reply, or (None, 'low')."""
    clock, conf = None, "low"
    for line in resp.splitlines():
        u = line.upper()
        if "CLOCK:" in u:
            digits = "".join(ch for ch in u.split("CLOCK:", 1)[1] if ch.isdigit())
            if digits:
                clock = int(digits)
        if "CONF:" in u and "HIGH" in u:
            conf = "high"
    if clock is None or not (1 <= clock <= 12):
        return None, conf
    deg = (clock % 12) * 30.0            # 12->0, 3->90, 6->180, 9->270
    if deg > 180:
        deg -= 360                       # wrap to [-180,180]
    return deg, conf

def _nav_thread():
    """Continuously ask the VLM which way the path leads; update committed heading."""
    global _committed_heading, _nav_conf, _nav_age
    sm_sin = sm_cos = None
    while True:
        t0 = time.time()
        mm = _minimap_http()
        t_fetch = time.time() - t0
        if mm is not None:
            if VLM_UPSCALE > 1:
                mm = mm.resize((mm.width * VLM_UPSCALE, mm.height * VLM_UPSCALE),
                               Image.LANCZOS)
            try:
                mm.save("nav_sent.jpg", quality=85)   # exactly what the VLM sees
            except Exception:
                pass
            # timestamp-bust LM Studio's KV cache (identical prompts return stale replies)
            prompt = f"Which clock direction is the gold destination marker? [t:{time.time():.3f}]"
            t_inf0 = time.time()
            resp = ask(prompt, VLM_NAV_SYSTEM, image=mm, temperature=0.1,
                       max_tokens=24, img_fmt="JPEG")
            t_inf = time.time() - t_inf0
            deg, conf = _parse_clock(resp)
            one_line = resp.replace("\n", " | ")[:48]
            print(f"  ⟳ nav: fetch {int(t_fetch*1000)}ms  infer {int(t_inf*1000)}ms  "
                  f"→ {one_line!r}"
                  + ("" if deg is not None else "  (unparsed → holding)"))
            if deg is not None:
                s, c = math.sin(math.radians(deg)), math.cos(math.radians(deg))
                if sm_sin is None:
                    sm_sin, sm_cos = s, c
                else:
                    sm_sin = NAV_EMA * s + (1 - NAV_EMA) * sm_sin
                    sm_cos = NAV_EMA * c + (1 - NAV_EMA) * sm_cos
                with _nav_lock:
                    _committed_heading = math.degrees(math.atan2(sm_sin, sm_cos))
                    _nav_conf = conf
                    _nav_age  = time.time()
        time.sleep(max(0.0, VLM_PERIOD - (time.time() - t0)))

def nav_get():
    with _nav_lock:
        return _committed_heading, _nav_conf, (time.time() - _nav_age if _committed_heading is not None else None)

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
        set_sticks(0.0, -1.0); time.sleep(0.2); release()
    elif direction == "down":
        set_sticks(0.0,  1.0); time.sleep(0.2); release()
    else:
        button("a")

# ══════════════════════════════════════════════════════════════════════════════
# Stuck detection
# ══════════════════════════════════════════════════════════════════════════════

class StuckDetector:
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.still  = 0
        self.last   = 1.0
        self.last_side = 1.0
        self.last_t    = 0.0
        self.streak    = 0

    def record(self, screen: Image.Image):
        grey = np.array(screen)[200:800, 300:1620].mean(axis=2).astype(np.float32)
        self.frames.append(grey)

    def is_stuck(self) -> bool:
        if len(self.frames) < 2:
            return False
        self.last  = float(ssim(self.frames[-2], self.frames[-1], data_range=255))
        self.still = self.still + 1 if self.last > SSIM_THRESHOLD else 0
        return self.still >= STUCK_TICKS

    def recover(self) -> float:
        """Free Geralt from an obstacle. FIRST try a forward jump — rocks and wood
        piles usually clear with a hop and it doesn't throw away the heading. If the
        stuck repeats, escalate to backing off and sidestepping around it. Returns the
        edge-around side (+1/-1), or 0 after a jump (no lateral bias, keep heading)."""
        self.still = 0
        self.frames.clear()
        now = time.time()
        self.streak = self.streak + 1 if (now - self.last_t) < 4.0 else 1
        self.last_t = now

        if self.streak == 1:                            # first attempt: hop the obstacle
            print(f"  ⚠  stuck (ssim={self.last:.3f}) — jump forward")
            move_dir(0.0, 0.85); button(JUMP_BUTTON); time.sleep(0.35)
            move_dir(0.0, 0.90); time.sleep(0.20)
            release()
            return 0

        self.last_side = -self.last_side               # escalate: alternate side
        side = self.last_side
        esc  = min(self.streak, 3)                      # 2..3
        back = 0.30 + 0.10 * esc
        turn = 0.40 + 0.20 * esc
        tag  = "R" if side > 0 else "L"
        print(f"  ⚠  stuck (ssim={self.last:.3f}) streak={self.streak} — back off, edge {tag}")
        move_dir(180.0, 0.70);       time.sleep(back)   # peel off
        move_dir(side * 90.0, 0.90); time.sleep(turn)   # sidestep around
        release()
        return side

    def reset(self):
        self.still  = 0
        self.streak = 0
        self.frames.clear()

# ══════════════════════════════════════════════════════════════════════════════
# Arrival detection — stop at the destination instead of overshooting
# ══════════════════════════════════════════════════════════════════════════════

class ArrivalDetector:
    """Fed each exploration tick with the gold marker's radial distance (px from the
    minimap centre, None if no marker) and the near-dot count. Returns True once we've
    arrived at the active quest destination, so the loop can release and hold.

    Arrival is gated on having entered the arrival ZONE first, then either reading
    very-near or vanishing-with-no-trail for ARRIVAL_CONFIRM ticks. The gate is what
    makes this robust: a marker that's merely occluded mid-field, or a far rim-clamped
    target, never enters the zone and so can't trigger a false arrival. Once arrived,
    we keep holding until a NEW far marker or a returning trail signals a next
    objective (the latter also covers a rim-clamped new target that reads as no gold)."""

    def __init__(self, mm_radius: float):
        self.R = mm_radius
        self.reset()

    def update(self, gold_dist: Optional[float], n_near: int,
               dist_m: Optional[float] = None) -> bool:
        # Prefer the game's objective distance in metres when available (clean scalar,
        # no rim-clamp, no centre-jitter); fall back to the gold-pixel geometry.
        if dist_m is not None:
            return self._update_m(dist_m)
        return self._update_px(gold_dist, n_near)

    def _update_m(self, d: float) -> bool:
        if self.arrived:
            if d > ARRIVAL_RESUME_M:                 # new objective further out → resume
                self.reset()
                return False
            return True
        if 0 < d <= ARRIVAL_ZONE_M:                  # a NUMERIC close read arms us; NEARBY
            self.seen_close = True                   # (d==0) deliberately does NOT self-arm,
        if d > ARRIVAL_RESUME_M:                     # so a stray false 'NEARBY' while far
            self.reset()                             # (no prior count-down) can't fake arrival
            return False
        cond = self.seen_close and d <= ARRIVAL_DIST_M
        self.confirm = self.confirm + 1 if cond else 0
        if self.confirm >= ARRIVAL_CONFIRM:
            self.arrived = True
        return self.arrived

    def _update_px(self, gold_dist: Optional[float], n_near: int) -> bool:
        near_r   = ARRIVAL_NEAR_FRAC   * self.R
        zone_r   = ARRIVAL_RADIUS_FRAC * self.R
        resume_r = ARRIVAL_RESUME_FRAC * self.R

        # Already arrived → hold until there's clear evidence of a NEXT objective:
        # a marker reappears far out, or the dotted trail comes back (new route drawn).
        if self.arrived:
            if (gold_dist is not None and gold_dist > resume_r) or n_near > ARRIVAL_MAX_DOTS:
                self.reset()
                return False
            return True

        if gold_dist is not None:
            if gold_dist <= zone_r:
                self.seen_close = True
            if gold_dist > resume_r:            # marker jumped far → new objective, not arrival
                self.reset()
                return False

        # Arrival conditions only count once we've actually been close.
        cond = self.seen_close and (
            (gold_dist is not None and gold_dist <= near_r) or      # very near, still drawn
            (gold_dist is None and n_near <= ARRIVAL_MAX_DOTS))      # diamond→icon + trail gone
        self.confirm = self.confirm + 1 if cond else 0
        if self.confirm >= ARRIVAL_CONFIRM:
            self.arrived = True
        return self.arrived

    def reset(self):
        self.seen_close = False
        self.confirm    = 0
        self.arrived    = False

# ══════════════════════════════════════════════════════════════════════════════
# Progress detection — catch wall-presses the viewport and SSIM both miss
# ══════════════════════════════════════════════════════════════════════════════

class ProgressDetector:
    """Viewport-INDEPENDENT stall check. Edge-based viewport perception is blind to
    low-contrast / horizontal-edged walls (a shadowed log cabin), so it reports
    'clear' while Geralt presses one; the live-lock counter (keyed on that flag)
    can't fire, and full-frame SSIM is defeated by idle animation + swaying foliage.
    We watch the gold marker's RADIAL distance instead — rotation-invariant, so a
    weave can't fool it. If it isn't closing while we steer straight at the marker,
    we're blocked. Active only on gold/latch steering with no trail (open-ground
    straight approach); while following a trail the route may legitimately curve
    away from the marker, which removes the detour false-positive."""

    def __init__(self):
        self.dists = deque(maxlen=PROGRESS_WINDOW)
        self.side  = 1.0

    def record(self, gold_dist: Optional[float], source: str, mm_radius: float):
        if source not in ("gold", "latch") or gold_dist is None \
           or gold_dist > PROGRESS_CLAMP_FRAC * mm_radius:
            self.dists.clear()
            return
        self.dists.append(gold_dist)

    def stalled(self) -> bool:
        if len(self.dists) < PROGRESS_WINDOW:
            return False
        net   = self.dists[0] - self.dists[-1]           # >0 = net closer over the window
        rng   = max(self.dists) - min(self.dists)        # how much it moved at all
        # Stalled = didn't get meaningfully closer AND barely moved. The range term
        # rejects a legitimate detour (distance rises then falls — large range, he IS
        # translating); the net term lets slow-but-steady real progress through.
        return net < PROGRESS_EPS and rng < PROGRESS_RANGE

    def recover(self) -> float:
        """Reverse off the wall, then sidestep along it, alternating sides on repeat.
        Deliberately NOT the viewport widest-gap — its histogram is blind to this wall,
        so it would just steer us back into it."""
        self.dists.clear()
        self.side = -self.side
        tag = "R" if self.side > 0 else "L"
        print(f"  🧱 NO PROGRESS (gold not closing) — reverse + sidestep {tag}")
        move_dir(180.0, 0.70);            time.sleep(0.35)   # back off the wall
        move_dir(self.side * 90.0, 0.90); time.sleep(0.40)   # sidestep along it
        release()
        return self.side

    def reset(self):
        self.dists.clear()

# ══════════════════════════════════════════════════════════════════════════════
# Objective-distance reader — OCR the footprints/step counter, validated
# ══════════════════════════════════════════════════════════════════════════════

class DistanceReader:
    """Reads the objective distance (metres) from the HUD step counter. This is the
    game's own ground-truth proximity — no rim-clamp, no centre-jitter, a clean scalar
    with no angle to flip — so it's the preferred proximity gauge for arrival. But it's
    a small stylised HUD number next to a footprints glyph, so a raw OCR read is a
    liability (we watched the interact OCR hallucinate 'Wuegoue'); a bad read here could
    fake an arrival. So every read is validated: digits only, plausibility ceiling, a
    one-off large jump is rejected as a glitch unless a consecutive read confirms it (a
    real objective change), and last-good is held across brief failures."""

    def __init__(self):
        self.reset()

    def read(self, screen: Image.Image) -> Optional[int]:
        raw = ocr(crop(screen, DISTANCE_REGION))
        self.age += 1

        # "NEARBY" — the game's own "you're basically there" (< ~10 steps), shown
        # INSTEAD of a number in the final stretch. It's the clearest arrival signal
        # in the HUD, so treat it as distance 0. Fuzzy match (OCR may give NEAR8Y /
        # NEAREY): letters-only, must start with NEAR — digits can't false-match this.
        letters = "".join(c for c in raw.upper() if c.isalpha())
        if letters.startswith("NEAR"):
            self.last, self.age, self.pending = 0, 0, None
            return 0

        digits = "".join(c for c in raw if c.isdigit())
        val = int(digits) if digits else None

        if val is None or not (0 < val <= DISTANCE_MAX):
            self.pending = None
            return self.last if self.age <= DISTANCE_STALE else None

        if self.last is None or abs(val - self.last) <= DISTANCE_MAX_JUMP:
            self.last, self.age, self.pending = val, 0, None     # plausible: trust
            return val

        # large jump: glitch unless a consecutive read confirms it (real change)
        if self.pending is not None and abs(val - self.pending) <= DISTANCE_CONFIRM_TOL:
            self.last, self.age, self.pending = val, 0, None
            return val
        self.pending = val
        return self.last if self.age <= DISTANCE_STALE else None

    def reset(self):
        self.last    = None       # last trusted distance (m), or None
        self.age     = 999        # reads since last good (for staleness)
        self.pending = None       # a big-jump candidate awaiting confirmation

# ══════════════════════════════════════════════════════════════════════════════
# Startup checks
# ══════════════════════════════════════════════════════════════════════════════

def check_services() -> bool:
    ok = True
    for url, name in [(f"{SCREEN_HTTP}/health",  f"screen      {GAME_PC}:5003"),
                      (f"{CONTROL_HTTP}/health", f"controller  {GAME_PC}:5002"),
                      (OCR_URL.replace('/ocr', '/health'), f"ocr   {BEACON}:5001")]:
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
    arrival = ArrivalDetector(_MM_R)
    progress = ProgressDetector()
    dist_reader = DistanceReader()

    print("UMA v8.0 — checking services...")
    if not check_services():
        print("\nOne or more services unreachable.")
        sys.exit(1)

    if NAV_MODE == "vlm":
        threading.Thread(target=_nav_thread, daemon=True).start()
        print(f"\nNavigation: VLM (Qwen reads the minimap every ~{VLM_PERIOD:.1f}s)")
    else:
        print("\nNavigation: pixel near-field estimator")

    print("Obstacle avoidance: " + (f"VIEWPORT (VFH, FOV {VIEW_FOV_DEG:.0f}°, "
          f"{N_SECTORS} sectors) — calibrate VIEWPORT_REGION via viewport_annotated.png"
          if VIEWPORT_AVOID else "OFF (trail + SSIM recovery only)"))
    print("Dialogue/interact: " + ("ENABLED" if DIALOGUE_ENABLE
          else "DISABLED (no key presses on prompts/choices — DIALOGUE=1 to enable)"))

    print(f"All services OK.  Starting in 5s — W3 must be running on {GAME_PC}...\n")
    for i in range(5, 0, -1):
        print(f"  {i}", end="\r", flush=True); time.sleep(1)
    print("UMA v8.0 running.  Ctrl-C to stop.\n")

    tick          = 0
    last_mode     = ""
    post_load     = False
    combat_streak = 0
    sm_sin = sm_cos = None        # smoothed heading vector (None until first trusted frame)
    edge_bias = 0.0               # lateral bias (deg) held after a stuck, to arc around
    edge_ticks = 0
    blocked_streak = 0            # consecutive blocked ticks → live-lock detection (Fix 1)
    commit_ticks   = 0            # ticks left driving a committed gap heading
    commit_heading = 0.0          # the gap bearing we're committing to

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
            release()
            arrival.reset()        # re-evaluate arrival fresh on the next exploration entry
            progress.reset()

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
            time.sleep(2); sd.reset(); arrival.reset(); progress.reset(); dist_reader.reset(); post_load = False

        # ── Cutscene ───────────────────────────────────────────────────────
        if mode == "CUTSCENE":
            print("  🎬 waiting"); last_mode = mode; time.sleep(2); continue

        # ── Choices ────────────────────────────────────────────────────────
        if mode == "CHOICES":
            if not DIALOGUE_ENABLE:
                release()
                print("  💬 CHOICES — dialogue disabled, holding (DIALOGUE=1 to enable)")
                last_mode = mode
                time.sleep(max(0, TICK - (time.time() - t0)))
                continue
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
        # No camera reorient needed: directional control turns Geralt toward the
        # path on the next frame and the camera auto-trails.
        last_mode = mode

        sd.record(screen)
        if sd.is_stuck():
            side = sd.recover()
            edge_bias  = side * EDGE_BIAS_DEG            # veer this way for a while...
            edge_ticks = EDGE_BIAS_TICKS                 # ...to arc around the obstacle
            sm_sin = sm_cos = None                       # drop stale heading after recovery
            commit_ticks = 0                             # SSIM recovery supersedes a commit
            time.sleep(max(0, TICK - (time.time() - t0)))
            continue

        # Commit-to-gap drive (Fix 1): after a live-lock we drive the chosen gap
        # straight through at full speed for a few ticks, IGNORING the oscillating
        # trail/gold input that caused the lock. SSIM (above) still supersedes.
        if commit_ticks > 0:
            commit_ticks -= 1
            print(f"  🧭 commit {commit_heading:+.0f}° ({commit_ticks} left)  →  spd {COMMIT_SPEED:.2f}")
            move_dir(commit_heading, COMMIT_SPEED)
            time.sleep(max(0, TICK - (time.time() - t0)))
            continue

        # Hold the edge-around bias for a few ticks after a stuck so Geralt arcs
        # past the obstacle instead of re-converging straight back into it.
        bias = 0.0
        if edge_ticks > 0:
            bias = edge_bias
            edge_ticks -= 1

        # Interact prompt — gated to every Nth tick (OCR is a beacon round-trip).
        # Off when DIALOGUE_ENABLE is false: no A presses, and the OCR round-trip is
        # skipped entirely (also steadies the tick time).
        if DIALOGUE_ENABLE and tick % INTERACT_EVERY == 0:
            interact = ocr(crop(screen, INTERACT_REGION))
            if interact.strip():
                print(f"  💬 '{interact}' → A")
                release(); button("a"); time.sleep(0.4)
                continue

        # Objective distance (metres) — read-only OCR of the step counter, gated to
        # every Nth tick (it changes slowly). The validated reader holds last-good, so
        # dist_m is usable every tick for proximity decisions.
        if DISTANCE_ENABLE and tick % DISTANCE_EVERY == 0:
            dist_reader.read(screen)
            if DEBUG_EVERY and tick % DEBUG_EVERY == 0:
                crop(screen, DISTANCE_REGION).save("distance_raw.png")  # calibrate the box
        dist_m = dist_reader.last if DISTANCE_ENABLE else None

        # ── 1. Desired heading from navigation (trail or VLM) ────────────────
        if NAV_MODE == "vlm":
            heading_c, conf, age = nav_get()
            if heading_c is not None:
                decay   = math.exp(-age / TURN_TAU)     # turn toward path, then straighten
                desired = heading_c * decay + bias
                nav_str = f"vlm {heading_c:+.0f}° conf={conf} age={age:.1f}s"
                have    = True
            else:
                desired, nav_str, have = bias, "vlm waiting — fwd", False
        else:
            h_raw, n_near, conc, source = path_heading(
                screen,
                prev_heading=(math.degrees(math.atan2(sm_sin, sm_cos)) if sm_sin is not None else None),
                save=(DEBUG_EVERY and tick % DEBUG_EVERY == 0))

            # ── Arrival — stop before overshooting the destination ───────────
            # Prefer the metre distance (clean); _last_gold_dist is the pixel fallback.
            # Once arrived we release and hold; resume is handled inside the detector.
            if ARRIVAL_ENABLE and arrival.update(_last_gold_dist, n_near, dist_m):
                release()
                sm_sin = sm_cos = None                       # drop heading so we don't lurch on resume
                gd = ("NEARBY" if dist_m == 0 else
                      f"{dist_m}m" if dist_m is not None else
                      f"{_last_gold_dist:.0f}px" if _last_gold_dist is not None else "gone")
                print(f"  🎯 ARRIVED — holding ({gd}, dots {n_near})")
                last_mode = mode
                time.sleep(max(0, TICK - (time.time() - t0)))
                continue

            # ── Progress — viewport-independent stall check ──────────────────
            # Gold radial distance not closing while steering at the marker = blocked,
            # even when the viewport reports clear (it's blind to this wall). Recover
            # by reversing + sidestepping, then arc around via the edge bias.
            progress.record(_last_gold_dist, source, _MM_R)
            if progress.stalled():
                side = progress.recover()
                edge_bias, edge_ticks = side * EDGE_BIAS_DEG, EDGE_BIAS_TICKS
                sm_sin = sm_cos = None
                blocked_streak = 0
                last_mode = mode
                time.sleep(max(0, TICK - (time.time() - t0)))
                continue

            # Trust the trail ('path'), the gold bearing ('gold'), or its latch ('latch').
            trust = source in ("path", "gold", "latch")
            if trust:
                s, c = math.sin(math.radians(h_raw)), math.cos(math.radians(h_raw))
                if sm_sin is None:
                    sm_sin, sm_cos = s, c
                else:
                    sm_sin = HEADING_EMA * s + (1 - HEADING_EMA) * sm_sin
                    sm_cos = HEADING_EMA * c + (1 - HEADING_EMA) * sm_cos

            if sm_sin is not None:
                desired = math.degrees(math.atan2(sm_sin, sm_cos)) + bias
                raw_str = f"{h_raw:+.0f}" if h_raw is not None else "--"
                nav_str = (f"raw {raw_str} [{source} n={n_near} c={conc:.2f}]"
                           + ("" if trust else " hold"))
                have    = True
            else:
                desired = bias
                nav_str = f"no trail [{source} n={n_near} c={conc:.2f}] — fwd"
                have    = False

        # ── 2. Viewport obstacle avoidance — deflect toward the clearest gap ──
        # The trail says WHERE to go; the viewport says what's physically in the way
        # (poles/walls the minimap can't see). Silent when the path ahead is clear.
        # Skipped during an active stuck-recovery arc (edge_ticks) so the two don't
        # fight — the recovery sidestep is already open-loop.
        if VIEWPORT_AVOID and edge_ticks == 0:
            heading, blocked_ahead, clearance, vp_src = viewport_avoid(
                screen, desired, save=(DEBUG_EVERY and tick % DEBUG_EVERY == 0))
        else:
            heading, blocked_ahead, clearance, vp_src = desired, False, 1.0, "off"

        # ── Live-lock detection (Fix 1) — blocked tick after tick, going nowhere ─
        # Counts consecutive blocked ticks. Past LIVELOCK_TICKS we stop nibbling at
        # the obstacle and COMMIT to the widest clear gap (or, if even that's too
        # narrow, hand off to the SSIM back-off). This is what breaks the barn-style
        # oscillation that SSIM can't see because the frames keep changing.
        blocked_streak = blocked_streak + 1 if blocked_ahead else 0
        if blocked_streak >= LIVELOCK_TICKS:
            gap_ang, gap_w = viewport_widest_gap(screen)
            blocked_streak = 0
            sm_sin = sm_cos = None                       # drop the oscillating heading
            if gap_w >= COMMIT_MIN_GAP:
                commit_heading, commit_ticks = gap_ang, COMMIT_TICKS
                print(f"  🧭 LIVE-LOCK → commit gap {gap_ang:+.0f}° (w={gap_w:.0f}°) ×{COMMIT_TICKS}")
                move_dir(commit_heading, COMMIT_SPEED)
            else:
                print(f"  🧱 LIVE-LOCK, no usable gap (w={gap_w:.0f}°) — back off")
                side = sd.recover()
                edge_bias, edge_ticks = side * EDGE_BIAS_DEG, EDGE_BIAS_TICKS
            time.sleep(max(0, TICK - (time.time() - t0)))
            continue

        # ── 3. Speed + drive ─────────────────────────────────────────────────
        base = 1.0 if not have else \
            SPEED_FLOOR + (1 - SPEED_FLOOR) * max(0.0, math.cos(math.radians(heading)))
        speed = base * (SLOW_ON_BLOCK if blocked_ahead else 1.0)

        eb     = f" edge{bias:+.0f}" if bias else ""
        defl   = _ang_diff(heading, desired)
        vp_str = ("" if not VIEWPORT_AVOID else
                  f"  ⛞{vp_src} Δ{defl:+.0f}° clr={clearance:.2f}"
                  + (" BLOCK" if blocked_ahead else ""))
        dm_str = ("" if dist_m is None else
                  "  📏NEARBY" if dist_m == 0 else f"  📏{dist_m}m")
        print(f"  🗺  heading {heading:+.1f}° ({nav_str}{eb}){vp_str}{dm_str}  →  spd {speed:.2f}")
        move_dir(heading, speed)

        elapsed = time.time() - t0
        print(f"  ⏱  {int(elapsed * 1000)}ms")
        time.sleep(max(0, TICK - elapsed))

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        release()
        print("\n\nUMA stopped. Controller released.")
