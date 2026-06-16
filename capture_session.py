#!/usr/bin/env python3
"""
UMA capture — log an Xbox-pad gameplay session for imitation learning.

Design rule: LOG RAW, DERIVE LATER.
  • The VIDEO is the frame store (recorded by a separate encoder — see --recorder).
  • This script logs the CONTROLLER state + timestamps, and writes a session MANIFEST.
  • Frames and inputs are aligned OFFLINE using a shared CLOCK_MONOTONIC origin
    (t0_monotonic in the manifest) plus a one-time visual SYNC gesture you perform at
    the start, so you can eyeball-verify the offset in the recording.

Why evdev for the controller: the kernel delivers analog-stick events as absolute axes
with sub-ms latency, on the same machine clock we anchor the recorder to — no
inverse-dynamics guessing, no reading a burned-in overlay back out of pixels. We stamp
each row with our own CLOCK_MONOTONIC read at the moment we receive the event.

Modes
  --peek            Live-dump the pad's axes/buttons (no files). Use this FIRST to
                    confirm you're reading the PHYSICAL pad (Steam Input not swallowing
                    it) and to eyeball that the sticks move 1:1. This is also the
                    bench-test tool for the laptop-vs-desktop capture decision.
  (default)         Full session: manifest + optional recorder + controller JSONL log.

Output (per session dir):
  manifest.json     resolution/fps/encoder, t0_monotonic & t0_wall, device + axis caps,
                    label, sync note — everything needed to align + reproduce.
  controller.jsonl  one row per input CHANGE, plus a keyframe every --keyframe-ms so
                    long still periods still have rows (trivial to resample offline).
  video.<ext>       written by the recorder (or by your external OBS, see --recorder).
"""

import argparse, json, os, signal, socket, subprocess, sys, time
from datetime import datetime

try:
    from evdev import InputDevice, list_devices, ecodes, AbsInfo
except ImportError:
    sys.exit("python-evdev not installed.  pip install evdev  (or apt install python3-evdev)")

CLOCK = time.CLOCK_MONOTONIC
def mono() -> float:
    return time.clock_gettime(CLOCK)

def btn_name(code) -> str:
    """A clean button name. evdev maps some codes to MULTIPLE names, returning a
    list/tuple (e.g. BTN_SOUTH/BTN_A) — take the first and always return a string."""
    n = ecodes.BTN.get(code) or ecodes.KEY.get(code) or str(code)
    if isinstance(n, (list, tuple)):
        n = n[0]
    return str(n)

# Axis evdev codes we care about (Xbox via xpad/xone). Triggers are 0..max; sticks signed.
STICK_AXES   = {"lx": ecodes.ABS_X,  "ly": ecodes.ABS_Y,
                "rx": ecodes.ABS_RX, "ry": ecodes.ABS_RY}
TRIGGER_AXES = {"lt": ecodes.ABS_Z,  "rt": ecodes.ABS_RZ}
HAT_AXES     = {"hx": ecodes.ABS_HAT0X, "hy": ecodes.ABS_HAT0Y}


# ── device discovery ────────────────────────────────────────────────────────────────
def find_pad(path: str | None) -> InputDevice:
    if path:
        return InputDevice(path)
    candidates = []
    for p in list_devices():
        try:
            d = InputDevice(p)
        except Exception:
            continue
        caps = d.capabilities()
        abs_codes = {c for c, _ in caps.get(ecodes.EV_ABS, [])}
        key_codes = set(caps.get(ecodes.EV_KEY, []))
        is_pad = ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes and (
            ecodes.BTN_GAMEPAD in key_codes or ecodes.BTN_SOUTH in key_codes)
        if is_pad:
            candidates.append(d)
    if not candidates:
        sys.exit("No gamepad found under /dev/input/event*.\n"
                 "  • Is the pad connected and are you in the 'input' group?\n"
                 "    (sudo usermod -aG input $USER, then re-login)\n"
                 "  • If Steam is running, it may expose a *virtual* pad — disable Steam\n"
                 "    Input for the game so the physical device is what's read.\n"
                 "  • List devices:  python3 -c \"from evdev import list_devices,InputDevice as I;"
                 "[print(p, I(p).name) for p in list_devices()]\"")
    if len(candidates) > 1:
        print("Multiple gamepads found — using the first. Override with --device:")
        for d in candidates:
            print(f"    {d.path}  {d.name}")
    return candidates[0]


def axis_caps(dev: InputDevice) -> dict:
    """Record absinfo (min/max/flat) per axis so normalized values are reproducible."""
    caps = {}
    info = {c: ai for c, ai in dev.capabilities().get(ecodes.EV_ABS, [])}
    for name, code in {**STICK_AXES, **TRIGGER_AXES, **HAT_AXES}.items():
        ai = info.get(code)
        if ai is not None:
            caps[name] = {"code": code, "min": ai.min, "max": ai.max,
                          "flat": ai.flat, "res": ai.resolution}
    return caps


def norm_stick(v, ai):   # → [-1, 1], no deadzone (kept faithful; apply deadzone at train time)
    span = (ai["max"] - ai["min"]) or 1
    return max(-1.0, min(1.0, 2.0 * (v - ai["min"]) / span - 1.0))

def norm_trig(v, ai):    # → [0, 1]
    span = (ai["max"] - ai["min"]) or 1
    return max(0.0, min(1.0, (v - ai["min"]) / span))


# ── peek mode (bench test / sync verification) ───────────────────────────────────────
def peek(dev: InputDevice, caps: dict, seconds: float):
    print(f"PEEK  {dev.path}  {dev.name}\n"
          f"Move the sticks/triggers; press buttons. Ctrl-C to stop "
          f"({'∞' if seconds <= 0 else f'{seconds:.0f}s'}).\n")
    state = {k: 0.0 for k in (*STICK_AXES, *TRIGGER_AXES)}
    pressed: set[str] = set()
    t_end = None if seconds <= 0 else mono() + seconds
    last_print = 0.0
    try:
        while t_end is None or mono() < t_end:
            r, _, _ = __import__("select").select([dev.fd], [], [], 0.05)
            if r:
                for e in dev.read():
                    if e.type == ecodes.EV_ABS:
                        for k, code in STICK_AXES.items():
                            if e.code == code and k in caps: state[k] = norm_stick(e.value, caps[k])
                        for k, code in TRIGGER_AXES.items():
                            if e.code == code and k in caps: state[k] = norm_trig(e.value, caps[k])
                    elif e.type == ecodes.EV_KEY:
                        name = btn_name(e.code)
                        (pressed.add if e.value else pressed.discard)(name)
            now = mono()
            if now - last_print > 0.08:
                last_print = now
                btns = " ".join(sorted(pressed)) or "—"
                print(f"\rL({state['lx']:+.2f},{state['ly']:+.2f}) "
                      f"R({state['rx']:+.2f},{state['ry']:+.2f}) "
                      f"LT {state['lt']:.2f} RT {state['rt']:.2f}  [{btns}]      ",
                      end="", flush=True)
    except KeyboardInterrupt:
        pass
    print("\npeek done.")


# ── recorder launch ──────────────────────────────────────────────────────────────────
def gsr_monitor() -> str | None:
    """Auto-pick a monitor name from `gpu-screen-recorder --list-capture-options`.
    On AMD/Wayland gsr can ONLY capture monitors (not individual windows), so we capture
    the monitor the game is on — specifying it by name skips the portal picker. If W3
    runs fullscreen/borderless on that monitor, this captures exactly the game."""
    try:
        out = subprocess.run(["gpu-screen-recorder", "--list-capture-options"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    methods = {"portal", "screen", "screen-direct", "focused", "window", "region"}
    mons = []
    for line in out.splitlines():
        tok = line.strip().replace("|", " ").split()
        if not tok:
            continue
        name = tok[0]
        if name in methods:
            continue
        if "-" in name or name.upper().startswith(("EDP", "DP", "HDMI", "DVI", "VGA", "DSI")):
            mons.append(name)
    return mons[0] if mons else None


def start_recorder(kind: str, out_video: str, fps: int, window: str, size: str, audio: str):
    """Returns a Popen (or None for external/OBS). Best-effort: logging works regardless."""
    if kind == "external":
        print("\nRECORDER = external: start your OBS/recording NOW (VAAPI/h264), then\n"
              "             press Enter. (t0 is set on Enter — keep the sync gesture below.)")
        input("  ↵ when recording is rolling… ")
        return None

    if kind == "gsr":
        win = window
        if win == "auto":
            mon = gsr_monitor()
            if mon:
                win = mon
                print(f"  gsr: auto-selected monitor '{win}' (no picker)")
            else:
                win = "portal"
                print("  gsr: couldn't auto-detect a monitor — falling back to portal picker.\n"
                      "       (run `gpu-screen-recorder --list-capture-options` and pass\n"
                      "        --window <monitor-name> to skip the picker next time.)")
        cmd = ["gpu-screen-recorder", "-w", win, "-f", str(fps), "-c", "mp4",
               "-q", "very_high", "-o", out_video]
        if audio and audio.lower() != "none":
            cmd[-2:-2] = ["-a", audio]      # insert '-a <device>' before '-o video'
            print(f"  gsr: capturing audio '{audio}' "
                  f"(navigation ignores it; kept for review / future combat work)")
    elif kind == "ffmpeg":
        # X11 / XWayland path (Proton games run under XWayland, so :0 often sees them).
        cmd = ["ffmpeg", "-y", "-f", "x11grab", "-framerate", str(fps),
               "-video_size", size, "-i", ":0.0",
               "-vaapi_device", "/dev/dri/renderD128",
               "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-qp", "23",
               out_video]
    else:
        sys.exit(f"unknown recorder {kind!r}")

    print(f"\nRECORDER = {kind}:  {' '.join(cmd)}")
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        sys.exit(f"'{cmd[0]}' not found. Install it, or use --recorder external (OBS).")
    time.sleep(1.5)   # let the encoder spin up before we stamp t0 / start logging
    if p.poll() is not None:
        sys.exit(f"recorder exited immediately (code {p.returncode}). "
                 f"On KDE Wayland try --recorder gsr --window portal, or --recorder external.")
    return p


# ── full session ─────────────────────────────────────────────────────────────────────
def session(args):
    dev  = find_pad(args.device)
    caps = axis_caps(dev)
    os.makedirs(args.out, exist_ok=True)
    ext = "mp4"
    out_video = os.path.join(args.out, f"video.{ext}")
    out_log   = os.path.join(args.out, "controller.jsonl")
    out_man   = os.path.join(args.out, "manifest.json")

    print(f"DEVICE  {dev.path}  {dev.name}")
    print(f"AXES    {', '.join(caps)}")
    print(f"OUT     {args.out}")

    rec = start_recorder(args.recorder, out_video, args.fps, args.window, args.size, args.audio)

    # t0 — the shared monotonic origin. Everything (video PTS, every log row) maps to this.
    t0_mono, t0_wall = mono(), time.time()

    tag_code = getattr(ecodes, args.tag_button, None)
    manifest = {
        "schema": 1,
        "created": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "label": args.label,
        "recorder": args.recorder,
        "video": os.path.basename(out_video) if rec is not None or args.recorder == "external" else None,
        "fps": args.fps,
        "audio": args.audio,
        "size_hint": args.size,
        "t0_monotonic": t0_mono,          # CLOCK_MONOTONIC at session start
        "t0_wall": t0_wall,               # epoch seconds, for human reference
        "device": {"path": dev.path, "name": dev.name},
        "axis_caps": caps,                # absinfo per axis → reversible normalization
        "stick_axes": list(STICK_AXES), "trigger_axes": list(TRIGGER_AXES),
        "hat_axes": list(HAT_AXES),
        "tag_button": args.tag_button,
        "keyframe_ms": args.keyframe_ms,
        "notes": "Rows: t_rel = mono - t0_monotonic. Align video frame PTS to t_rel. "
                 "A SYNC gesture (full stick deflection ~1s) is logged at the start so "
                 "the recorder offset can be verified by eye.",
    }
    with open(out_man, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"MANIFEST written ({out_man})")

    # state + writer
    state = {k: 0.0 for k in (*STICK_AXES, *TRIGGER_AXES, *HAT_AXES)}
    pressed: set[str] = set()
    tag_held = False
    log = open(out_log, "w", buffering=1)   # line-buffered: never lose rows on a hard stop
    rows = 0

    def write_row(t):
        nonlocal rows
        log.write(json.dumps({
            "t": round(t - t0_mono, 4),
            **{k: round(state[k], 4) for k in STICK_AXES},
            **{k: round(state[k], 4) for k in TRIGGER_AXES},
            "hx": state["hx"], "hy": state["hy"],
            "btn": sorted(pressed),
            "tag": tag_held,
        }) + "\n")
        rows += 1

    stop = {"v": False}
    signal.signal(signal.SIGINT,  lambda *a: stop.update(v=True))
    signal.signal(signal.SIGTERM, lambda *a: stop.update(v=True))

    print("\n>>> SYNC: hold the LEFT STICK fully in one direction for ~1s, then release.")
    print(">>> Then play. Hold the tag button to mark RECOVERY/odd-angle demos.")
    print(">>> Ctrl-C to end the session.\n")

    import select
    kf = args.keyframe_ms / 1000.0
    last_kf = mono()
    write_row(mono())                       # initial row
    try:
        while not stop["v"]:
            r, _, _ = select.select([dev.fd], [], [], kf)
            now = mono()
            changed = False
            if r:
                try:
                    for e in dev.read():
                        if e.type == ecodes.EV_ABS:
                            for k, code in STICK_AXES.items():
                                if e.code == code and k in caps:
                                    state[k] = norm_stick(e.value, caps[k]); changed = True
                            for k, code in TRIGGER_AXES.items():
                                if e.code == code and k in caps:
                                    state[k] = norm_trig(e.value, caps[k]); changed = True
                            for k, code in HAT_AXES.items():
                                if e.code == code:
                                    state[k] = e.value; changed = True
                        elif e.type == ecodes.EV_KEY:
                            nm = btn_name(e.code)
                            (pressed.add if e.value else pressed.discard)(nm)
                            if tag_code is not None and e.code == tag_code:
                                tag_held = bool(e.value)
                                print(f"  🔖 tag {'ON ' if tag_held else 'OFF'}  "
                                      f"(t={now - t0_mono:7.2f}s, rows={rows})")
                            changed = True
                except BlockingIOError:
                    pass
            if changed:
                write_row(now); last_kf = now
            elif now - last_kf >= kf:        # keyframe during still periods
                write_row(now); last_kf = now
    finally:
        write_row(mono())
        log.close()
        if rec is not None:
            rec.send_signal(signal.SIGINT)   # let the encoder finalize the mp4 moov atom
            try: rec.wait(timeout=8)
            except subprocess.TimeoutExpired: rec.terminate()
        dur = mono() - t0_mono
        # append a footer to the manifest
        manifest["ended"] = datetime.now().astimezone().isoformat()
        manifest["duration_s"] = round(dur, 2)
        manifest["rows"] = rows
        with open(out_man, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nSESSION done: {rows} rows over {dur:.1f}s  →  {args.out}")
        print("Next: verify the recording — check in-game FPS (MangoHud) and that the\n"
              "encoder reported NO dropped frames. If the iGPU couldn't hold W3+encode,\n"
              "move capture to b365/b550 (discrete GPU / NVENC).")


def main():
    ap = argparse.ArgumentParser(description="UMA gameplay + controller capture")
    ap.add_argument("--peek", action="store_true", help="live-dump pad state; no files")
    ap.add_argument("--peek-seconds", type=float, default=0, help="0 = until Ctrl-C")
    ap.add_argument("--device", help="evdev path, e.g. /dev/input/event20 (default: autodetect)")
    ap.add_argument("--out", default=os.path.join(
        "/home/danja/pro/uma-agent/sessions", f"{datetime.now():%Y%m%d_%H%M%S}"),
        help="session output dir (default: /home/danja/pro/uma-agent/sessions/<timestamp>)")
    ap.add_argument("--recorder", choices=["gsr", "ffmpeg", "external"], default="gsr",
                    help="gsr=gpu-screen-recorder (default), ffmpeg=x11grab+VAAPI, external=OBS")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--audio", default="default_output",
                    help="gsr audio source (default 'default_output' = game sound; "
                         "'none' to disable). Navigation ignores audio, but it's kept for "
                         "session review and future combat/state work. "
                         "List devices: gpu-screen-recorder --list-audio-devices")
    ap.add_argument("--window", default="portal",
                    help="gsr target. 'portal' (default) opens the picker once so you can "
                         "select the W3 WINDOW (captures it at its own 1080p res, lets you "
                         "alt-tab freely). Or a monitor name, or 'auto' to auto-pick a monitor.")
    ap.add_argument("--size", default="1920x1080", help="ffmpeg x11grab capture size")
    ap.add_argument("--keyframe-ms", type=int, default=100, help="row cadence during stick-still periods")
    ap.add_argument("--tag-button", default="BTN_MODE",
                    help="evdev button to mark recovery demos (default BTN_MODE = Xbox guide, "
                         "inert in W3/Steam). Hold it during recovery/odd-angle demos.")
    ap.add_argument("--label", default="", help="free-text note stored in the manifest")
    args = ap.parse_args()

    if args.peek:
        dev = find_pad(args.device)
        peek(dev, axis_caps(dev), args.peek_seconds)
    else:
        session(args)


if __name__ == "__main__":
    main()
