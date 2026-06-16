#!/usr/bin/env python3
"""
align_extract.py — turn a capture session (video + controller log) into a BC dataset.

Pipeline:
  1. AUTO-ALIGN: cross-correlate world-motion (from the video) with stick-speed (from the
     log) to recover the per-session time offset (session_t = video_t + offset). This is
     robust to the variable portal-picker delay — no hand-reading the sync slam. Reports
     the correlation; a weak peak (<MIN_CORR) warns that alignment is unreliable.
  2. WALK frames at --sample-fps; for each, session_t = video_t + offset, look up the
     HELD controller state, and run uma8.detect_mode. Keep only EXPLORATION (drop
     loading / cutscene / combat / menu), excluding a margin around mode transitions.
  3. PERCEPTION FEATURES via the uma8 parsers (extract_dots, find_gold_marker) — the raw
     things the agent SEES (dot geometry, gold bearing). NOT path_heading: that's the
     rule policy's decision; feeding it would clone the rule agent instead of learning
     from the human. Label = the human's left stick (lx, ly).
  4. SAVE an .npz (features X, stick Y, aux) + print a coverage report.

Usage:
  python3 align_extract.py --session /path/to/session_dir [--uma8-path /path/to/uma8_dir]
  (expects video.mp4 / controller.jsonl / manifest.json in the session dir)
"""
import argparse, json, math, os, sys
import numpy as np

# ── feature layout (documented, fixed size) ─────────────────────────────────────────
DOT_BINS = 12                      # angular bins over [-180,180]; minimap is rotation-
                                   # relative (up = forward), so bin 0-area = "ahead"
FEATURE_NAMES = (
    [f"near_b{i}" for i in range(DOT_BINS)] +     # near-ring dot histogram
    [f"far_b{i}"  for i in range(DOT_BINS)] +     # far-ring dot histogram
    ["n_near", "n_far",
     "has_gold", "gold_sin", "gold_cos", "gold_dist",
     "near_sin", "near_cos", "near_dist"]
)
N_FEATURES = len(FEATURE_NAMES)


def build_features(uma8, im):
    """(mode, feature_vector, raw_dict) for one full-frame PIL image."""
    mode = uma8.detect_mode(im)
    dots, origin, _ = uma8.extract_dots(im)
    mx, my, mw, mh = uma8.MINIMAP_REGION
    mm = np.array(im.crop((mx, my, mx + mw, my + mh)))
    gold_ang, gold_pt, gold_px = uma8.find_gold_marker(mm, origin)

    R = uma8._MM_R
    near = np.zeros(DOT_BINS, np.float32)
    far  = np.zeros(DOT_BINS, np.float32)
    n_near = n_far = 0
    nearest = None
    for d in dots:
        dd = math.hypot(d[0] - origin[0], d[1] - origin[1])
        if dd <= uma8.DOT_EXCLUDE_R:
            continue
        ang = uma8._ang(origin, d)                 # deg, up=0, +=right
        b = int((ang + 180.0) / 360.0 * DOT_BINS) % DOT_BINS
        if dd <= uma8.R_NEAR:
            near[b] += 1.0; n_near += 1
            if nearest is None or dd < nearest[0]:
                nearest = (dd, ang)
        else:
            far[b] += 1.0; n_far += 1

    has_gold = 1.0 if gold_ang is not None else 0.0
    g = math.radians(gold_ang) if gold_ang is not None else 0.0
    gold_dist = (gold_px / R) if gold_px else 0.0
    if nearest is not None:
        na = math.radians(nearest[1]); nd = nearest[0] / R
        near_sin, near_cos, near_dist = math.sin(na), math.cos(na), nd
    else:
        near_sin = near_cos = near_dist = 0.0

    feat = np.concatenate([
        near, far,
        np.array([n_near, n_far, has_gold, math.sin(g), math.cos(g), gold_dist,
                  near_sin, near_cos, near_dist], np.float32)]).astype(np.float32)
    raw = {"n_dots": len(dots), "gold_ang": gold_ang, "gold_px": float(gold_px or 0)}
    return mode, feat, raw


# ── controller log → held-state lookup ───────────────────────────────────────────────
def load_controller(path):
    rows = [json.loads(l) for l in open(path)]
    t = np.array([r["t"] for r in rows], np.float64)
    cols = {k: np.array([r.get(k, 0.0) for r in rows], np.float32)
            for k in ("lx", "ly", "rx", "ry", "lt", "rt")}
    tag = np.array([1 if r.get("tag") else 0 for r in rows], np.int8)
    return t, cols, tag, rows


def held_at(t_axis, arr, stime):
    """Step-hold lookup: the controller holds its value between rows."""
    idx = np.clip(np.searchsorted(t_axis, stime, side="right") - 1, 0, len(t_axis) - 1)
    return arr[idx]


# ── auto-align via cross-correlation ─────────────────────────────────────────────────
def video_motion(cap, cv2, hz=10.0, max_seconds=None):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / hz)))
    prev = None; D = []; VT = []; fi = 0
    while True:
        if not cap.grab():
            break
        if fi % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[200:900, 350:1150].astype(np.float32)
            if prev is not None:
                D.append(np.abs(g - prev).mean()); VT.append(fi / fps)
            prev = g
        fi += 1
        if max_seconds and fi / fps > max_seconds:
            break
    return np.array(VT), np.array(D), fps


def find_offset(VT, D, t_axis, spd, lo=-6.0, hi=6.0, dstep=0.05):
    Dz = (D - D.mean()) / (D.std() + 1e-9)
    best = (0.0, -1.0)
    for off in np.arange(lo, hi + 1e-9, dstep):
        s = held_at(t_axis, spd, VT + off)
        if s.std() < 1e-6:
            continue
        sz = (s - s.mean()) / (s.std() + 1e-9)
        c = float((Dz * sz).mean())
        if c > best[1]:
            best = (float(off), c)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="dir with video.mp4/controller.jsonl/manifest.json")
    ap.add_argument("--video", default=None); ap.add_argument("--controller", default=None)
    ap.add_argument("--manifest", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--uma8-path", default=None, help="dir containing uma8.py (for the parsers)")
    ap.add_argument("--sample-fps", type=float, default=10.0, help="frames/sec to extract (walking is slow; 10 is plenty)")
    ap.add_argument("--offset", type=float, default=None, help="override auto-aligned offset (s)")
    ap.add_argument("--transition-margin", type=float, default=0.4, help="drop frames within this of a mode change")
    ap.add_argument("--min-corr", type=float, default=0.4, help="warn if alignment correlation is below this")
    args = ap.parse_args()

    sess = args.session
    video = args.video or os.path.join(sess, "video.mp4")
    if not os.path.exists(video):
        video = os.path.join(sess, "head.mp4")
    ctrl  = args.controller or os.path.join(sess, "controller.jsonl")
    manf  = args.manifest or os.path.join(sess, "manifest.json")
    out   = args.out or os.path.join(sess, "dataset.npz")

    if args.uma8_path:
        sys.path.insert(0, args.uma8_path)
    import cv2
    import uma8
    uma8.ocr = lambda *a, **k: ""        # offline: no beacon. detect_mode's CHOICES OCR-
                                         # confirm just fails (rare on exploration frames).

    manifest = json.load(open(manf))
    print(f"session : {sess}")
    print(f"video   : {os.path.basename(video)}   fps(manifest)={manifest.get('fps')}")
    t_axis, cols, tag, _ = load_controller(ctrl)
    spd = np.sqrt(cols["lx"] ** 2 + cols["ly"] ** 2)
    print(f"controller rows: {len(t_axis)}  ({t_axis[0]:.2f}..{t_axis[-1]:.2f}s)")

    # 1) alignment
    if args.offset is not None:
        offset = args.offset; corr = float("nan")
        print(f"offset  : {offset:+.2f}s (manual override)")
    else:
        cap = cv2.VideoCapture(video)
        VT, D, _ = video_motion(cap, cv2, hz=10.0)
        cap.release()
        offset, corr = find_offset(VT, D, t_axis, spd)
        print(f"offset  : {offset:+.2f}s  (cross-corr {corr:.3f})  → session_t = video_t + offset")
        if corr < args.min_corr:
            print(f"  ⚠  weak alignment (corr<{args.min_corr}). Treat this dataset with caution;\n"
                  f"     next session do a SHARP left-stick slam for a crisp anchor.")

    # 2+3) walk frames, segment, extract
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / args.sample_fps)))
    from PIL import Image

    rec = []   # (video_t, session_t, mode, feat, lx, ly, rx, ry, lt, rt, tag, n_dots)
    mode_counts = {}
    fi = 0
    while True:
        if not cap.grab():
            break
        if fi % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                vt = fi / fps
                st = vt + offset
                im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                mode, feat, raw = build_features(uma8, im)
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                rec.append((vt, st, mode, feat,
                            float(held_at(t_axis, cols["lx"], st)),
                            float(held_at(t_axis, cols["ly"], st)),
                            float(held_at(t_axis, cols["rx"], st)),
                            float(held_at(t_axis, cols["ry"], st)),
                            float(held_at(t_axis, cols["lt"], st)),
                            float(held_at(t_axis, cols["rt"], st)),
                            int(held_at(t_axis, tag, st)),
                            raw["n_dots"]))
        fi += 1
    cap.release()

    # transition margin: drop EXPLORATION frames too close to a non-EXPLORATION frame
    modes = [r[2] for r in rec]
    keep = []
    for i, r in enumerate(rec):
        if r[2] != "EXPLORATION":
            continue
        st = r[1]
        bad = any(abs(rec[j][1] - st) <= args.transition_margin and rec[j][2] != "EXPLORATION"
                  for j in range(max(0, i - 12), min(len(rec), i + 13)))
        if not bad:
            keep.append(r)

    if not keep:
        print("\nNo EXPLORATION frames survived — check alignment / regions.")
        return

    X   = np.stack([r[3] for r in keep]).astype(np.float32)
    Y   = np.array([[r[4], r[5]] for r in keep], np.float32)     # left stick (label)
    aux = np.array([[r[6], r[7], r[8], r[9], r[10], r[11]] for r in keep], np.float32)  # rx,ry,lt,rt,tag,n_dots
    vts = np.array([r[0] for r in keep], np.float32)
    sts = np.array([r[1] for r in keep], np.float32)

    np.savez_compressed(out, X=X, Y=Y, aux=aux, video_t=vts, session_t=sts,
                        feature_names=np.array(FEATURE_NAMES),
                        aux_names=np.array(["rx", "ry", "lt", "rt", "tag", "n_dots"]),
                        offset=np.float32(offset), corr=np.float32(corr if 'corr' in dir() else 0))

    # 4) report
    total = len(rec)
    print(f"\n── coverage (sampled {total} frames @ {args.sample_fps:.0f}fps) ─────────────")
    for m, c in sorted(mode_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {m:12s} {c:5d}  {100*c/total:5.1f}%")
    print(f"  → EXPLORATION kept after transition margin: {len(keep)}")
    moving = int((np.abs(Y).max(axis=1) > 0.2).sum())
    print(f"  → of those, frames with stick pushed (>0.2): {moving} ({100*moving/len(keep):.0f}%)")
    tagged = int(aux[:, 4].sum())
    print(f"  → tagged (recovery) frames: {tagged}")

    # steering-density gauge. You steer mostly with lx (left-stick X), with rx (camera) as an
    # occasional reinforcement — so measure the COMBINED steer, not rx alone (which hides
    # turn-rich sessions). steer = lx + rx-when-reinforcing.
    lx = Y[:, 0]; rx = aux[:, 0]; ly = Y[:, 1]
    moving = np.abs(ly) > 0.2
    oppose = (np.abs(lx) > 0.15) & (np.abs(rx) > 0.15) & (np.sign(lx) != np.sign(rx))
    steer = np.clip(lx + np.where(oppose, 0.0, rx), -1.0, 1.0)
    s = steer[moving] if moving.any() else steer
    turn_frac = float((np.abs(s) > 0.3).mean())
    bar = "█" * int(turn_frac * 40)
    print(f"\n── steering density (combined lx+rx, on moving frames) ──────")
    print(f"  real turns |steer|>0.3 : {int((np.abs(s)>0.3).sum()):5d} / {len(s)}  ({100*turn_frac:.1f}%)  {bar}")
    print(f"  steer std {s.std():.2f}   |   lx-turns {100*np.mean(np.abs(lx[moving])>0.3):.0f}%  "
          f"rx-turns {100*np.mean(np.abs(rx[moving])>0.3):.0f}%  (lx is the main wheel)")
    print(f"  rx-opposing (scan) frames dropped from steer: {int(oppose.sum())}")
    if turn_frac < 0.15:
        print(f"  ⚠  turn-poor (<15%) — capture winding terrain and corner more decisively.")
    else:
        print(f"  ✓  turn-rich ({100*turn_frac:.0f}% ≥ 15%) — good steering probe.")

    print(f"\nsaved {out}  —  X{X.shape} (features), Y{Y.shape} (left stick)")
    print(f"label balance: lx[{Y[:,0].min():+.2f},{Y[:,0].max():+.2f}]  ly[{Y[:,1].min():+.2f},{Y[:,1].max():+.2f}]")


if __name__ == "__main__":
    main()
