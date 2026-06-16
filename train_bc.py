#!/usr/bin/env python3
"""
train_bc.py — behavioral cloning of the left stick from perception features.

Trains a small MLP: 33 perception features (dot histogram + gold bearing, from
align_extract.py) → left-stick (lx, ly). This is the first learned navigator: it
imitates how YOU steered given what the agent sees.

Key choices:
  • TIME-BLOCK split, not random. Adjacent frames are near-duplicates; a random split
    leaks them across train/val and inflates the score. We hold out whole contiguous
    time blocks so the val number is an honest generalization estimate.
  • tanh output (stick ∈ [-1,1]); weighted MSE so tagged recovery frames can be upweighted.
  • Reports held-out R²/corr per axis, plus turn-direction accuracy (does it steer the
    correct way on real turns), against a predict-the-mean baseline. Saves model + a
    predicted-vs-actual plot.

Usage:
  python3 train_bc.py --data session_dir/dataset.npz [--epochs 300] [--tag-weight 3]
"""
import argparse, json, os
import numpy as np
import torch, torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, hidden=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 2), nn.Tanh())          # bound to stick range

    def forward(self, x):
        return self.net(x)


def time_block_split(session_t, n_blocks=25, val_every=5):
    """Hold out whole contiguous time blocks (no adjacent-frame leakage)."""
    order = np.argsort(session_t)
    blocks = np.array_split(order, n_blocks)
    tr, va = [], []
    for i, b in enumerate(blocks):
        (va if i % val_every == 0 else tr).append(b)
    return np.concatenate(tr), np.concatenate(va)


def metrics(pred, true, names, steer_idx):
    out = {}
    for j, name in enumerate(names):
        p, t = pred[:, j], true[:, j]
        ss_res = ((t - p) ** 2).sum(); ss_tot = ((t - t.mean()) ** 2).sum() + 1e-9
        r2 = 1 - ss_res / ss_tot
        cc = np.corrcoef(p, t)[0, 1] if (p.std() > 1e-9 and t.std() > 1e-9) else 0.0
        out[name] = (float(r2), float(cc))
    # steer-direction accuracy: on real turns (|steer|>0.3), is the sign right?
    s = steer_idx
    m = np.abs(true[:, s]) > 0.3
    turn_acc = float((np.sign(pred[m, s]) == np.sign(true[m, s])).mean()) if m.any() else float("nan")
    return out, turn_acc, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.npz from align_extract.py")
    ap.add_argument("--target", choices=["steer_combined", "throttle_steer", "lxly"],
                    default="steer_combined",
                    help="what to predict. 'steer_combined' (default): throttle=ly, steer=lx+rx "
                         "(rx kept only when it reinforces lx). 'throttle_steer': ly + rx only. "
                         "'lxly': raw left stick.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--history", type=int, default=1,
                    help="stack the last N frames of features (temporal context). N=1 is single-"
                         "frame (default). Built from existing dataset.npz using session_t contiguity "
                         "— no re-extraction needed. Steering likely needs N>1.")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag-weight", type=float, default=3.0, help="loss weight on tagged recovery frames")
    ap.add_argument("--steer-weight", type=float, default=0.0,
                    help="upweight the steer-axis loss by (1 + k*|steer|) so rare turns aren't "
                         "drowned out by abundant go-straight frames. k=0 is off; try 3, 8.")
    ap.add_argument("--out", default=None, help="model output path (default beside data)")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
    aux = d["aux"]; aux_names = list(d["aux_names"])
    tag = aux[:, aux_names.index("tag")] if "tag" in aux_names else np.zeros(len(X))
    st = d["session_t"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # choose what the model predicts
    if args.target == "lxly":
        T = Y.astype(np.float32); tnames = ["lx", "ly"]; steer_idx = 0
    elif args.target == "throttle_steer":      # old: right-stick steer only
        rx = aux[:, aux_names.index("rx")].astype(np.float32)
        T = np.stack([Y[:, 1], rx], axis=1).astype(np.float32)
        tnames = ["throttle(ly)", "steer(rx)"]; steer_idx = 1
    else:  # steer_combined (default): steer = lx (dominant) + rx when it REINFORCES lx;
           # drop rx when it opposes lx (camera scanning, not steering). throttle = ly.
        lx = Y[:, 0].astype(np.float32); rx = aux[:, aux_names.index("rx")].astype(np.float32)
        oppose = (np.abs(lx) > 0.15) & (np.abs(rx) > 0.15) & (np.sign(lx) != np.sign(rx))
        rx_eff = np.where(oppose, 0.0, rx)
        steer = np.clip(lx + rx_eff, -1.0, 1.0).astype(np.float32)
        T = np.stack([Y[:, 1], steer], axis=1).astype(np.float32)
        tnames = ["throttle(ly)", "steer(lx+rx)"]; steer_idx = 1
        print(f"  steer_combined: dropped rx on {int(oppose.sum())} opposing (scan) frames")
    print(f"data: X{X.shape} target={args.target} {tnames}  tagged={int(tag.sum())}  device={dev}")
    print(f"target balance: {tnames[0]}[{T[:,0].min():+.2f},{T[:,0].max():+.2f}]  "
          f"{tnames[1]}[{T[:,1].min():+.2f},{T[:,1].max():+.2f}]")

    # temporal context: stack last N frames' features (built from existing data; a window is
    # only valid if its frames are contiguous in session_t, so we don't stack across the gaps
    # left by dropped cutscene/combat/menu segments).
    N = max(1, args.history)
    if N > 1:
        order = np.argsort(st)
        Xo, sto = X[order], st[order]
        period = np.median(np.diff(sto))           # ≈ 1/sample_fps
        dt_max = period * 1.5
        rows, keep_idx = [], []
        for i in range(N - 1, len(Xo)):
            if np.all(np.diff(sto[i - N + 1:i + 1]) < dt_max):
                rows.append(Xo[i - N + 1:i + 1].reshape(-1))
                keep_idx.append(order[i])
        keep_idx = np.array(keep_idx)
        X = np.array(rows, np.float32); T = T[keep_idx]; tag = tag[keep_idx]; st = st[keep_idx]
        print(f"history N={N}: stacked features → X{X.shape}  "
              f"({len(keep_idx)} frames had {N} contiguous predecessors)")

    tr, va = time_block_split(st)
    # show the class imbalance that makes steering hard to learn
    frac_turn = float((np.abs(T[:, steer_idx]) > 0.3).mean())
    print(f"steer imbalance: only {100*frac_turn:.1f}% of frames are real turns "
          f"(|{tnames[steer_idx]}|>0.3) — equal-weight MSE can score well by predicting ~0.")
    # standardize on TRAIN only; save params for inference parity
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xtr = torch.tensor(Xn[tr]); Ytr = torch.tensor(T[tr])
    Xva = torch.tensor(Xn[va]); Yva = torch.tensor(T[va])
    # per-(frame,axis) loss weights: tag upweight on all axes; steer-magnitude upweight on
    # the steer axis only, so rare turns dominate the steering loss without distorting throttle.
    base = np.where(tag[tr] > 0, args.tag_weight, 1.0).astype(np.float32)
    W = np.repeat(base[:, None], 2, axis=1)
    W[:, steer_idx] *= (1.0 + args.steer_weight * np.abs(T[tr, steer_idx]))
    wtr = torch.tensor(W.astype(np.float32))
    print(f"split: train {len(tr)}  val {len(va)} (time-block, no leakage)  "
          f"steer-weight k={args.steer_weight}")

    model = MLP(X.shape[1], args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xtr, Ytr, wtr, Xva, Yva = (t.to(dev) for t in (Xtr, Ytr, wtr, Xva, Yva))

    best = (1e9, None); hist = []
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        pred = model(Xtr)
        loss = (wtr * (pred - Ytr) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vloss = ((model(Xva) - Yva) ** 2).mean().item()
        hist.append((loss.item(), vloss))
        if vloss < best[0]:
            best = (vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()})
        if ep % 50 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:4d}  train {loss.item():.4f}  val {vloss:.4f}")

    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pv = model(Xva).cpu().numpy()
    yv = Yva.cpu().numpy()

    # baselines + metrics
    mean_pred = np.repeat(T[tr].mean(0)[None], len(va), 0)
    mb, _, _ = metrics(mean_pred, yv, tnames, steer_idx)
    mm, turn_acc, n_turn = metrics(pv, yv, tnames, steer_idx)
    print("\n── held-out performance (vs predict-the-mean baseline) ──")
    for ax in tnames:
        print(f"  {ax:14s}: R²={mm[ax][0]:+.3f} corr={mm[ax][1]:+.3f}   "
              f"(baseline R²={mb[ax][0]:+.3f})")
    print(f"  steer-direction accuracy on real turns (|{tnames[steer_idx]}|>0.3, n={n_turn}): {turn_acc*100:.0f}%")

    out = args.out or os.path.join(os.path.dirname(args.data), "bc_model.pt")
    torch.save({"state_dict": best[1], "mu": mu, "sd": sd, "target": args.target,
                "target_names": tnames, "steer_idx": steer_idx, "history": N,
                "feature_names": list(d["feature_names"]), "hidden": args.hidden}, out)
    print(f"\nsaved model → {out}")

    # plot: loss curve + predicted-vs-actual
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        h = np.array(hist)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(h[:, 0], label="train"); ax[0].plot(h[:, 1], label="val")
        ax[0].set_title("loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
        for k, name in enumerate(tnames):
            ax[k+1].scatter(yv[:, k], pv[:, k], s=6, alpha=.3)
            ax[k+1].plot([-1, 1], [-1, 1], "k--", lw=.8)
            ax[k+1].set_title(f"{name}  corr={mm[name][1]:.2f}")
            ax[k+1].set_xlabel("human"); ax[k+1].set_ylabel("model"); ax[k+1].set_xlim(-1,1); ax[k+1].set_ylim(-1,1)
        plt.tight_layout(); p = out.replace(".pt", "_eval.png"); plt.savefig(p, dpi=110)
        print(f"saved plot  → {p}")
    except Exception as e:
        print("(plot skipped:", e, ")")


if __name__ == "__main__":
    main()
