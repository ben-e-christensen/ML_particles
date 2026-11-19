"""
bead_count_check.py - label-free check of the CNN on EVERY frame: the drum always holds the same
number of beads, so any frame where the count is off has a detection mistake.

    python bead_count_check.py bead_cnn_30k.pt RUN_DIR [RUN_DIR ...] [--expect 473] [--stride 1]

    e.g. python bead_count_check.py bead_cnn_30k.pt "C:/Users/ben/Desktop/burst_test_2026-11-17_17_20/rpm01" \
             "C:/Users/ben/Desktop/burst_test_2026-11-17_17_20/rpm26"

Writes into count_check_<model>/ :
    counts.csv               every frame: run, frame, count, off-by, weakest detection score
    counts.svg / counts.png  bead count vs frame, one panel per run (red dots = frames that are off)
    off_frames.csv           only the frames that are off - the ones worth labelling next
    off/<run>__<frame>.png   up to --max-images per run: detections drawn on the frame
                             (green = detection, magenta = weak detection, score < --weak)

Limits: a frame with one miss AND one false hit still counts 473 - so a perfect count means
"probably right", not "proven right". The weakest-score column helps catch those: a frame whose
weakest detection is far below the rest deserves a look even if the count is fine.
Speed: ~0.15 s/frame on the RTX 3060 (1000 frames ~ 2.5 min); several times slower on CPU.
"""
import argparse, os, re, sys, time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_detector as bd
from bead_cnn import load_cnn, detect_cnn, list_frames, drum_crop

# chart colours (light surface)
SURFACE, INK, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SERIES, CRITICAL = "#2a78d6", "#d03b3b"


def run_name(d):
    d = os.path.abspath(d).rstrip("\\/")
    return f"{os.path.basename(os.path.dirname(d))}_{os.path.basename(d)}"


def draw(gray, df, weak, crop, path, title):
    x0, y0, x1, y1 = crop
    k = 2
    o = cv2.resize(cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), None, fx=k, fy=k,
                   interpolation=cv2.INTER_CUBIC)
    r = int(bd.BEAD_PX / 2 * k)
    for x, y, s in df[["x", "y", "score"]].values:
        p = (int((x - x0) * k), int((y - y0) * k))
        if s < weak:
            cv2.circle(o, p, r + 3, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(o, p, r, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(o, (0, 0), (o.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(o, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, o)


def plot(df, expect, out_base):
    runs = list(dict.fromkeys(df["run"]))
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 1.9 * len(runs) + 0.6), sharex=False,
                             squeeze=False, facecolor=SURFACE)
    for ax, run in zip(axes[:, 0], runs):
        d = df[df.run == run]
        off = d[d["count"] != expect]
        ax.set_facecolor(SURFACE)
        ax.axhline(expect, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(d["idx"], d["count"], color=SERIES, lw=1.5, zorder=2)
        if len(off):
            ax.scatter(off["idx"], off["count"], s=28, color=CRITICAL, edgecolor=SURFACE, lw=1.5, zorder=3)
        lo, hi = min(d["count"].min(), expect - 2), max(d["count"].max(), expect + 2)
        ax.set_ylim(lo - 1, hi + 1)
        ax.set_title(f"{run}   -   {len(off)} of {len(d)} frames off "
                     f"(expected {expect})", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylabel("beads", color=MUTED, fontsize=9)
    axes[-1, 0].set_xlabel("frame number", color=MUTED, fontsize=9)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("runs", nargs="+", help="frame folders (e.g. the rpm folders)")
    ap.add_argument("--expect", type=int, default=473, help="beads that are always in the drum")
    ap.add_argument("--stride", type=int, default=1, help="check every Nth frame (1 = all)")
    ap.add_argument("--weak", type=float, default=0.5, help="detections below this score are drawn magenta")
    ap.add_argument("--max-images", type=int, default=15, help="annotated off-frames saved per run")
    ap.add_argument("--out", default=None, help="output folder (default count_check_<model name>)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    out = a.out or f"count_check_{os.path.splitext(os.path.basename(a.model))[0]}"
    os.makedirs(os.path.join(out, "off"), exist_ok=True)
    m = load_cnn(a.model, a.device)
    print(f"model {a.model} on {m['device']}, expecting {a.expect} beads per frame")
    rows = []
    for rd in a.runs:
        run = run_name(rd)
        files = list_frames(rd)[::a.stride]
        if not files:
            print(f"  ! no frames in {rd}"); continue
        t0, n_img = time.time(), 0
        for i, f in enumerate(files):
            g = cv2.imread(os.path.join(rd, f), cv2.IMREAD_GRAYSCALE)
            df = detect_cnn(m, g)
            n = len(df)
            weakest = float(df["score"].min()) if n else np.nan
            idx = int(re.search(r"frame_(\d+)", f).group(1))
            rows.append(dict(run=run, frame=f, idx=idx, count=n, off_by=n - a.expect,
                             weakest_score=round(weakest, 3), n_weak=int((df["score"] < a.weak).sum()) if n else 0))
            if n != a.expect and n_img < a.max_images:
                draw(g, df, a.weak, drum_crop({"rim": m["rim"]}, g.shape),
                     os.path.join(out, "off", f"{run}__{os.path.splitext(f)[0]}.png"),
                     f"{run} {f}: {n} beads ({n - a.expect:+d})   magenta = weak (score < {a.weak})")
                n_img += 1
            if (i + 1) % 100 == 0 or i + 1 == len(files):
                el = time.time() - t0
                print(f"  {run}: {i+1}/{len(files)} frames ({el:.0f} s, ~{el/(i+1)*(len(files)-i-1):.0f} s left)",
                      flush=True)
    if not rows:
        raise SystemExit("nothing checked")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "counts.csv"), index=False)
    off = df[df["count"] != a.expect]
    off.to_csv(os.path.join(out, "off_frames.csv"), index=False)
    plot(df, a.expect, os.path.join(out, "counts"))

    print("\n=== per run ===")
    s = df.groupby("run", sort=False).agg(frames=("count", "size"),
                                          off=("off_by", lambda v: int((v != 0).sum())),
                                          min_count=("count", "min"), max_count=("count", "max"),
                                          weakest=("weakest_score", "min"))
    s["off_%"] = (100 * s["off"] / s["frames"]).round(1)
    print(s.to_string())
    print(f"\n{len(off)} of {len(df)} frames off. Wrote {out}/counts.svg (+png), counts.csv, off_frames.csv"
          + (f", and annotated off-frames in {out}/off/" if len(off) else ""))


if __name__ == "__main__":
    main()