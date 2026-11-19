"""
bead_fix_frames.py - fix the frames bead_count_check.py flagged, with the NETWORK's guesses
pre-filled, and save them as training labels (marked "hard" so training can favour them).

    python bead_fix_frames.py count_check_bead_cnn_30k/off_frames.csv --model bead_cnn_30k.pt \
        --labels bead_labels_uniform [--expect 473] [--only rpm06]

Each frame opens showing:
    green            the network's detections
    MAGENTA ring     detections it was unsure of (score < 0.5) - often the false hit
    YELLOW dashed    spots it ALMOST called a bead (score 0.08-0.3) - often the missing one;
                     left-click on one to add it
    title            live count vs the expected count ("472 / 473 - need +1")

Controls (toolbar zoom/pan OFF when clicking):
    left click     add a bead (snaps onto a yellow candidate if you click near one)
    right click    remove the nearest bead
    u              undo
    a / enter      approve -> save, next frame       (only when the count matches, unless --allow-off)
    s              skip this frame (not saved)
    r              reset to the network's guess
    q              quit - everything approved so far is saved; rerun to continue where you left off

Saves into the run's own labels subfolder (the same place bead_trainer.py uses):
    <labels>/<run>/<frame>.csv   bead centres + removed spots (kind=not_bead) - normal label files
    <labels>/<run>/hard.txt      list of frames fixed here -> bead_cnn.py train --hard-boost uses it
Frames that already have a label file are skipped.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_detector as bd
from bead_cnn import load_cnn, detect_cnn, heatmap, THRESH

NEAR_LO = 0.08      # heatmap peaks between this and THRESH are shown as "almost a bead"
WEAK = 0.5          # detections below this score get a magenta ring
VIEW_PAD = 40


def near_misses(m, gray, det_xy):
    hm, x0, y0, s = heatmap(m, gray)
    pk = peak_local_max(hm, min_distance=max(2, int(0.4 * m["work_px"])), threshold_abs=NEAR_LO,
                        exclude_border=False)
    pk = pk[hm[pk[:, 0], pk[:, 1]] < THRESH]
    xy = pk[:, ::-1].astype(float) / s + [x0, y0]
    xy = xy[bd.inside_wall({"wall": m["wall"]}, xy)]
    if len(xy) and len(det_xy):
        xy = xy[cKDTree(det_xy).query(xy)[0] > 0.6 * m["bead_px"]]
    return xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_csv", help="off_frames.csv from bead_count_check.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--expect", type=int, default=473)
    ap.add_argument("--only", default=None, help="only runs whose name contains this (e.g. rpm06)")
    ap.add_argument("--allow-off", action="store_true",
                    help="allow approving a frame whose count isn't --expect (e.g. a bead really left view)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    todo = pd.read_csv(a.off_csv)
    if a.only:
        todo = todo[todo["run"].str.contains(a.only)]
    run_dirs = {}
    for run in todo["run"].unique():
        info = os.path.join(a.labels, run, "run_info.json")
        if not os.path.exists(info):
            print(f"  ! no {info} - run '{run}' skipped (label at least one frame of it with bead_trainer.py first)")
            continue
        with open(info) as fh:
            run_dirs[run] = json.load(fh)["frame_dir"]
    lab_path = lambda run, f: os.path.join(a.labels, run, os.path.splitext(f)[0] + ".csv")
    todo = todo[todo["run"].isin(run_dirs) & ~todo.apply(lambda r: os.path.exists(lab_path(r.run, r.frame)), axis=1)]
    todo = todo.reset_index(drop=True)
    if not len(todo):
        raise SystemExit("nothing to do - every listed frame already has labels")
    m = load_cnn(a.model, a.device)
    print(f"{len(todo)} frames to fix, model {a.model} on {m['device']}")

    for k in ("keymap.save", "keymap.quit", "keymap.home", "keymap.all_axes", "keymap.fullscreen"):
        if k in plt.rcParams:
            plt.rcParams[k] = []
    st = dict(i=0, done=0)
    fig, ax = plt.subplots(figsize=(14, 9))

    def load(i):
        r = todo.iloc[i]
        g = cv2.imread(os.path.join(run_dirs[r.run], r.frame), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"  ! can't read {r.frame} in {run_dirs[r.run]} - skipped"); return False
        df = detect_cnn(m, g)
        P = df[["x", "y"]].values
        st.update(i=i, gray=g, pts=[tuple(p) for p in P], kind=["guess"] * len(P),
                  score=list(df["score"].values), removed=[], hist=[], near=near_misses(m, g, P))
        print(f"[{i+1}/{len(todo)}] {r.run} {r.frame}: network found {len(P)} "
              f"({len(P) - a.expect:+d}), {int((df['score'] < WEAK).sum())} unsure, "
              f"{len(st['near'])} near-misses")
        redraw(reset=True)
        return True

    def redraw(reset=False):
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        ax.imshow(st["gray"], cmap="gray", vmin=0, vmax=255)
        P = np.array(st["pts"]) if st["pts"] else np.zeros((0, 2))
        K = np.array(st["kind"]); S = np.array(st["score"], float)
        if len(P):
            sure = (K == "guess") & (S >= WEAK); weak = (K == "guess") & (S < WEAK); add = K == "added"
            ax.scatter(*P[sure].T, s=30, facecolors="none", edgecolors="lime", lw=0.9)
            ax.scatter(*P[weak].T, s=90, facecolors="none", edgecolors="magenta", lw=1.8)
            ax.scatter(*P[add].T, s=30, facecolors="none", edgecolors="cyan", lw=1.4)
        if len(st["near"]):
            ax.scatter(*st["near"].T, s=90, facecolors="none", edgecolors="yellow", lw=1.2, linestyles="--")
        if st["removed"]:
            ax.scatter(*np.array(st["removed"]).T, s=40, marker="x", c="red", lw=1.2)
        n = len(P); need = a.expect - n
        r = todo.iloc[st["i"]]
        ax.set_title(f"{r.run}  {r.frame}   ({st['i']+1}/{len(todo)})     "
                     f"{n} / {a.expect}" + ("   OK" if need == 0 else f"   need {need:+d}") +
                     "\nmagenta = unsure | yellow dashed = almost a bead (click to add) | "
                     "L add | R remove | u undo | a approve | s skip | r reset | q quit", fontsize=10,
                     color="black" if need == 0 else "red")
        ax.axis("off")
        if reset:
            cx, cy, R = m["rim"]
            ax.set_xlim(cx - R - VIEW_PAD, cx + R + VIEW_PAD); ax.set_ylim(cy + R + VIEW_PAD, cy - R - VIEW_PAD)
        else:
            ax.set_xlim(xl); ax.set_ylim(yl)
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or e.xdata is None or getattr(fig.canvas.toolbar, "mode", ""):
            return
        p = np.array([e.xdata, e.ydata])
        if e.button == 1:
            if len(st["near"]):                         # snap onto a yellow candidate
                d = np.hypot(*(st["near"] - p).T); j = int(d.argmin())
                if d[j] < 0.6 * m["bead_px"]:
                    p = st["near"][j]
                    st["near"] = np.delete(st["near"], j, 0)
                    st["hist"].append(("add", tuple(p)))
                    st["pts"].append(tuple(p)); st["kind"].append("added"); st["score"].append(1.0)
                    redraw(); return
            st["pts"].append(tuple(p)); st["kind"].append("added"); st["score"].append(1.0)
            st["hist"].append(("add", None))
        elif e.button == 3 and st["pts"]:
            P = np.array(st["pts"]); d = np.hypot(*(P - p).T); j = int(d.argmin())
            if d[j] > m["bead_px"]:
                return
            q, kd, sc = st["pts"].pop(j), st["kind"].pop(j), st["score"].pop(j)
            if kd == "guess":
                st["removed"].append(q)
            st["hist"].append(("rem", j, q, kd, sc))
        redraw()

    def undo():
        if not st["hist"]:
            return
        h = st["hist"].pop()
        if h[0] == "add":
            st["pts"].pop(); st["kind"].pop(); st["score"].pop()
            if h[1] is not None:                        # it came from a yellow candidate: put it back
                st["near"] = np.vstack([st["near"], np.array(h[1])[None]]) if len(st["near"]) else np.array([h[1]])
        else:
            _, j, q, kd, sc = h
            st["pts"].insert(j, q); st["kind"].insert(j, kd); st["score"].insert(j, sc)
            if kd == "guess":
                st["removed"].remove(q)
        redraw()

    def approve():
        r = todo.iloc[st["i"]]
        P = np.array(st["pts"]); K = np.array(st["kind"])
        if len(P) != a.expect and not a.allow_off:
            print(f"  count is {len(P)}, not {a.expect} - fix it first (or rerun with --allow-off)")
            return False
        Pr, _ = bd.refine_constrained(st["gray"], P, [m["template"]])
        bad = ~np.isfinite(Pr[:, 0]); Pr[bad] = P[bad]
        R = np.array(st["removed"]) if st["removed"] else np.zeros((0, 2))
        df = pd.concat([pd.DataFrame({"x": Pr[:, 0], "y": Pr[:, 1], "kind": "bead",
                                      "source": np.where(K == "added", "you", "cnn")}),
                        pd.DataFrame({"x": R[:, 0], "y": R[:, 1], "kind": "not_bead", "source": "you"})])
        df.to_csv(lab_path(r.run, r.frame), index=False)
        with open(os.path.join(a.labels, r.run, "hard.txt"), "a") as fh:
            fh.write(r.frame + "\n")
        st["done"] += 1
        print(f"  saved: {len(P)} beads, you added {int((K == 'added').sum())}, removed {len(R)}")
        return True

    def advance():
        i = st["i"] + 1
        while i < len(todo):
            if load(i):
                return
            i += 1
        print(f"all done - {st['done']} frames saved"); plt.close(fig)

    def on_key(e):
        if e.key == "u":
            undo()
        elif e.key in ("a", "enter"):
            if approve():
                advance()
        elif e.key == "s":
            advance()
        elif e.key == "r":
            load(st["i"])
        elif e.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    i = 0
    while i < len(todo) and not load(i):
        i += 1
    fig.tight_layout()
    plt.show()
    print(f"\n{st['done']} frames saved as labels this session. Next:\n"
          f"  python bead_cnn.py train --labels {a.labels} --init {a.model} --hard-boost 4 --steps 3000 "
          f"--out bead_cnn_fixed.pt")


if __name__ == "__main__":
    main()