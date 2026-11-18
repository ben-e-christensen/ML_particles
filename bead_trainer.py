r"""
bead_trainer.py - "it guesses, you fix, it learns" loop for the bead detector.

    python bead_trainer.py FRAME_DIR [--model bead_rf.joblib] [--labels bead_labels] [--step N]

For each frame it shows where the current model thinks every bead is. You fix the mistakes
and approve. Every approved frame becomes training data and the model is retrained before
the next frame is shown, so each frame should need fewer fixes than the last. The number of
fixes you had to make per frame is logged - that IS the accuracy measurement.

Starting a fresh model (new camera position / new lighting):
    1. python bead_setup.py FRAME_DIR      -> bead_setup.json (drum circle + bead size)
    2. python bead_trainer.py FRAME_DIR --model NEW.joblib --labels NEW_labels
With no model file yet, the first frame is pre-filled by a plain blob finder that only looks
inside the drum circle (no learning - just "bead-sized bright spots", then the same template
centring + no-overlap rule the detector uses). Fix it like any other frame and approve.
Or --no-seed to start that first frame blank and click every bead yourself.
If the seed is way off:
    --seed-thresh 0.5   fewer, surer seeds (raise if it marks junk; lower, e.g. 0.2, if it misses lots)
    --seed-dark          if beads look DARKER than the gaps between them
In corrections_log.csv the seeded frame is the row with frames_trained_on = 0.
Every size setting is scaled to bead_px from the setup file, and the model remembers it.

Controls (click in the image, with the toolbar zoom/pan tool switched OFF):
    left click        add a bead where you click (one it missed) - anywhere on the bead is fine,
                      the exact centre is found automatically when you approve
    right click       remove the nearest marked bead (a wrong one)
    u                 undo last add/remove
    a  or  enter      APPROVE this frame -> save, retrain, show next frame
    s                 skip this frame (don't save it)
    r                 reset this frame to the model's original guess
    q                 quit (everything approved so far is already saved)
    toolbar magnifier zoom in; press it again to get back to clicking

Marker colours:  green = model's guess   MAGENTA = model's best extra guesses to reach --target
                 (500 by default; these are the least sure, check them first)
                 cyan = you added   red x = you removed

Frames visited are spread through the run (--step, default = about 12 frames across the
folder) so the model sees different bed positions / lighting, not 12 near-identical frames.

Double-checking everything you've approved (one frame after another, your saved beads shown):
    python bead_trainer.py FRAME_DIR --review
    -> a or s with no edits = move on, nothing rewritten; edit then a = overwrite + retrain

Fixing a frame you already approved:
    python bead_trainer.py FRAME_DIR --redo 1776104886053_frame_000005.jpg
    -> opens it with YOUR saved beads (not a new guess); fix, press a, it overwrites and retrains.
    Several at once: --redo name1.jpg name2.jpg

Several frame folders (e.g. one per rpm): point every run at the SAME --model and --labels.
Each folder gets its own labels subfolder, named <parent folder>_<folder> (or --run NAME), so
identically named frames in different folders never collide, and the model trains on the approved
frames of ALL runs (each run remembers its frame folder in run_info.json).
Labels from before subfolders existed (sitting directly in the labels folder): move them once with
    python bead_trainer.py --adopt-flat bead_labels_uniform "C:\path\to\the\rpm01\they\came\from"

Files:
    bead_rf.joblib                       the model (overwritten after every approval)
    bead_labels/<run>/<frame>.csv        approved bead centres (kind=bead) and rejections (kind=not_bead)
    bead_labels/<run>/run_info.json      which frame folder the run's labels belong to
    bead_labels/<run>/corrections_log.csv   per frame: guesses, added, removed, error rate
"""
import argparse, glob, json, os, re, time
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_detector as bd      # bd.BEAD_PX etc. are read live: bead_setup.json rescales them
from bead_detector import (detect, load_model, sample_frame, fit_samples, build_template,
                           refine_centres, refine_constrained, fit_wall, MIN_NEIGHBOURS, _lcn)

# frame files: "frame_000005.png" or "1776104886053_frame_000005.jpg" (timestamp prefix optional)
FNAME_RE = re.compile(r"^(?:(\d{13})_)?frame_(\d+)\.(jpg|jpeg|png|tif+|bmp)$", re.I)
VIEW_PAD = 80
SEED_SIGMA_FRAC = 0.35   # LoG scale that best matches a disc of diameter BEAD_PX (radius / sqrt 2)


def seed_detect(gray, wall, thresh=0.35, dark=False):
    """No-model first guess for a fresh setup. Bead-sized blob finding (scale-normalised Laplacian
    of Gaussian on the contrast-normalised image) INSIDE the drum circle only, drop isolated hits
    (same neighbour rule as detect), then centre on an average-bead template built from these hits
    with the no-overlap rule. wall = (cx, cy, R) limit for bead centres from bead_setup.json.
    thresh is relative to a typical strong peak (90th percentile of peaks inside the drum), so a
    few very bright screw heads don't push every bead below the cut. Returns (n,2) xy."""
    H, W = gray.shape
    cx, cy, R = wall
    pad = int(2 * bd.BEAD_PX)
    x0, y0 = max(int(cx - R) - pad, 0), max(int(cy - R) - pad, 0)
    x1, y1 = min(int(cx + R) + pad, W), min(int(cy + R) + pad, H)
    n = _lcn(gray[y0:y1, x0:x1])
    if dark:
        n = -n
    s = SEED_SIGMA_FRAC * bd.BEAD_PX
    resp = -(s ** 2) * ndi.gaussian_laplace(n, s)          # bright bead-sized blob -> high
    pk = peak_local_max(resp, min_distance=max(2, int(0.6 * bd.BEAD_PX)), threshold_abs=1e-6,
                        exclude_border=False)
    vals = resp[pk[:, 0], pk[:, 1]]
    xy = pk[:, ::-1].astype(float) + [x0, y0]
    inside = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= R + bd.WALL_TOL
    xy, vals = xy[inside], vals[inside]
    if len(vals):
        keep = vals >= thresh * np.percentile(vals, 90)
        xy = xy[keep]
    print(f"  seed: {len(xy)} blob peaks inside the drum", end="")
    if len(xy) >= 20 and MIN_NEIGHBOURS:
        nn = cKDTree(xy).query_ball_point(xy, 1.5 * bd.BEAD_PX, return_length=True) - 1
        xy = xy[nn >= MIN_NEIGHBOURS]
        print(f", {len(xy)} after dropping isolated ones", end="")
    if len(xy) >= 20:
        T = build_template([(gray, xy)])
        xy, _ = refine_constrained(gray, xy, [T])
        xy = xy[np.isfinite(xy[:, 0])]
        print(f", {len(xy)} after centring / no-overlap", end="")
    print()
    return xy


def list_frames(frame_dir):
    out = []
    for f in os.listdir(frame_dir):
        m = FNAME_RE.match(f)
        if m:
            out.append((int(m.group(2)), f))
    return [f for _, f in sorted(out)]


def label_path(labels_dir, fname):
    return os.path.join(labels_dir, os.path.splitext(fname)[0] + ".csv")


def other_runs(labels_root, this_run):
    """[(run, frame_dir, [approved frame names])] for every OTHER run subfolder with run_info.json."""
    out = []
    for r in sorted(os.listdir(labels_root)):
        info = os.path.join(labels_root, r, "run_info.json")
        if r == this_run or not os.path.exists(info):
            continue
        with open(info) as fh:
            odir = json.load(fh)["frame_dir"]
        if not os.path.isdir(odir):
            print(f"  ! run '{r}': frame folder {odir} not found - its labels are skipped")
            continue
        fr = [f for f in list_frames(odir) if os.path.exists(label_path(os.path.join(labels_root, r), f))]
        if fr:
            out.append((r, odir, fr))
    return out


def adopt_flat(labels_root, frame_dir):
    """One-time move of old-layout label files (directly in labels_root) into a run subfolder."""
    fd = os.path.abspath(frame_dir).rstrip("\\/")
    run = f"{os.path.basename(os.path.dirname(fd))}_{os.path.basename(fd)}"
    dst = os.path.join(labels_root, run)
    os.makedirs(dst, exist_ok=True)
    moved = 0
    for c in os.listdir(labels_root):
        src = os.path.join(labels_root, c)
        if os.path.isfile(src) and c.lower().endswith(".csv"):
            if os.path.exists(os.path.join(dst, c)):
                raise SystemExit(f"{dst}/{c} already exists - not overwriting")
            os.replace(src, os.path.join(dst, c)); moved += 1
    with open(os.path.join(dst, "run_info.json"), "w") as fh:
        json.dump({"frame_dir": fd}, fh, indent=2)
    print(f"moved {moved} files into {dst}/ (run '{run}', frames in {fd})")


def load_labels(path):
    df = pd.read_csv(path)
    if "kind" not in df:
        df["kind"] = "bead"
    return df[df.kind == "bead"][["x", "y"]].values, df[df.kind == "not_bead"][["x", "y"]].values


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--adopt-flat":
        adopt_flat(sys.argv[2], sys.argv[3]); return
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("--model", default="bead_rf.joblib")
    ap.add_argument("--labels", default="bead_labels")
    ap.add_argument("--step", type=int, default=None, help="frames to jump between reviews")
    ap.add_argument("--start", type=int, default=None, help="index of first frame to review")
    ap.add_argument("--target", type=int, default=500,
                    help="beads per frame; if the model finds fewer, its best extra candidates are "
                         "shown in MAGENTA for you to keep or right-click away (0 = off)")
    ap.add_argument("--redo", nargs="+", default=None,
                    help="re-open already-approved frame(s) by file name to fix them")
    ap.add_argument("--review", action="store_true",
                    help="step through EVERY approved frame in the labels folder to double-check/edit")
    ap.add_argument("--run", default=None,
                    help="name for this frame folder's labels subfolder "
                         "(default: <parent folder>_<folder>, e.g. burst_test_2026-11-17_17_20_rpm06)")
    ap.add_argument("--setup", default="bead_setup.json",
                    help="from bead_setup.py: bead size + drum circle (required for a new model)")
    ap.add_argument("--seed-thresh", type=float, default=0.35,
                    help="no-model blob finder: min peak strength vs a typical strong peak "
                         "(raise = fewer false seeds, lower = fewer missed)")
    ap.add_argument("--no-seed", action="store_true",
                    help="no model yet: start the first frame BLANK and click every bead yourself")
    ap.add_argument("--seed-dark", action="store_true",
                    help="no-model blob finder: beads are darker than the gaps between them")
    a = ap.parse_args()
    setup = None
    if os.path.exists(a.setup):
        with open(a.setup) as fh:
            setup = json.load(fh)
        bd.set_bead_px(setup["bead_px"])
        print(f"setup {a.setup}: bead pitch {setup['bead_px']} px, drum limit for bead centres "
              f"centre ({setup['wall'][0]}, {setup['wall'][1]}) R {setup['wall'][2]} px")
    elif not os.path.exists(a.model):
        raise SystemExit(f"no model and no {a.setup} - run bead_setup.py FRAME_DIR first "
                         "(sets the drum region and bead size)")
    if setup and os.path.exists(a.model):
        mpx = joblib.load(a.model).get("bead_px", 19.0)
        if abs(mpx - setup["bead_px"]) / setup["bead_px"] > 0.05:
            raise SystemExit(f"{a.model} was trained at {mpx:.1f} px beads but {a.setup} says "
                             f"{setup['bead_px']} px - use a new --model (and --labels) for this setup")
    if setup is None:   # existing model, no setup file: use the bead size the model was trained at
        bd.set_bead_px(joblib.load(a.model).get("bead_px", 19.0))
    setup_wall = tuple(setup["wall"]) if setup else None
    # ---- labels live in one subfolder per run (frame folder), so rpm01/frame_000020 and
    #      rpm06/frame_000020 never collide; the model trains on every run's approved frames ----
    fd = os.path.abspath(a.frame_dir).rstrip("\\/")
    run = a.run or f"{os.path.basename(os.path.dirname(fd))}_{os.path.basename(fd)}"
    os.makedirs(a.labels, exist_ok=True)
    flat = [c for c in os.listdir(a.labels) if FNAME_RE.match(os.path.splitext(c)[0] + ".png")
            and c.lower().endswith(".csv")]
    if flat:
        raise SystemExit(
            f"{a.labels}/ has {len(flat)} label files directly in it (old layout, before runs had their own "
            f"subfolders). Move them into a subfolder named after the run they came from, e.g. for rpm01 of "
            f"burst_test_2026-11-17_17_20:\n"
            f"    python bead_trainer.py --adopt-flat {a.labels} \"C:\\path\\to\\that\\rpm01\"\n"
            f"then rerun this command.")
    run_dir = os.path.join(a.labels, run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_info.json"), "w") as fh:
        json.dump({"frame_dir": fd}, fh, indent=2)
    log_path = os.path.join(run_dir, "corrections_log.csv")
    print(f"run '{run}': labels in {run_dir}")

    files = list_frames(a.frame_dir)
    if not files:
        raise SystemExit("no frames found")
    step = a.step or max(1, len(files) // 12)
    read = lambda f: cv2.imread(os.path.join(a.frame_dir, f), cv2.IMREAD_GRAYSCALE)

    # ---- training rows from frames already approved (so retraining is fast) ----
    samples = {}            # key: frame name for this run, "otherrun/frame" for other runs
    first_pair = None       # (gray, beads) of some approved frame, for a bead template
    for f in files:
        p = label_path(run_dir, f)
        if os.path.exists(p):
            beads, notb = load_labels(p)
            if len(beads) >= 10:
                g = read(f)
                samples[f] = sample_frame(g, beads, notb)
                first_pair = first_pair or (g, beads)
    n_here = len(samples)
    for other, odir, ofiles in other_runs(a.labels, run):
        n0 = len(samples)
        for f in ofiles:
            beads, notb = load_labels(label_path(os.path.join(a.labels, other), f))
            g = cv2.imread(os.path.join(odir, f), cv2.IMREAD_GRAYSCALE)
            if g is None or len(beads) < 10:
                continue
            samples[f"{other}/{f}"] = sample_frame(g, beads, notb)
            first_pair = first_pair or (g, beads)
        print(f"  + {len(samples) - n0} approved frames from run '{other}'")
    print(f"{len(files)} frames in folder, {n_here} already approved here "
          f"({len(samples)} across all runs), reviewing every {step}th")

    def all_walls():
        if setup_wall is not None:                 # the circle you clicked beats one fitted to beads
            return setup_wall
        sets = []
        for f in files:
            p = label_path(run_dir, f)
            if os.path.exists(p):
                sets.append(load_labels(p)[0])
        return fit_wall(sets) if sets else None

    def apply_setup(m):
        """drum circle + search box from bead_setup.json onto a model (no-op without a setup)"""
        if setup_wall is not None and m is not None:
            cx, cy, R = setup_wall
            pad = 2 * bd.BEAD_PX
            m["wall"] = setup_wall
            m["wall_from_setup"] = True
            m["roi_exact"] = (int(cx - R - pad), int(cy - R - pad), int(cx + R + pad), int(cy + R + pad))
        return m

    model = apply_setup(load_model(a.model)) if os.path.exists(a.model) else None
    if model is not None and model.get("wall") is None:
        model["wall"] = all_walls()
        if model["wall"]:
            print("drum wall fitted from your labels: centre (%.0f, %.0f), radius %.0f px" % model["wall"])
    if model is None and samples:
        model = apply_setup(fit_samples(*zip(*samples.values())))
    if model is not None and model.get("template") is None and first_pair is not None:
        model["template"] = build_template([first_pair])
    if model is None:
        print("No model yet: the first frame is pre-filled by the plain blob finder - fix it and "
              "approve. After that the model guesses.")

    if a.review:
        a.redo = [f for f in files if os.path.exists(label_path(run_dir, f))]
        if not a.redo:
            raise SystemExit(f"no approved frames in {run_dir}/")
        print(f"REVIEW: {len(a.redo)} approved frames. a = keep/save (retrains only if you changed it), "
              "s = next without saving")
    if a.redo:
        names = [os.path.basename(r) for r in a.redo]
        missing = [n for n in names if n not in files]
        if missing:
            raise SystemExit(f"not in folder: {missing}")
        order = [files.index(n) for n in names]
    else:
        order = [i for i in range(a.start or 0, len(files), step)]
        order = [i for i in order if files[i] not in samples]
    if not order:
        raise SystemExit("every frame at this --step is already approved; use a different --step/--start")

    # stop matplotlib's default 's' (save) / 'q' / 'a' / 'r' shortcuts fighting ours
    for k in ("keymap.save", "keymap.quit", "keymap.home", "keymap.all_axes", "keymap.fullscreen"):
        if k in plt.rcParams:
            plt.rcParams[k] = []
    st = dict(k=0, gray=None, guess=None, pts=None, kind=None, removed=None, hist=None, quit=False)
    fig, ax = plt.subplots(figsize=(14, 9))

    def load(k):
        f = files[order[k]]
        g = read(f)
        saved = label_path(run_dir, f)
        fk = None
        if a.redo and os.path.exists(saved):
            d, _ = load_labels(saved)
            print(f"[{k+1}/{len(order)}] {f}: REDO - showing your saved {len(d)} beads")
        elif model is not None:
            t0 = time.time()
            dd = detect(model, g, target=a.target or None)
            d = dd[["x", "y"]].values
            fk = dd["forced"].values if "forced" in dd else np.zeros(len(d), bool)
            nf = int(fk.sum())
            print(f"[{k+1}/{len(order)}] {f}: model found {len(d) - nf}"
                  + (f", +{nf} extra candidates (magenta) to reach {a.target}" if nf else "")
                  + f" ({time.time()-t0:.0f} s)")
        elif a.no_seed:
            d = np.zeros((0, 2))
            print(f"[{k+1}/{len(order)}] {f}: no model yet - blank, click every bead once")
        else:
            t0 = time.time()
            print(f"[{k+1}/{len(order)}] {f}: no model yet - blob finder")
            d = seed_detect(g, setup_wall, a.seed_thresh, a.seed_dark)
            print(f"  seed found {len(d)} beads ({time.time()-t0:.0f} s) - all shown green, fix them")
        st.update(k=k, gray=g, guess=d.copy(), pts=[tuple(p) for p in d],
                  kind=list(np.where(fk, "forced", "guess")) if fk is not None else ["guess"] * len(d),
                  removed=[], hist=[])
        ax.clear()
        redraw(reset_view=True)

    def redraw(reset_view=False):
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        ax.imshow(st["gray"], cmap="gray", vmin=0, vmax=255)
        P = np.array(st["pts"]) if st["pts"] else np.zeros((0, 2))
        K = np.array(st["kind"])
        if setup_wall is not None:                 # drum limit from bead_setup.json
            ax.add_patch(plt.Circle(setup_wall[:2], setup_wall[2], fill=False, ec="yellow",
                                    lw=0.6, ls="--", alpha=0.6))
        if len(P):
            gs = P[K == "guess"]; ad = P[K == "added"]; fo = P[K == "forced"]
            ax.scatter(fo[:, 0], fo[:, 1], s=60, facecolors="none", edgecolors="magenta", lw=1.6)
            ax.scatter(gs[:, 0], gs[:, 1], s=30, facecolors="none", edgecolors="lime", lw=0.9)
            ax.scatter(ad[:, 0], ad[:, 1], s=30, facecolors="none", edgecolors="cyan", lw=1.4)
        if st["removed"]:
            R = np.array(st["removed"])
            ax.scatter(R[:, 0], R[:, 1], s=40, marker="x", c="red", lw=1.2)
        n_add = int((K == "added").sum()); n_rem = len(st["removed"])
        n_forced = int((K == "forced").sum()) if len(K) else 0
        ax.set_title(f"{files[order[st['k']]]}   ({st['k']+1}/{len(order)})   beads: {len(P)}"
                     + (f" ({n_forced} magenta = unsure extras)" if n_forced else "") + "   "
                     f"you added {n_add}, removed {n_rem}\n"
                     "L-click add | R-click remove | u undo | a approve+retrain | s skip | r reset | q quit",
                     fontsize=10)
        ax.axis("off")
        if reset_view:
            ref = P if len(P) else None
            if ref is None and samples:
                ref = None
            if ref is not None:
                ax.set_xlim(ref[:, 0].min() - VIEW_PAD, ref[:, 0].max() + VIEW_PAD)
                ax.set_ylim(ref[:, 1].max() + VIEW_PAD, ref[:, 1].min() - VIEW_PAD)
        else:
            ax.set_xlim(xl); ax.set_ylim(yl)
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or e.xdata is None or getattr(fig.canvas.toolbar, "mode", ""):
            return
        if e.button == 1:
            st["pts"].append((e.xdata, e.ydata)); st["kind"].append("added")
            st["hist"].append(("add",))
        elif e.button == 3 and st["pts"]:
            P = np.array(st["pts"])
            d = np.hypot(P[:, 0] - e.xdata, P[:, 1] - e.ydata)
            j = int(d.argmin())
            if d[j] > bd.BEAD_PX:
                return
            p, kd = st["pts"].pop(j), st["kind"].pop(j)
            if kd in ("guess", "forced"):
                st["removed"].append(p)
            st["hist"].append(("rem", j, p, kd))
        redraw()

    def undo():
        if not st["hist"]:
            return
        h = st["hist"].pop()
        if h[0] == "add":
            st["pts"].pop(); st["kind"].pop()
        else:
            _, j, p, kd = h
            st["pts"].insert(j, p); st["kind"].insert(j, kd)
            if kd in ("guess", "forced"):
                st["removed"].remove(p)

    def approve():
        nonlocal model
        f = files[order[st["k"]]]
        P = np.array(st["pts"]); K = np.array(st["kind"])
        if len(P) < 10:
            print("  fewer than 10 beads marked - not saving"); return False
        # your clicks only say WHICH blobs are beads; the exact centre is set by template matching
        T = model.get("template") if model is not None else None
        if T is None:
            T = build_template([(st["gray"], P)])
        Ts = (model.get("templates") if model is not None else None) or [T]
        Pr, _ = refine_constrained(st["gray"], P, Ts)
        bad = ~np.isfinite(Pr[:, 0])            # no legal spot found -> keep where you put it
        Pr[bad] = P[bad]
        P = Pr
        R = np.array(st["removed"]) if st["removed"] else np.zeros((0, 2))
        if a.redo and not st["hist"]:
            print(f"  no changes to {f} - left as is")
            return True
        df = pd.concat([pd.DataFrame({"x": P[:, 0], "y": P[:, 1], "kind": "bead",
                                      "source": np.where(K == "added", "you", "model")}),
                        pd.DataFrame({"x": R[:, 0], "y": R[:, 1], "kind": "not_bead", "source": "you"})])
        df.to_csv(label_path(run_dir, f), index=False)
        n_add = int((K == "added").sum()); n_rem = len(R); n_guess = len(st["guess"])
        row = dict(frame=f, frames_trained_on=len(samples), model_guesses=n_guess, missed_added=n_add,
                   false_removed=n_rem, final_beads=len(P),
                   recall_pct=round(100 * (len(P) - n_add) / len(P), 1),
                   precision_pct=round(100 * (n_guess - n_rem) / n_guess, 1) if n_guess else np.nan)
        if a.redo:
            print(f"  re-approved {f}: {len(P)} beads (overwrote saved labels; accuracy log unchanged)")
        else:
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
        if not a.redo:
            print(f"  approved: {len(P)} beads. You added {n_add}, removed {n_rem} "
              f"-> model was at recall {row['recall_pct']}%, precision {row['precision_pct']}% on this frame")
        ax.set_title("retraining... (window will say Not Responding until done - check the terminal)")
        fig.canvas.draw(); fig.canvas.flush_events()
        print(f"  retraining on {len(samples) + (f not in samples)} frames...", flush=True)
        t0 = time.time()
        samples[f] = sample_frame(st["gray"], P, R)
        prev = model or {}
        model = fit_samples(*zip(*samples.values()))
        model["template"] = T
        model["wall"] = all_walls()                # setup circle, else refitted from every approved frame
        apply_setup(model)
        for k in ("templates", "dim_thresh"):      # keep the lit/shadowed templates across retrains
            if prev.get(k) is not None:
                model[k] = prev[k]
        joblib.dump(model, a.model, compress=3)
        print(f"  retrained on {len(samples)} frames in {time.time()-t0:.0f} s, saved {a.model}")
        return True

    def advance():
        if st["k"] + 1 >= len(order):
            print("no more frames at this step. Done.")
            plt.close(fig); return
        load(st["k"] + 1)

    def on_key(e):
        if e.key == "u":
            undo(); redraw()
        elif e.key in ("a", "enter"):
            if approve():
                advance()
        elif e.key == "s":
            advance()
        elif e.key == "r":
            load(st["k"])
        elif e.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    load(0)
    fig.tight_layout()
    plt.show()

    if os.path.exists(log_path):
        print("\n=== corrections log (accuracy on each frame BEFORE you fixed it) ===")
        print(pd.read_csv(log_path).to_string(index=False))


if __name__ == "__main__":
    main()