"""
bead_cnn.py - convolutional-network bead detector (small U-Net -> bead-centre heatmap).

Why: the random forest in bead_detector.py judges each pixel from a ~1-bead window, so anything
with bead-sized bright bumps (the aluminium disks) looks like beads to it. This network sees a
region many beads across for every output pixel, so it learns from your labels what a whole
disk, the bed edge and a packed bed look like - partly covered or not.

Uses the SAME training data as the forest: the approved frames in your labels folder
(bead centres + the spots you right-clicked away, which are weighted extra as "not a bead"),
and the drum circle + bead size from bead_setup.json.

The labels folder holds one subfolder per run (rpm folder), made by bead_trainer.py; training
uses every run in it, so FRAME_DIR is not needed (it is only for an old flat labels folder).

Fair comparison first (holds out frames neither model trains on; also trains a forest on the
same frames and scores both on the held-out ones):
    python bead_cnn.py train --labels bead_labels_uniform --holdout 2 --compare-forest
      -> prints recall / precision for both, writes compare_<frame>.png per held-out frame
         (left = forest, right = CNN; green = detection, red + = your label,
          blue ring = missed bead, yellow ring = false hit)

After fixing flagged frames with bead_fix_frames.py, top the model up (minutes, not an hour):
    python bead_cnn.py train --labels bead_labels_uniform --init bead_cnn_30k.pt --hard-boost 4 \
        --steps 3000 --out bead_cnn_fixed.pt

Real model on every approved frame:
    python bead_cnn.py train --labels bead_labels_uniform --out bead_cnn.pt

Run on a frame and look at it:
    python bead_cnn.py detect bead_cnn.pt FRAME.png [--labels LABELS.csv]   -> cnn_<frame>.png

GPU is used automatically when PyTorch can see one (NVIDIA/CUDA, or Apple MPS); otherwise CPU.
A model trained on one machine loads on any other (GPU or not). --device cpu forces CPU.

Install PyTorch once per machine, inside the venv. On Windows a plain "pip install torch" gives
a CPU-only build, so for an NVIDIA card get the exact command from https://pytorch.org/get-started/locally/
(Stable / Windows / Pip / Python / newest CUDA) - it is "python -m pip install torch --index-url
https://download.pytorch.org/whl/cuXXX". Check it worked:
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
No GPU: python -m pip install torch

Speed guide: RTX 3060 - training (4000 steps) a few minutes, detection < 1 s per frame.
CPU - expect 10-20x slower; use --steps 1500 for a quick look.
"""
import argparse, json, os, re, sys, time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import peak_local_max

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bead_detector as bd

FNAME_RE = re.compile(r"^(?:(\d{13})_)?frame_(\d+)\.(jpg|jpeg|png|tif+|bmp)$", re.I)
WORK_PX   = 16.0     # every frame is resampled so beads are this many px apart (camera-independent)
CROP      = 256      # training crop size in work px (= 16 beads across)
SIGMA     = 0.13     # heatmap blob size, as a fraction of a bead
REJECT_W  = 5.0      # extra loss weight around spots you right-clicked away
THRESH    = 0.3      # heatmap peak threshold at detect time


# ----------------------------------------------------------------------------- network
def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4 down-samplings (16x): each output pixel sees ~ 180 work px = ~11 beads across."""
    def __init__(self, ch=(24, 48, 96, 160, 256)):
        super().__init__()
        self.ch = ch
        self.enc = nn.ModuleList([block(1, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1))])
        self.head = nn.Conv2d(ch[0], 1, 1)
        nn.init.constant_(self.head.bias, -2.2)       # start predicting "mostly not a bead"

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for d in self.dec:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = d(torch.cat([x, s], 1))
        return self.head(x)


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- data
def list_frames(frame_dir):
    out = [(int(m.group(2)), f) for f in os.listdir(frame_dir) if (m := FNAME_RE.match(f))]
    return [f for _, f in sorted(out)]


def load_labelled(frame_dir, labels_dir):
    """[(name, gray, beads_xy, rejected_xy)] for every frame with an approved label file.
    labels_dir with run subfolders (each holding run_info.json, made by bead_trainer.py): every run
    is loaded and names are "run/frame". Old flat layout: needs frame_dir."""
    runs = [r for r in sorted(os.listdir(labels_dir))
            if os.path.exists(os.path.join(labels_dir, r, "run_info.json"))]
    if runs:
        out = []
        for r in runs:
            with open(os.path.join(labels_dir, r, "run_info.json")) as fh:
                fd = json.load(fh)["frame_dir"]
            if not os.path.isdir(fd):
                print(f"  ! run '{r}': frame folder {fd} not found - skipped"); continue
            got = load_labelled(fd, os.path.join(labels_dir, r))
            out += [(f"{r}/{f}", g, b, j) for f, g, b, j in got]
            print(f"  run '{r}': {len(got)} approved frames")
        return out
    if frame_dir is None:
        raise SystemExit(f"{labels_dir} has no run subfolders - give the frame folder too")
    out = []
    for f in list_frames(frame_dir):
        p = os.path.join(labels_dir, os.path.splitext(f)[0] + ".csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if "kind" not in d:
            d["kind"] = "bead"
        b = d[d.kind == "bead"][["x", "y"]].values.astype(float)
        r = d[d.kind == "not_bead"][["x", "y"]].values.astype(float)
        if len(b) >= 10:
            out.append((f, cv2.imread(os.path.join(frame_dir, f), cv2.IMREAD_GRAYSCALE), b, r))
    return out


def hard_names(labels_dir):
    """frames listed in <labels>/<run>/hard.txt (written by bead_fix_frames.py), as "run/frame"."""
    out = set()
    if not os.path.isdir(labels_dir):
        return out
    for r in os.listdir(labels_dir):
        p = os.path.join(labels_dir, r, "hard.txt")
        if os.path.exists(p):
            with open(p) as fh:
                out |= {f"{r}/{ln.strip()}" for ln in fh if ln.strip()}
    return out


def norm(x01):
    return (x01 - 0.4) / 0.25


def render_target(pts, n, sigma):
    """Gaussian heatmap, exactly 1 at the pixel nearest each centre."""
    t = np.zeros((n, n), np.float32)
    if not len(pts):
        return t
    r = int(np.ceil(3 * sigma))
    gy, gx = np.mgrid[-r:r + 1, -r:r + 1]
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if not (-r <= ix < n + r and -r <= iy < n + r):
            continue
        g = np.exp(-((gx + ix - x) ** 2 + (gy + iy - y) ** 2) / (2 * sigma ** 2))
        x0, y0 = ix - r, iy - r
        xs0, ys0 = max(x0, 0), max(y0, 0)
        xs1, ys1 = min(x0 + 2 * r + 1, n), min(y0 + 2 * r + 1, n)
        if xs1 <= xs0 or ys1 <= ys0:
            continue
        sub = g[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
        t[ys0:ys1, xs0:xs1] = np.maximum(t[ys0:ys1, xs0:xs1], sub)
        if 0 <= ix < n and 0 <= iy < n:
            t[iy, ix] = 1.0
    return t


class Sampler:
    """Random rotated / flipped / rescaled / re-lit crops from inside the drum."""
    def __init__(self, frames, bead_px, rim, rng, weights=None):
        self.frames, self.rng = frames, rng
        w = np.ones(len(frames)) if weights is None else np.asarray(weights, float)
        self.p = w / w.sum()                      # how often each frame is drawn
        self.scale = WORK_PX / bead_px
        self.cx, self.cy, self.R = rim
        self.sigma = SIGMA * WORK_PX

    def one(self):
        rng = self.rng
        f, g, b, rj = self.frames[rng.choice(len(self.frames), p=self.p)]
        rad = self.R * np.sqrt(rng.random()); ang = rng.random() * 2 * np.pi
        c = np.array([self.cx + rad * np.cos(ang), self.cy + rad * np.sin(ang)])
        th = rng.random() * 360
        s = self.scale * rng.uniform(0.9, 1.1)
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), th, s)
        M[:, 2] += CROP / 2 - c
        if rng.random() < 0.5:                                   # mirror
            M = np.array([[-1, 0, CROP], [0, 1, 0]], float) @ np.vstack([M, [0, 0, 1]])
        img = cv2.warpAffine(g, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32) / 255
        # loss only counts inside the drum (beads can't be outside; detection never looks there)
        drum = np.zeros(g.shape, np.uint8)
        cv2.circle(drum, (int(self.cx), int(self.cy)), int(self.R + 0.5 / self.scale * WORK_PX), 1, -1)
        valid = cv2.warpAffine(drum, M, (CROP, CROP), flags=cv2.INTER_NEAREST, borderValue=0).astype(np.float32)
        tp = lambda P: (np.c_[P, np.ones(len(P))] @ M.T) if len(P) else np.zeros((0, 2))
        tgt = render_target(tp(b), CROP, self.sigma)
        w = valid.copy()
        R = tp(rj)
        if len(R):
            yy, xx = np.mgrid[0:CROP, 0:CROP]
            for x, y in R:
                if -10 < x < CROP + 10 and -10 < y < CROP + 10:
                    w[np.hypot(xx - x, yy - y) < 0.4 * WORK_PX] *= REJECT_W
        # lighting / sensor jitter
        img = np.clip(img, 0, 1) ** rng.uniform(0.75, 1.35)
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
        img = img + rng.normal(0, rng.uniform(0, 0.03), img.shape).astype(np.float32)
        return norm(img), tgt, w

    def batch(self, n):
        x, t, w = zip(*[self.one() for _ in range(n)])
        return (torch.from_numpy(np.stack(x)[:, None]), torch.from_numpy(np.stack(t)[:, None]),
                torch.from_numpy(np.stack(w)[:, None]))


def focal_loss(logit, t, w):
    """CenterNet heatmap loss: positives = centre pixels; negatives down-weighted near centres."""
    p = torch.sigmoid(logit.float()).clamp(1e-4, 1 - 1e-4)
    pos = (t >= 0.999).float() * (w > 0).float()
    lp = -torch.log(p) * (1 - p) ** 2 * pos
    ln = -torch.log(1 - p) * p ** 2 * (1 - t) ** 4 * (1 - pos) * w
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1)


# ----------------------------------------------------------------------------- train / detect
def train(frames, setup, steps, batch, device, seed=0, log_every=250, every=0, callback=None,
          weights=None, init=None, lr=2e-3):
    """every / callback: call callback(step, net) every `every` steps (net in eval mode during it)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    S = Sampler(frames, setup["bead_px"], setup["rim"], rng, weights)
    if init:                                      # fine-tune: start from an existing model's weights
        m0 = torch.load(init, map_location=device, weights_only=True)
        net = UNet(tuple(m0["ch"])).to(device)
        net.load_state_dict(m0["state"])
    else:
        net = UNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    net.train()
    t0, run = time.time(), []
    for it in range(1, steps + 1):
        x, t, w = (v.to(device, non_blocking=True) for v in S.batch(batch))
        with torch.autocast(device.type, enabled=use_amp):
            out = net(x)
        loss = focal_loss(out, t, w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run.append(loss.item())
        if it % log_every == 0 or it == steps:
            el = time.time() - t0
            print(f"  step {it:5d}/{steps}  loss {np.mean(run):.3f}  ({el:.0f} s, ~{el / it * (steps - it):.0f} s left)",
                  flush=True)
            run = []
        if every and callback and it % every == 0 and it < steps:
            net.eval()
            callback(it, net)
            net.train()
    net.eval()
    return net


def make_model(net, setup, frames, T=None):
    bd.set_bead_px(setup["bead_px"])
    if T is None:
        T = bd.build_template([(g, b) for _, g, b, _ in frames[:8]])
    return dict(state=net.state_dict(), ch=list(net.ch), bead_px=float(setup["bead_px"]),
                rim=[float(v) for v in setup["rim"]], wall=[float(v) for v in setup["wall"]],
                template=torch.from_numpy(T), work_px=WORK_PX,
                trained_on=[f for f, *_ in frames])


def live_model(net, setup, T, device):
    """model dict usable by detect_cnn straight from a network that is still training"""
    return dict(net=net, device=device, template=T, bead_px=float(setup["bead_px"]),
                rim=[float(v) for v in setup["rim"]], wall=[float(v) for v in setup["wall"]],
                work_px=WORK_PX)


def score_frames(m, hold):
    """totals over the held-out frames: recall, precision, missed, false"""
    nl = nd = tp = 0
    for _, g, b, _ in hold:
        sc = bd.score(detect_cnn(m, g)[["x", "y"]].values, b)
        nl += sc["n_lab"]; nd += sc["n_det"]; tp += len(sc["tp_l"])
    return dict(recall=tp / max(nl, 1), precision=tp / max(nd, 1), missed=nl - tp, false=nd - tp)


def save_model(m, path):
    torch.save(m, path)


def load_cnn(path, device=None):
    device = pick_device(device)
    m = torch.load(path, map_location=device, weights_only=True)
    net = UNet(tuple(m["ch"])).to(device)
    net.load_state_dict(m["state"]); net.eval()
    m["net"], m["device"] = net, device
    m["template"] = m["template"].cpu().numpy()       # used by the CPU-side centring step
    bd.set_bead_px(m["bead_px"])
    return m


@torch.no_grad()
def heatmap(m, gray):
    """Bead-centre probability over the drum box, in work px. Returns (heat, x0, y0, scale)."""
    cx, cy, R = m["rim"]
    pad = 2 * m["bead_px"]
    H, W = gray.shape
    x0, y0 = max(int(cx - R - pad), 0), max(int(cy - R - pad), 0)
    x1, y1 = min(int(cx + R + pad), W), min(int(cy + R + pad), H)
    s = m["work_px"] / m["bead_px"]
    crop = cv2.resize(gray[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = crop.shape
    ph, pw = (-h) % 16, (-w) % 16
    x = norm(np.pad(crop.astype(np.float32) / 255, ((0, ph), (0, pw)), mode="reflect"))
    dev = m["device"]
    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
        out = m["net"](torch.from_numpy(x)[None, None].to(dev))
    return torch.sigmoid(out.float())[0, 0, :h, :w].cpu().numpy(), x0, y0, s


def detect_cnn(m, gray, thresh=THRESH, refine=True):
    """DataFrame x, y, score in raw-frame px. refine = the same template centring + no-overlap
    rule the forest uses (sub-pixel)."""
    hm, x0, y0, s = heatmap(m, gray)
    pk = peak_local_max(hm, min_distance=max(2, int(0.4 * m["work_px"])), threshold_abs=thresh,
                        exclude_border=False)
    xy, sc = [], []
    for i, j in pk:                                   # sub-pixel: weighted centroid of 3x3
        ys = slice(max(i - 1, 0), min(i + 2, hm.shape[0]))      # clipped to the map so peaks on
        xs = slice(max(j - 1, 0), min(j + 2, hm.shape[1]))      # the edge get a matching grid
        wv = hm[ys, xs]; gy, gx = np.mgrid[ys, xs]
        xy.append(((wv * gx).sum() / wv.sum(), (wv * gy).sum() / wv.sum())); sc.append(hm[i, j])
    xy = np.array(xy).reshape(-1, 2) / s + [x0, y0]
    sc = np.array(sc)
    keep = bd.inside_wall({"wall": m["wall"]}, xy)
    xy, sc = xy[keep], sc[keep]
    if refine and len(xy) >= 5:
        r, _ = bd.refine_constrained(gray, xy, [m["template"]])
        ok = np.isfinite(r[:, 0])
        xy, sc = r[ok], sc[ok]
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "score": sc})


def forest_baseline(train_frames, setup):
    """A forest trained the way bead_trainer.py does it, on the same frames, for a fair comparison."""
    bd.set_bead_px(setup["bead_px"])
    S = [bd.sample_frame(g, b, r) for _, g, b, r in train_frames]
    m = bd.fit_samples(*zip(*S))
    m["template"] = bd.build_template([(g, b) for _, g, b, _ in train_frames[:8]])
    cx, cy, R = setup["wall"]; pad = 2 * setup["bead_px"]
    m["wall"] = tuple(setup["wall"])
    m["roi_exact"] = (int(cx - R - pad), int(cy - R - pad), int(cx + R + pad), int(cy + R + pad))
    return m


def drum_crop(setup, shape):
    cx, cy, R = setup["rim"]; H, W = shape
    return (max(int(cx - R), 0), max(int(cy - R), 0), min(int(cx + R), W), min(int(cy + R), H))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("frame_dir", nargs="?", default=None,
                                               help="only needed for an old flat labels folder")
    t.add_argument("--labels", required=True); t.add_argument("--setup", default="bead_setup.json")
    t.add_argument("--out", default="bead_cnn.pt")
    t.add_argument("--steps", type=int, default=4000); t.add_argument("--batch", type=int, default=16)
    t.add_argument("--holdout", type=int, default=0,
                   help="hold this many approved frames out of training and score on them")
    t.add_argument("--compare-forest", action="store_true",
                   help="with --holdout: also train a forest on the same frames and score it too")
    t.add_argument("--device", default=None, help="cuda / cpu / mps (default: best available)")
    t.add_argument("--skip-train", action="store_true",
                   help="don't train: reuse the model in --out (same --holdout) and just run the comparison")
    t.add_argument("--eval-every", type=int, default=0,
                   help="with --holdout: save a checkpoint and score the held-out frames every N steps "
                        "(into <out>_report/), so one long run shows where it stops improving")
    t.add_argument("--hard-boost", type=float, default=1.0,
                   help="draw frames fixed with bead_fix_frames.py (hard.txt) this many times more often")
    t.add_argument("--init", default=None,
                   help="start from this model's weights instead of from scratch (fine-tune; use fewer steps)")
    t.add_argument("--lr", type=float, default=None,
                   help="peak learning rate (default 2e-3 from scratch, 5e-4 with --init)")
    t.add_argument("--crop", type=int, default=CROP, help="training crop size in work px (multiple of 16)")
    d = sub.add_parser("detect"); d.add_argument("model"); d.add_argument("frame")
    d.add_argument("--labels", default=None); d.add_argument("--thresh", type=float, default=THRESH)
    d.add_argument("--device", default=None)
    a = ap.parse_args()

    if a.cmd == "detect":
        m = load_cnn(a.model, a.device)
        g = cv2.imread(a.frame, cv2.IMREAD_GRAYSCALE)
        t0 = time.time(); det = detect_cnn(m, g, a.thresh)[["x", "y"]].values
        print(f"{len(det)} beads ({time.time() - t0:.2f} s on {m['device']})")
        lab = sc = None; title = f"CNN: {len(det)} beads"
        if a.labels:
            L = pd.read_csv(a.labels)
            lab = L[L.kind == "bead"][["x", "y"]].values if "kind" in L else L[["x", "y"]].values
            sc = bd.score(det, lab); bd._print_score("vs labels", sc)
            title += f"  recall {100*sc['recall']:.1f}%  precision {100*sc['precision']:.1f}%"
        out = f"cnn_{os.path.splitext(os.path.basename(a.frame))[0]}.png"
        bd.draw_check(g, lab, det, sc, drum_crop({"rim": m["rim"]}, g.shape), out, title)
        print(f"wrote {out}")
        return

    with open(a.setup) as fh:
        setup = json.load(fh)
    bd.set_bead_px(setup["bead_px"])
    frames = load_labelled(a.frame_dir, a.labels)
    if not frames:
        raise SystemExit(f"no approved frames in {a.labels}/")
    device = pick_device(a.device)
    globals()["CROP"] = a.crop
    print(f"{len(frames)} approved frames, {sum(len(b) for _, _, b, _ in frames)} beads, "
          f"{sum(len(r) for *_, r in frames)} rejections | device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    hold = []
    if a.holdout:
        if a.holdout >= len(frames):
            raise SystemExit("need more approved frames than --holdout")
        idx = np.linspace(0, len(frames) - 1, a.holdout + 2)[1:-1].round().astype(int)
        hold = [frames[i] for i in idx]
        frames = [f for i, f in enumerate(frames) if i not in set(idx)]
        print(f"held out (never trained on): {[f for f, *_ in hold]}")

    if a.skip_train:
        used = set(torch.load(a.out, map_location="cpu", weights_only=True)["trained_on"])
        leaked = [f for f, *_ in hold if f in used]
        if leaked:
            raise SystemExit(f"{a.out} was trained on {leaked} - can't score on them; retrain with --holdout")
        print(f"reusing {a.out} (trained on {len(used)} frames, none of the held-out ones)")
    stem = os.path.splitext(a.out)[0]
    rdir = stem + "_report"               # every output of this run goes here -> runs never overwrite
    if hold:
        os.makedirs(rdir, exist_ok=True)
    prog_path = os.path.join(rdir, "progress.csv")
    if not a.skip_train:
        if a.eval_every and not hold:
            raise SystemExit("--eval-every needs --holdout (it scores the held-out frames)")
        print(f"training CNN on {len(frames)} frames, {a.steps} steps x {a.batch} crops...")
        T = bd.build_template([(g, b) for _, g, b, _ in frames[:8]])
        if os.path.exists(prog_path):
            os.remove(prog_path)

        def checkpoint(it, net):
            ck = os.path.join(rdir, f"{os.path.basename(stem)}_step{it:06d}.pt")
            save_model(make_model(net, setup, frames, T), ck)
            r = dict(step=it, **score_frames(live_model(net, setup, T, device), hold))
            pd.DataFrame([r]).to_csv(prog_path, mode="a", header=not os.path.exists(prog_path), index=False)
            print(f"  >> step {it}: held-out recall {100*r['recall']:.2f}%  precision {100*r['precision']:.2f}%  "
                  f"({r['missed']} missed, {r['false']} false)  saved {ck}", flush=True)

        t0 = time.time()
        hard = hard_names(a.labels)
        is_hard = [f in hard for f, *_ in frames]
        weights = [a.hard_boost if h else 1.0 for h in is_hard]
        if any(is_hard):
            share = sum(w for w, h in zip(weights, is_hard) if h) / sum(weights)
            print(f"  {sum(is_hard)} hard frames (from hard.txt) drawn {a.hard_boost:g}x as often "
                  f"-> {100*share:.0f}% of training crops")
        lr = a.lr or (5e-4 if a.init else 2e-3)
        if a.init:
            print(f"  fine-tuning from {a.init} (peak learning rate {lr:g})")
        net = train(frames, setup, a.steps, a.batch, device, every=a.eval_every, callback=checkpoint,
                    weights=weights, init=a.init, lr=lr)
        m = make_model(net, setup, frames, T)
        save_model(m, a.out)
        print(f"saved {a.out} ({time.time() - t0:.0f} s)")
        if a.eval_every:
            checkpoint_final = dict(step=a.steps, **score_frames(live_model(net, setup, T, device), hold))
            pd.DataFrame([checkpoint_final]).to_csv(prog_path, mode="a", header=not os.path.exists(prog_path),
                                                    index=False)
    if not hold:
        return
    m = load_cnn(a.out, a.device)
    fm = None
    if a.compare_forest:
        print("training a forest on the same frames for comparison...")
        fm = forest_baseline(frames, setup)
    rows = []
    for f, g, b, _ in hold:
        crop = drum_crop(setup, g.shape)
        t0 = time.time(); dc = detect_cnn(m, g)[["x", "y"]].values; tc = time.time() - t0
        sc = bd.score(dc, b); bd._print_score(f"  {f} CNN   ", sc)
        rows.append(dict(frame=f, model="cnn", recall=sc["recall"], precision=sc["precision"],
                         missed=len(sc["fn"]), false=len(sc["fp"]), seconds=round(tc, 2)))
        tmp_c, tmp_f = os.path.join(rdir, "_c.png"), os.path.join(rdir, "_f.png")
        bd.draw_check(g, b, dc, sc, crop, tmp_c,
                      f"CNN  recall {100*sc['recall']:.1f}%  precision {100*sc['precision']:.1f}%  "
                      f"({len(sc['fn'])} missed, {len(sc['fp'])} false)")
        panels = [cv2.imread(tmp_c)]
        if fm is not None:
            bd.set_bead_px(setup["bead_px"])
            t0 = time.time(); df = bd.detect(fm, g)[["x", "y"]].values; tf = time.time() - t0
            sf = bd.score(df, b); bd._print_score(f"  {f} forest", sf)
            rows.append(dict(frame=f, model="forest", recall=sf["recall"], precision=sf["precision"],
                             missed=len(sf["fn"]), false=len(sf["fp"]), seconds=round(tf, 2)))
            bd.draw_check(g, b, df, sf, crop, tmp_f,
                          f"FOREST  recall {100*sf['recall']:.1f}%  precision {100*sf['precision']:.1f}%  "
                          f"({len(sf['fn'])} missed, {len(sf['fp'])} false)")
            panels.insert(0, cv2.imread(tmp_f))
        out = os.path.join(rdir, f"compare_{os.path.splitext(f)[0].replace('/', '__')}.png")
        cv2.imwrite(out, np.hstack(panels))
        print(f"  wrote {out}")
    for p in (os.path.join(rdir, "_c.png"), os.path.join(rdir, "_f.png")):
        if os.path.exists(p):
            os.remove(p)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(rdir, "compare_scores.csv"), index=False)
    print("\n" + df.groupby("model")[["recall", "precision", "missed", "false", "seconds"]].mean().round(3).to_string())
    if os.path.exists(prog_path):
        print("\n=== held-out score vs training steps (totals over the held-out frames) ===")
        print(pd.read_csv(prog_path).round(4).to_string(index=False))
    print(f"\n(everything from this run is in {rdir}/). For the real model, rerun WITHOUT --holdout "
          "so it trains on every frame, with the step count that scored best here.")


if __name__ == "__main__":
    main()