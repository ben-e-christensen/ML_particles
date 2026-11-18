"""
bead_detector.py - learned bead-centre detector (random forest on per-pixel image features,
same idea as ilastik). Trained from your hand-dotted labels; handles the uneven lighting
because it sees locally contrast-normalised features at several scales.

Bead size: every length below is tuned for beads BEAD_PX = 19 px apart (the old camera). For a
different camera/zoom, run bead_setup.py once - it measures the pitch, and set_bead_px() rescales
every length. Models remember the pitch they were trained at and re-apply it when loaded.

Train (with an honest held-out check first):
    python bead_detector.py train FRAME.jpg labels_frame0.csv --out bead_rf.joblib [--bead-px N]
      -> trains on the LEFT half of your labels, scores on the RIGHT half it never saw,
         writes holdout_check.png, then retrains on ALL labels and saves the model.

Run on any frame and eyeball it:
    python bead_detector.py detect bead_rf.joblib FRAME.jpg [--labels labels.csv]
      -> detect_<frame>.png (with --labels it also scores against them)

Add the lit + shadowed bead templates to an existing model (no labels needed, run once):
    python bead_detector.py upgrade bead_rf.joblib FRAME1.jpg FRAME2.jpg ... (about 5 frames
    spread over the run)
  or, better, from your approved labels:
    python bead_detector.py upgrade bead_rf.joblib IMAGE_FOLDER --labels bead_labels

Label-free precision numbers for two CONSECUTIVE frames:
    python bead_detector.py quality bead_rf.joblib FRAME_N.jpg FRAME_N+1.jpg

Used as a module by verify_ptv.py:  from bead_detector import load_model, detect
"""
import argparse, os, time
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from sklearn.ensemble import RandomForestClassifier
import joblib

BEAD_PX       = 19      # bead centre-to-centre pitch in raw px (measured from neighbour spacing)
REFINE_HALF   = 13      # half-size of the patch used for centre refinement
REFINE_SEARCH = 4       # template is cropped by this much so it can slide inside the patch
WIDE_SEARCH   = 6       # final refinement may move a centre at most this many px
MIN_SEP_PX    = 16.5    # two bead centres can never be closer than this (~0.86 pitch; real touching
                        # beads measure 17.5-19 px, so this only catches clear overlaps)
GROW_THRESH   = 0.3     # weaker forest peaks (>= this) are accepted ONLY if they sit one bead-width
                        # from an already confirmed bead and overlap nothing (tested on held-out frames:
                        # 0.3 recovers ~40% of misses for ~no extra false hits; 0.2 adds false hits)
TOUCH_RANGE   = (16.0, 22.0)   # px from a confirmed bead that counts as "touching it"
LONER_PROB    = 0.8     # a bead with fewer than 2 neighbours (even none) is kept if the forest is this sure
FILL_FLOOR    = 0.02    # weakest forest peak considered when filling up to a target count
FILL_TOUCH    = 22.0    # a filled-in bead must sit within this of an existing bead (no floating beads)
FILL_MIN_DIST = 6       # min px between fill-in candidates
WALL_TOL      = 5.0     # px a bead centre may sit beyond the fitted drum-wall circle (labels never exceed ~4)
DIM_MIN_COUNT = 15      # need at least this many shadowed beads to build a separate dim template
POS_RADIUS    = 2.5     # pixels this close to a label = "centre"
NEG_RADIUS    = 5.0     # pixels farther than this from every label = "not centre"
NEG_PER_POS   = 4
SIGMAS        = (1, 2, 3.5, 6, 9)
LCN_SIGMA     = 12      # local contrast normalisation scale (~0.7 bead) -> kills lighting gradient
PROB_SMOOTH   = 1.5
MIN_DIST      = 8       # min px between detected centres (~half a bead)
THRESH        = 0.5     # probability peak threshold
MATCH_GATE    = 6       # px: detection within this of a label counts as a hit
MIN_NEIGHBOURS = 2      # drop detections with fewer than this many others within 1.5 bead
                        # (kills isolated false hits on background/hardware; packed beads have 4-6)
ROI_PAD       = 60      # px around the labelled area used for training negatives
DETECT_PAD    = 150     # extra px around that area searched at detect time (bed rotates/shifts between frames)

_BASE_PX = 19.0         # the pitch all the numbers above were tuned at


def set_bead_px(px):
    """Rescale every length above for beads `px` apart (measured with bead_setup.py).
    Call before training/detecting. load_model() calls it with the model's own pitch."""
    global BEAD_PX, REFINE_HALF, REFINE_SEARCH, WIDE_SEARCH, MIN_SEP_PX, TOUCH_RANGE, FILL_TOUCH
    global FILL_MIN_DIST, WALL_TOL, POS_RADIUS, NEG_RADIUS, SIGMAS, LCN_SIGMA, PROB_SMOOTH
    global MIN_DIST, MATCH_GATE, ROI_PAD, DETECT_PAD
    s = float(px) / _BASE_PX
    BEAD_PX       = float(px)
    REFINE_SEARCH = max(2, int(round(4 * s)))
    WIDE_SEARCH   = max(3, int(round(6 * s)))
    REFINE_HALF   = max(REFINE_SEARCH + 3, int(round(13 * s)))
    MIN_SEP_PX    = 16.5 * s
    TOUCH_RANGE   = (16.0 * s, 22.0 * s)
    FILL_TOUCH    = 22.0 * s
    FILL_MIN_DIST = max(3, int(round(6 * s)))
    WALL_TOL      = 5.0 * s
    POS_RADIUS    = 2.5 * s
    NEG_RADIUS    = 5.0 * s
    SIGMAS        = tuple(v * s for v in (1, 2, 3.5, 6, 9))
    LCN_SIGMA     = 12 * s
    PROB_SMOOTH   = 1.5 * s
    MIN_DIST      = max(3, int(round(8 * s)))
    MATCH_GATE    = 6 * s
    ROI_PAD       = int(round(60 * s))
    DETECT_PAD    = int(round(150 * s))


def features(gray_u8):
    g = gray_u8.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(g, (0, 0), LCN_SIGMA)
    sd = np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), LCN_SIGMA)) + 1e-3
    lcn = (g - mu) / sd
    F = []
    for im in (g, lcn):
        for s in SIGMAS:
            F.append(ndi.gaussian_filter(im, s))
            F.append(ndi.gaussian_laplace(im, s))
            F.append(ndi.gaussian_gradient_magnitude(im, s))
            hxx = ndi.gaussian_filter(im, s, order=(0, 2))
            hyy = ndi.gaussian_filter(im, s, order=(2, 0))
            hxy = ndi.gaussian_filter(im, s, order=(1, 1))
            tr = (hxx + hyy) / 2
            dt = np.sqrt(((hxx - hyy) / 2) ** 2 + hxy ** 2)
            F.append(tr + dt); F.append(tr - dt)          # Hessian eigenvalues (blob shape)
    return np.stack(F, -1)


def sample_frame(gray, labels_xy, not_bead_xy=None, use_mask=None, seed=0):
    """Training rows from one labelled frame. Returns X, y, roi.
    not_bead_xy: places you explicitly rejected -> always included as negatives."""
    H, W = gray.shape
    X = features(gray)
    yy, xx = np.mgrid[0:H, 0:W]
    grid = np.c_[xx.ravel(), yy.ravel()]
    d = cKDTree(labels_xy).query(grid)[0].reshape(H, W)
    roi = np.zeros((H, W), bool)
    x0, y0 = np.maximum(labels_xy.min(0) - ROI_PAD, 0).astype(int)
    x1, y1 = np.minimum(labels_xy.max(0) + ROI_PAD, [W, H]).astype(int)
    roi[y0:y1, x0:x1] = True
    if use_mask is not None:
        roi &= use_mask
    pos = np.flatnonzero((roi & (d < POS_RADIUS)).ravel())
    negs = np.flatnonzero((roi & (d > NEG_RADIUS)).ravel())
    rng = np.random.default_rng(seed)
    neg = rng.choice(negs, min(len(negs), NEG_PER_POS * len(pos)), replace=False)
    if not_bead_xy is not None and len(not_bead_xy):
        dn = cKDTree(np.asarray(not_bead_xy)).query(grid)[0].reshape(H, W)
        hard = np.flatnonzero(((dn < POS_RADIUS + 1) & (d > NEG_RADIUS)).ravel())
        neg = np.unique(np.r_[neg, hard])
    Xf = X.reshape(-1, X.shape[-1])
    idx = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return Xf[idx].copy(), y, (int(x0), int(y0), int(x1), int(y1))


MAX_TRAIN_ROWS = 100_000   # cap on pooled training pixels so retraining time stays flat
                           # as you approve more frames (each frame gives ~50k rows)


def fit_samples(Xs, ys, rois, seed=0):
    """Fit one model on rows pooled from any number of labelled frames.
    Rows are subsampled evenly across frames to MAX_TRAIN_ROWS total, so every frame
    still contributes (more frames = more variety, not more time)."""
    rng = np.random.default_rng(seed)
    per = MAX_TRAIN_ROWS // len(Xs)
    Xp, yp = [], []
    for X, y in zip(Xs, ys):
        idx = rng.choice(len(y), per, replace=False) if len(y) > per else np.arange(len(y))
        Xp.append(X[idx]); yp.append(y[idx])
    clf = RandomForestClassifier(150, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    clf.fit(np.vstack(Xp), np.concatenate(yp))
    r = np.array(rois)
    roi = (int(r[:, 0].min()), int(r[:, 1].min()), int(r[:, 2].max()), int(r[:, 3].max()))
    return {"clf": clf, "roi": roi, "bead_px": BEAD_PX}


def train_model(gray, labels_xy, use_mask=None, seed=0):
    X, y, roi = sample_frame(gray, labels_xy, use_mask=use_mask, seed=seed)
    m = fit_samples([X], [y], [roi], seed)
    m["template"] = build_template([(gray, labels_xy)])
    m["templates"], m["dim_thresh"] = build_templates([(gray, labels_xy)])
    return m


def _lcn(gray_u8):
    g = gray_u8.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(g, (0, 0), LCN_SIGMA)
    sd = np.sqrt(cv2.GaussianBlur((g - mu) ** 2, (0, 0), LCN_SIGMA)) + 1e-3
    return (g - mu) / sd


def refine_centres(gray, xy, template):
    """Slide each rough centre (<= REFINE_SEARCH px) to where the local image best matches the
    average-bead template (normalised cross-correlation, sub-pixel). This is what sets the final
    position - the hand dots / clicks only need to say WHICH blob is a bead, not exactly where."""
    if template is None or len(xy) == 0:
        return np.asarray(xy, float)
    n = _lcn(gray)
    R, sr = REFINE_HALF, REFINE_SEARCH
    Tc = template[sr:-sr, sr:-sr]
    out = []
    for x, y in xy:
        p = cv2.getRectSubPix(n, (2 * R + 1, 2 * R + 1), (float(x), float(y)))
        res = cv2.matchTemplate(p, Tc, cv2.TM_CCOEFF_NORMED)
        _, _, _, loc = cv2.minMaxLoc(res)
        i, j = loc[1], loc[0]
        dx = dy = 0.0
        if 0 < j < res.shape[1] - 1:
            a, b, c = res[i, j - 1], res[i, j], res[i, j + 1]
            dx = float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-9), -1, 1))
        if 0 < i < res.shape[0] - 1:
            a, b, c = res[i - 1, j], res[i, j], res[i + 1, j]
            dy = float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-9), -1, 1))
        out.append((x + j - sr + dx, y + i - sr + dy))
    return np.array(out)


def build_template(frames_and_points, iters=3):
    """Average bead appearance from [(gray, xy), ...]; iterated so imprecise dots average out."""
    R = REFINE_HALF
    pts = [np.asarray(p, float) for _, p in frames_and_points]
    T = None
    for _ in range(iters):
        acc, cnt = 0, 0
        for (g, _), p in zip(frames_and_points, pts):
            n = _lcn(g)
            for x, y in p:
                acc = acc + cv2.getRectSubPix(n, (2 * R + 1, 2 * R + 1), (float(x), float(y))); cnt += 1
        T = (acc / cnt).astype(np.float32)
        pts = [refine_centres(g, p0, T) for (g, _), p0 in zip(frames_and_points, [np.asarray(p, float) for _, p in frames_and_points])]
    return T


def fit_wall(label_sets):
    """Drum wall as a circle through the outermost beads. label_sets = list of (n,2) arrays of
    approved bead centres, ideally from several frames. The camera and drum axis are fixed, so one
    circle holds for the whole run. Returns (cx, cy, R) where R is the radius at which the centres
    of beads touching the wall sit. No bead centre can be outside it."""
    from scipy.spatial import ConvexHull
    sets = [np.asarray(L, float) for L in label_sets if len(L) >= 10]
    if not sets:
        return None
    # the convex hull of a bed is its wall arc plus the straight chord across the free surface;
    # a straight chord contributes almost no hull corners, so the corners are nearly all wall beads
    pts = np.vstack([L[ConvexHull(L).vertices] for L in sets])
    keep = np.ones(len(pts), bool)
    for _ in range(8):
        p = pts[keep]
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        c = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
        cx, cy, R = c[0], c[1], np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R
        keep = np.abs(r) < max(2.0, 2.5 * 1.4826 * np.median(np.abs(r[keep])))
    return float(cx), float(cy), float(R)


def inside_wall(model, xy, extra=0.0):
    """True for points that are inside the drum (or everything, if the model has no wall yet)."""
    w = model.get("wall") if model is not None else None
    xy = np.asarray(xy, float)
    if w is None or len(xy) == 0:
        return np.ones(len(xy), bool)
    return np.hypot(xy[:, 0] - w[0], xy[:, 1] - w[1]) <= w[2] + WALL_TOL + extra


def bead_brightness(gray, xy):
    """Smoothed raw brightness at each centre - separates shadowed beads from lit ones."""
    b = cv2.GaussianBlur(gray, (0, 0), 5)
    xy = np.asarray(xy)
    return b[xy[:, 1].round().astype(int).clip(0, gray.shape[0] - 1),
             xy[:, 0].round().astype(int).clip(0, gray.shape[1] - 1)].astype(float)


def _otsu_split(vals):
    """Threshold that best splits brightness values into two groups; None if not clearly bimodal."""
    # work in log-brightness: shadowed beads are several times darker, not a fixed amount darker
    v = np.sort(np.log(np.maximum(np.asarray(vals, float), 1.0)))
    if len(v) < 2 * DIM_MIN_COUNT:
        return None
    best, bt = -1, None
    for t in np.linspace(v[DIM_MIN_COUNT], v[-DIM_MIN_COUNT], 60):
        lo, hi = v[v < t], v[v >= t]
        if len(lo) < DIM_MIN_COUNT or len(hi) < DIM_MIN_COUNT:
            continue
        sep = len(lo) * len(hi) * (hi.mean() - lo.mean()) ** 2
        if sep > best:
            best, bt = sep, t
    if bt is None:
        return None
    lo, hi = v[v < bt], v[v >= bt]
    # only call it bimodal if the groups are well separated compared to their own widths
    return float(np.exp(bt)) if (hi.mean() - lo.mean()) > 2.0 * (lo.std() + hi.std()) else None


def build_templates(frames_and_points, iters=3):
    """Average-bead templates, one for lit beads and (if there are enough) one for shadowed beads,
    which look different: no highlight, softer rim. Returns (templates list, dim_threshold or None)."""
    fp = [(g, np.asarray(p, float)) for g, p in frames_and_points]
    br = np.concatenate([bead_brightness(g, p) for g, p in fp])
    thr = _otsu_split(br)
    if thr is None:
        return [build_template(fp, iters)], None
    lit = [(g, p[bead_brightness(g, p) >= thr]) for g, p in fp]
    dim = [(g, p[bead_brightness(g, p) < thr]) for g, p in fp]
    return [build_template(lit, iters), build_template(dim, iters)], thr


def refine_constrained(gray, xy, templates, dmin=None, sr=None):
    """Final centring. For every rough centre, score each position within +-sr px against every
    template (best template wins). Then place beads from MOST certain to LEAST certain; a bead may
    not be placed closer than dmin to one already placed, so a well-seen bead stops its poorly-seen
    neighbour from sliding onto it. Returns (xy, certainty); certainty = match score 0..1.
    Beads with no allowed position left are dropped (xy = nan).
    dmin / sr default to MIN_SEP_PX / WIDE_SEARCH at call time (so set_bead_px applies)."""
    dmin = MIN_SEP_PX if dmin is None else dmin
    sr = WIDE_SEARCH if sr is None else sr
    xy = np.asarray(xy, float)
    if len(xy) == 0:
        return xy, np.zeros(0)
    n = _lcn(gray)
    R = REFINE_HALF + sr - REFINE_SEARCH
    Tcs = [T[REFINE_SEARCH:-REFINE_SEARCH, REFINE_SEARCH:-REFINE_SEARCH] for T in templates]
    M = np.empty((len(xy), 2 * sr + 1, 2 * sr + 1), np.float32)
    for q, (x, y) in enumerate(xy):
        p = cv2.getRectSubPix(n, (2 * R + 1, 2 * R + 1), (float(x), float(y)))
        M[q] = np.max([cv2.matchTemplate(p, Tc, cv2.TM_CCOEFF_NORMED) for Tc in Tcs], axis=0)
    order = np.argsort(-M.reshape(len(xy), -1).max(1))
    X = np.full_like(xy, np.nan)
    cert = np.zeros(len(xy))
    nb = cKDTree(xy).query_ball_point(xy, dmin + 2 * sr + 2)
    off = np.arange(-sr, sr + 1)
    gx, gy = np.meshgrid(off, off)
    for q in order:
        res = M[q].copy()
        for k in nb[q]:
            if k != q and np.isfinite(X[k, 0]):
                res[np.hypot(xy[q, 0] + gx - X[k, 0], xy[q, 1] + gy - X[k, 1]) < dmin] = -np.inf
        if not np.isfinite(res).any():
            continue
        i, j = np.unravel_index(np.argmax(res), res.shape)
        dx = dy = 0.0
        if 0 < j < 2 * sr and np.isfinite(res[i, j - 1]) and np.isfinite(res[i, j + 1]):
            a, b, c = res[i, j - 1], res[i, j], res[i, j + 1]
            dx = float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-9), -1, 1))
        if 0 < i < 2 * sr and np.isfinite(res[i - 1, j]) and np.isfinite(res[i + 1, j]):
            a, b, c = res[i - 1, j], res[i, j], res[i + 1, j]
            dy = float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-9), -1, 1))
        X[q] = (xy[q, 0] + off[j] + dx, xy[q, 1] + off[i] + dy)
        cert[q] = float(res[i, j])
    return X, cert


def spacing_spread(xy):
    """Label-free precision check: spread (robust sigma, px) of touching-neighbour distances.
    Beads are 3.175 mm +/- 4%, so in a packed bed this is small if centres are right."""
    pr = np.array(list(cKDTree(xy).query_pairs(1.25 * BEAD_PX)))
    d = np.hypot(*(xy[pr[:, 0]] - xy[pr[:, 1]]).T)
    d = d[d > 0.75 * BEAD_PX]
    med = np.median(d)
    return med, 1.4826 * np.median(np.abs(d - med))


def rigid_jitter(xy_a, xy_b):
    """Label-free precision check: match beads between two consecutive frames, fit one rigid
    motion to all of them, return per-bead per-frame scatter about it (px). Includes any real
    non-rigid motion, so it is an upper bound on position noise."""
    dist, j = cKDTree(xy_b).query(xy_a)
    ok = dist < BEAD_PX / 2
    a, b = xy_a[ok], xy_b[j[ok]]
    for _ in range(3):
        ca, cb = a.mean(0), b.mean(0)
        U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        r = np.hypot(*((a - ca) @ (Vt.T @ U.T).T + cb - b).T)
        keep = r < 3 * np.median(r) + 1e-9
        if keep.all(): break
        a, b = a[keep], b[keep]
    return float(np.sqrt(np.mean(r[keep] ** 2)) / np.sqrt(2)), int(keep.sum())


def load_model(path):
    """Load a model and switch every length setting to the bead pitch it was trained at
    (older models without a stored pitch were all trained at 19 px)."""
    m = joblib.load(path)
    set_bead_px(m.get("bead_px", _BASE_PX))
    return m


def prob_map(model, gray, roi=None):
    """roi = (x0,y0,x1,y1) to only predict there (much faster). Default: the drum box from
    bead_setup.py if the model has one (roi_exact), else the training ROI + DETECT_PAD."""
    H, W = gray.shape
    if roi is None and model.get("roi_exact") is not None:
        x0, y0, x1, y1 = model["roi_exact"]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    elif roi is None:
        x0, y0, x1, y1 = model["roi"]
        x0, y0 = max(x0 - DETECT_PAD, 0), max(y0 - DETECT_PAD, 0)
        x1, y1 = min(x1 + DETECT_PAD, W), min(y1 + DETECT_PAD, H)
    else:
        x0, y0, x1, y1 = roi
    X = features(gray[y0:y1, x0:x1])
    p = model["clf"].predict_proba(X.reshape(-1, X.shape[-1]))[:, 1].reshape(y1 - y0, x1 - x0)
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = cv2.GaussianBlur(p.astype(np.float32), (0, 0), PROB_SMOOTH)
    return out


def fill_to_count(gray, P, xy, Ts, target, model=None):
    """Add the best remaining candidates until there are `target` beads. A candidate must be a
    forest peak >= FILL_FLOOR, fit the bead template, touch an existing bead and overlap none.
    Ranked by forest prob x template match. Returns (added_xy, added_score)."""
    X = np.asarray(xy, float)
    cand = peak_local_max(P, min_distance=FILL_MIN_DIST, threshold_abs=FILL_FLOOR,
                          exclude_border=False)[:, ::-1].astype(float)
    cand = cand[inside_wall(model, cand)]
    added, scores = [], []
    for _ in range(10):
        if len(X) >= target or not len(cand):
            break
        cand = cand[cKDTree(X).query(cand)[0] > 0.6 * MIN_SEP_PX]
        if not len(cand):
            break
        Xn, cn = refine_constrained(gray, np.vstack([X, cand]), Ts)
        C, cc = Xn[len(X):], cn[len(X):]
        ok = np.isfinite(C[:, 0])
        C, cc, src = C[ok], cc[ok], cand[ok]
        sc = P[src[:, 1].astype(int), src[:, 0].astype(int)] * np.clip(cc, 0, 1)
        tree = cKDTree(X)
        new = []
        for k in np.argsort(-sc):
            if len(X) + len(new) >= target:
                break
            p = C[k]
            d = tree.query(p)[0]
            if d < MIN_SEP_PX - 0.01 or d > FILL_TOUCH:
                continue
            if not inside_wall(model, p[None], extra=1.0)[0]:
                continue
            if any(np.hypot(*(p - q)) < MIN_SEP_PX for q in new):
                continue
            new.append(p); scores.append(float(sc[k]))
        if not new:
            break
        added += new
        X = np.vstack([X, np.array(new)])
    return np.array(added).reshape(-1, 2), np.array(scores)


def detect(model, gray, roi=None, thresh=THRESH, return_prob=False, refine=True, target=None):
    """Returns DataFrame x, y, prob, certainty, brightness, grown, forced (raw-frame px).
    refine=True does the certainty-ordered, no-overlap template centring (refine_constrained).
    target=500: if fewer beads were found, add the best remaining candidates up to that count
    (marked forced=True - on held-out frames only ~40% of these were real beads, see README)."""
    P = prob_map(model, gray, roi)
    pk = peak_local_max(P, min_distance=MIN_DIST, threshold_abs=thresh, exclude_border=False)
    xy = pk[:, ::-1].astype(float)
    # sub-pixel: probability-weighted centroid in a 5x5 patch
    ref = []
    for x, y in xy.astype(int):
        # clip the 5x5 patch to the image so peaks at the frame edge don't break it
        ys = slice(max(y - 2, 0), min(y + 3, P.shape[0]))
        xs = slice(max(x - 2, 0), min(x + 3, P.shape[1]))
        w = P[ys, xs]; gy, gx = np.mgrid[ys, xs]
        ref.append(((w * gx).sum() / w.sum(), (w * gy).sum() / w.sum()))
    xy = np.array(ref) if ref else np.zeros((0, 2))
    if len(xy):
        xy = xy[inside_wall(model, xy)]           # nothing can be outside the drum wall
    def pval(pts):
        return P[pts[:, 1].round().astype(int).clip(0, P.shape[0] - 1),
                 pts[:, 0].round().astype(int).clip(0, P.shape[1] - 1)]
    if len(xy) and MIN_NEIGHBOURS:
        nn = cKDTree(xy).query_ball_point(xy, 1.5 * BEAD_PX, return_length=True) - 1
        # packed beads have several neighbours; a lone bead on the surface has one, so keep
        # those too when the forest is confident
        # ...and a bead the forest is confident about is kept wherever it is, even with no
        # neighbours at all (a bead rolling alone down the surface, or in flight at higher rpm)
        xy = xy[(nn >= MIN_NEIGHBOURS) | (pval(xy) >= LONER_PROB)]
    cert = np.full(len(xy), np.nan)
    grown = np.zeros(len(xy), bool)
    if refine and len(xy) >= 20:
        Ts = model.get("templates")
        if not Ts:
            T = model.get("template")
            Ts = [T] if T is not None else [build_template([(gray, xy)])]
        seeds = xy
        xy, cert = refine_constrained(gray, xy, Ts)
        ok = np.isfinite(xy[:, 0])
        ok[ok] &= inside_wall(model, xy[ok], extra=1.0)
        xy, cert = xy[ok], cert[ok]
        grown = np.zeros(len(xy), bool)
        # ---- grow: let confirmed beads vouch for weak candidates next to them ----
        if GROW_THRESH and GROW_THRESH < thresh and len(xy):
            cand = peak_local_max(P, min_distance=MIN_DIST, threshold_abs=GROW_THRESH,
                                  exclude_border=False)[:, ::-1].astype(float)
            if len(cand):
                cand = cand[inside_wall(model, cand)]
            if len(cand):
                cand = cand[cKDTree(seeds).query(cand)[0] > MIN_DIST]      # not already used
            for _ in range(6):
                if not len(cand):
                    break
                d = cKDTree(xy).query(cand)[0]
                touch = (d >= TOUCH_RANGE[0]) & (d <= TOUCH_RANGE[1])
                if not touch.any():
                    break
                new = cand[touch]
                Xn, cn = refine_constrained(gray, np.vstack([xy, new]), Ts)
                add, addc = Xn[len(xy):], cn[len(xy):]
                good, goodc = [], []
                tree = cKDTree(xy)
                for p, c in zip(add, addc):
                    if not np.isfinite(p[0]) or tree.query(p)[0] < MIN_SEP_PX - 0.01:
                        continue
                    if not inside_wall(model, p[None], extra=1.0)[0]:
                        continue
                    if all(np.hypot(*(p - q)) >= MIN_SEP_PX for q in good):
                        good.append(p); goodc.append(c)
                cand = cand[~touch]
                if not good:
                    break
                xy = np.vstack([xy, np.array(good)])
                cert = np.r_[cert, goodc]
                grown = np.r_[grown, np.ones(len(good), bool)]
    forced = np.zeros(len(xy), bool)
    if refine and target and 20 <= len(xy) < target:
        add, addc = fill_to_count(gray, P, xy, Ts, target, model)
        if len(add):
            xy = np.vstack([xy, add]); cert = np.r_[cert, addc]
            grown = np.r_[grown, np.zeros(len(add), bool)]
            forced = np.r_[forced, np.ones(len(add), bool)]
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    if len(df):
        df["prob"] = P[df["y"].round().astype(int).clip(0, P.shape[0] - 1),
                        df["x"].round().astype(int).clip(0, P.shape[1] - 1)]
        df["certainty"] = cert                    # template match score, higher = better seen
        df["brightness"] = bead_brightness(gray, xy)
        df["grown"] = grown                       # True = weak hit accepted because it touches a confirmed bead
        df["forced"] = forced                     # True = added only to reach the target count
    return (df, P) if return_prob else df


def score(det_xy, lab_xy, gate=None):
    gate = MATCH_GATE if gate is None else gate
    C = cdist(lab_xy, det_xy) if len(det_xy) else np.full((len(lab_xy), 0), 1e6)
    C[C > gate] = 1e6
    r, c = linear_sum_assignment(C) if C.size else (np.array([], int), np.array([], int))
    ok = C[r, c] < 1e6
    tp_l, tp_d = r[ok], c[ok]
    fn = np.setdiff1d(np.arange(len(lab_xy)), tp_l)
    fp = np.setdiff1d(np.arange(len(det_xy)), tp_d)
    err = np.hypot(*(det_xy[tp_d] - lab_xy[tp_l]).T) if ok.any() else np.array([np.nan])
    return dict(recall=len(tp_l) / max(len(lab_xy), 1), precision=len(tp_l) / max(len(det_xy), 1),
                n_lab=len(lab_xy), n_det=len(det_xy), fn=fn, fp=fp, tp_l=tp_l, tp_d=tp_d,
                med_err=float(np.median(err)))


def draw_check(gray, lab, det, sc, crop, path, title, forced=None):
    """green circle = detection, red + = your label, BLUE ring = missed bead, YELLOW ring = false hit"""
    x0, y0, x1, y1 = crop
    k = 4
    o = cv2.resize(cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR), None, fx=k, fy=k,
                   interpolation=cv2.INTER_CUBIC)
    P = lambda x, y: (int((x - x0) * k), int((y - y0) * k))
    if lab is not None:
        for x, y in lab: cv2.drawMarker(o, P(x, y), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
    for x, y in det:   # bead-sized outline + centre dot, so you can judge the centring by eye
        cv2.circle(o, P(x, y), int(BEAD_PX / 2 * k), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(o, P(x, y), 2, (0, 255, 0), -1)
    if forced is not None:   # filled-in beads: magenta, so you can judge them separately
        for x, y in forced:
            cv2.circle(o, P(x, y), int(BEAD_PX / 2 * k), (255, 0, 255), 2, cv2.LINE_AA)
    if sc is not None:
        for x, y in lab[sc["fn"]]: cv2.circle(o, P(x, y), int(BEAD_PX * k * 0.7), (255, 0, 0), 3)
        for x, y in det[sc["fp"]]: cv2.circle(o, P(x, y), int(BEAD_PX * k * 0.7), (0, 255, 255), 3)
    cv2.rectangle(o, (0, 0), (o.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(o, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, o)


def _print_score(name, sc):
    print(f"{name}: recall {100*sc['recall']:.1f}%  precision {100*sc['precision']:.1f}%  "
          f"({sc['n_lab']} labels, {sc['n_det']} detections, {len(sc['fn'])} missed, "
          f"{len(sc['fp'])} false)  median offset from your dot {sc['med_err']:.2f} px")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("frame"); t.add_argument("labels")
    t.add_argument("--out", default="bead_rf.joblib")
    t.add_argument("--bead-px", type=float, default=None, help="bead pitch in px (default 19)")
    d = sub.add_parser("detect"); d.add_argument("model"); d.add_argument("frame")
    d.add_argument("--labels", default=None); d.add_argument("--thresh", type=float, default=THRESH)
    d.add_argument("--target", type=int, default=None, help="fill up to this many beads (e.g. 500)")
    q = sub.add_parser("quality"); q.add_argument("model"); q.add_argument("frame_a"); q.add_argument("frame_b")
    u = sub.add_parser("upgrade"); u.add_argument("model"); u.add_argument("frames", nargs="+")
    u.add_argument("--labels", default=None,
                   help="bead_labels folder: build templates from your approved beads instead of "
                        "from detections (then give the image FOLDER as the frames argument)")
    a = ap.parse_args()

    if a.cmd == "upgrade":
        # (re)build the lit + shadowed bead templates from the model's own detections on these
        # frames. Needs no labels. Run once on an existing model; pick ~5 frames spread over the run.
        m = load_model(a.model)
        fp = []
        if a.labels:
            import glob
            for c in sorted(glob.glob(os.path.join(a.labels, "*frame_*.csv"))):
                stem = os.path.splitext(os.path.basename(c))[0]
                hits = glob.glob(os.path.join(a.frames[0], stem + ".*"))
                if not hits:
                    print(f"  (no image for {stem}, skipped)"); continue
                d = pd.read_csv(c)
                if "kind" in d:
                    d = d[d.kind == "bead"]
                fp.append((cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE), d[["x", "y"]].values))
                print(f"  {stem}: {len(d)} approved beads")
            if not fp:
                raise SystemExit("no labelled frames found")
        else:
            for f in a.frames:
                g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                xy = detect(m, g, refine=False)[["x", "y"]].values
                print(f"  {os.path.basename(f)}: {len(xy)} beads")
                fp.append((g, xy))
        m["templates"], m["dim_thresh"] = build_templates(fp)
        m["template"] = m["templates"][0]
        if a.labels and not m.get("wall_from_setup"):
            m["wall"] = fit_wall([p for _, p in fp])
            if m["wall"]:
                print("drum wall circle: centre (%.1f, %.1f), bead-centre radius %.1f px" % m["wall"])
        joblib.dump(m, a.model, compress=3)
        if m["dim_thresh"] is None:
            print("no clearly separate shadowed group found -> one template")
        else:
            nd = sum(int((bead_brightness(g, p) < m["dim_thresh"]).sum()) for g, p in fp)
            print(f"built lit + shadowed templates (brightness split at {m['dim_thresh']:.0f}, "
                  f"{nd} shadowed beads used)")
        vis = lambda T: cv2.resize(cv2.normalize(T, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                   None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("templates.png", np.hstack([vis(T) for T in m["templates"]]))
        print(f"saved {a.model} and templates.png (left = lit bead, right = shadowed bead)")
        return

    if a.cmd == "quality":
        m = load_model(a.model)
        ga = cv2.imread(a.frame_a, cv2.IMREAD_GRAYSCALE); gb = cv2.imread(a.frame_b, cv2.IMREAD_GRAYSCALE)
        print("Label-free centre precision (uses NO hand dots). frame_a and frame_b must be consecutive.")
        thr = m.get("dim_thresh")
        for name, rf in (("raw detector    ", False), ("after refinement", True)):
            da = detect(m, ga, refine=rf); db = detect(m, gb, refine=rf)
            A = da[["x", "y"]].values; B = db[["x", "y"]].values
            groups = [("all beads", np.ones(len(A), bool))]
            if thr is not None:
                dim = bead_brightness(ga, A) < thr
                groups += [("lit      ", ~dim), ("shadowed ", dim)]
            close = len(cKDTree(A).query_pairs(MIN_SEP_PX - 0.5))
            print(f"  {name}: {len(A)} beads, {close} overlapping pairs")
            for gname, sel in groups:
                if sel.sum() < 10:
                    continue
                med, sp = spacing_spread(A[sel]); jit, n = rigid_jitter(A[sel], B)
                print(f"      {gname}: n={sel.sum():4d} | neighbour spacing {med:.2f} px, spread {sp:.2f} px "
                      f"| frame-to-frame jitter {jit:.3f} px")
        return

    if a.cmd == "train":
        if a.bead_px:
            set_bead_px(a.bead_px)
        gray = cv2.imread(a.frame, cv2.IMREAD_GRAYSCALE)
        lab = pd.read_csv(a.labels)[["x", "y"]].values
        H, W = gray.shape
        xs = np.median(lab[:, 0])
        print(f"{len(lab)} labels. Holdout check: train on x < {xs:.0f}, test on x > {xs:.0f}")
        t0 = time.time()
        left = np.zeros((H, W), bool); left[:, :int(xs) - 10] = True
        m = train_model(gray, lab[lab[:, 0] < xs], use_mask=left)
        m["roi"] = (max(int(lab[:, 0].min()) - ROI_PAD, 0), max(int(lab[:, 1].min()) - ROI_PAD, 0),
                    min(int(lab[:, 0].max()) + ROI_PAD, W), min(int(lab[:, 1].max()) + ROI_PAD, H))
        det = detect(m, gray)[["x", "y"]].values
        print(f"  trained + predicted in {time.time()-t0:.1f} s")
        edge = 5   # ignore a strip at the split so half-beads don't count
        dt = det[det[:, 0] > xs + edge]; lt = lab[lab[:, 0] > xs + edge]
        dtr = det[det[:, 0] < xs - edge]; ltr = lab[lab[:, 0] < xs - edge]
        sc_te = score(dt, lt); sc_tr = score(dtr, ltr)
        _print_score("  TRAIN half (seen)  ", sc_tr)
        _print_score("  TEST half (unseen) ", sc_te)
        crop = (max(int(xs) - 20, 0), max(int(lab[:, 1].min()) - 40, 0),
                min(int(lab[:, 0].max()) + 40, W), min(int(lab[:, 1].max()) + 40, H))
        draw_check(gray, lt, dt, sc_te, crop, "holdout_check.png",
                   f"UNSEEN half: recall {100*sc_te['recall']:.1f}%  precision {100*sc_te['precision']:.1f}%"
                   "   blue ring=missed  yellow ring=false")
        print("  wrote holdout_check.png")

        print("Training final model on ALL labels...")
        m = train_model(gray, lab)
        joblib.dump(m, a.out, compress=3)
        print(f"  saved {a.out}")
        med, sp = spacing_spread(lab)
        det = detect(m, gray)[["x", "y"]].values
        med2, sp2 = spacing_spread(det)
        print(f"  neighbour-spacing spread: your dots {sp:.2f} px  ->  refined detections {sp2:.2f} px "
              f"(pitch {med2:.2f} px)")

    else:
        m = load_model(a.model)
        gray = cv2.imread(a.frame, cv2.IMREAD_GRAYSCALE)
        t0 = time.time()
        dd = detect(m, gray, thresh=a.thresh, target=a.target)
        det = dd[["x", "y"]].values
        nf = int(dd["forced"].sum()) if "forced" in dd else 0
        print(f"{len(det) - nf} beads detected" + (f", +{nf} filled in to reach {a.target}" if a.target else "")
              + f" ({time.time()-t0:.1f} s)")
        lab = sc = None
        title = f"{len(det)} detections"
        if a.labels:
            lab = pd.read_csv(a.labels)[["x", "y"]].values
            sc = score(det, lab); _print_score("vs labels", sc)
            title += f"  recall {100*sc['recall']:.1f}%  precision {100*sc['precision']:.1f}%"
            print("  (if these labels were used for training this is NOT an independent test)")
        H, W = gray.shape
        x0, y0, x1, y1 = m["roi"]
        x0, y0 = max(x0 - DETECT_PAD, 0), max(y0 - DETECT_PAD, 0)
        x1, y1 = min(x1 + DETECT_PAD, W), min(y1 + DETECT_PAD, H)
        out = f"detect_{os.path.splitext(os.path.basename(a.frame))[0]}.png"
        draw_check(gray, lab, det, sc, (x0, y0, x1, y1), out, title,
                   forced=det[dd["forced"].values] if nf else None)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()