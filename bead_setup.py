"""
bead_setup.py - one-time setup per camera position: drum ROI + bead size.

    python bead_setup.py FRAME_DIR [--frame frame_000000.png] [--out bead_setup.json]

Two steps in one window (toolbar zoom OFF when clicking; zoom in as much as you like):

  STEP 1  DRUM WALL - left-click points on the INNER edge of the drum, where the outer beads
          press against it. Spread 6-10 points all the way round (not just along the bed).
          The fitted circle is drawn live (red). Right click = remove nearest point.
          Press n for step 2.

  STEP 2  BEAD SIZE - click the CENTRE of a bead, then the centre of a bead TOUCHING it.
          That's one pair (yellow line + its length). Do 5+ pairs in different parts of the bed.
          Right click = remove the last pair. b = back to step 1.

  Press w (or enter) to save. q = quit without saving.

Writes:
    bead_setup.json         bead_px (median pair length), drum rim circle, and the circle bead
                            CENTRES can't go past (rim radius - half a bead)
    bead_setup_check.png    the frame with both circles + your pairs drawn on, to check by eye

bead_trainer.py picks up bead_setup.json automatically (or --setup PATH): it scales every size
setting to bead_px and only looks for beads inside the drum.
"""
import argparse, json, os, re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FNAME_RE = re.compile(r"^(?:(\d{13})_)?frame_(\d+)\.(jpg|jpeg|png|tif+|bmp)$", re.I)


def fit_circle(p):
    p = np.asarray(p, float)
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    c = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    cx, cy = c[0], c[1]
    R = np.sqrt(c[2] + cx ** 2 + cy ** 2)
    resid = np.hypot(p[:, 0] - cx, p[:, 1] - cy) - R
    return float(cx), float(cy), float(R), resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("--frame", default=None, help="file name of the frame to use (default: middle one)")
    ap.add_argument("--out", default="bead_setup.json")
    a = ap.parse_args()

    files = sorted([f for f in os.listdir(a.frame_dir) if FNAME_RE.match(f)],
                   key=lambda f: int(FNAME_RE.match(f).group(2)))
    if not files:
        raise SystemExit("no frames found")
    fname = a.frame or files[len(files) // 2]
    gray = cv2.imread(os.path.join(a.frame_dir, fname), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"could not read {fname}")
    print(f"using {fname}  ({gray.shape[1]} x {gray.shape[0]} px)")

    for k in ("keymap.save", "keymap.quit", "keymap.back", "keymap.home", "keymap.fullscreen"):
        if k in plt.rcParams:
            plt.rcParams[k] = []
    st = dict(step=1, wall=[], pairs=[], pending=None, saved=False)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    artists = []

    def pitch():
        L = [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in st["pairs"]]
        return (float(np.median(L)), L) if L else (None, L)

    def redraw():
        for h in artists:
            h.remove()
        artists.clear()
        if st["wall"]:
            W = np.array(st["wall"])
            artists.append(ax.scatter(W[:, 0], W[:, 1], s=40, c="red", marker="+", lw=1.5))
        circ = None
        if len(st["wall"]) >= 3:
            cx, cy, R, res = fit_circle(st["wall"])
            circ = (cx, cy, R)
            artists.append(ax.add_patch(Circle((cx, cy), R, fill=False, ec="red", lw=1)))
            bp, _ = pitch()
            if bp:
                artists.append(ax.add_patch(Circle((cx, cy), R - bp / 2, fill=False, ec="yellow",
                                                   lw=0.8, ls="--")))
        for (x0, y0), (x1, y1) in st["pairs"]:
            artists.extend(ax.plot([x0, x1], [y0, y1], "-o", c="yellow", ms=3, lw=1))
            artists.append(ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{np.hypot(x1-x0, y1-y0):.1f}",
                                   color="yellow", fontsize=8))
        if st["pending"] is not None:
            artists.append(ax.scatter([st["pending"][0]], [st["pending"][1]], s=40, c="cyan"))
        bp, L = pitch()
        if st["step"] == 1:
            msg = (f"STEP 1 drum wall: click the INNER drum edge all the way round ({len(st['wall'])} pts"
                   + (f", circle R = {circ[2]:.1f} px, fit scatter {np.std(res):.1f} px" if circ else "")
                   + ")\nL-click add | R-click remove nearest | n = next step | q quit")
        else:
            msg = (f"STEP 2 bead size: click a bead centre, then a TOUCHING bead's centre "
                   f"({len(L)} pairs" + (f", median {bp:.2f} px" if bp else "") + ")\n"
                   "R-click remove last pair | b back | w or enter = SAVE | q quit")
        ax.set_title(msg, fontsize=10)
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or e.xdata is None or getattr(fig.canvas.toolbar, "mode", ""):
            return
        p = (float(e.xdata), float(e.ydata))
        if st["step"] == 1:
            if e.button == 1:
                st["wall"].append(p)
            elif e.button == 3 and st["wall"]:
                W = np.array(st["wall"])
                st["wall"].pop(int(np.hypot(W[:, 0] - p[0], W[:, 1] - p[1]).argmin()))
        else:
            if e.button == 1:
                if st["pending"] is None:
                    st["pending"] = p
                else:
                    st["pairs"].append((st["pending"], p)); st["pending"] = None
            elif e.button == 3:
                if st["pending"] is not None:
                    st["pending"] = None
                elif st["pairs"]:
                    st["pairs"].pop()
        redraw()

    def save():
        if len(st["wall"]) < 3:
            print("need at least 3 wall points"); return
        bp, L = pitch()
        if bp is None or len(L) < 3:
            print("need at least 3 bead pairs"); return
        cx, cy, R, res = fit_circle(st["wall"])
        out = dict(frame=fname, image_size=[int(gray.shape[1]), int(gray.shape[0])],
                   bead_px=round(bp, 2), bead_px_pairs=[round(v, 2) for v in L],
                   rim=[round(cx, 1), round(cy, 1), round(R, 1)],
                   wall=[round(cx, 1), round(cy, 1), round(R - bp / 2, 1)],
                   rim_points=[[round(x, 1), round(y, 1)] for x, y in st["wall"]])
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        # check image
        o = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.circle(o, (int(round(cx)), int(round(cy))), int(round(R)), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(o, (int(round(cx)), int(round(cy))), int(round(R - bp / 2)), (0, 255, 255), 1, cv2.LINE_AA)
        for x, y in st["wall"]:
            cv2.drawMarker(o, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for (x0, y0), (x1, y1) in st["pairs"]:
            cv2.line(o, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2, cv2.LINE_AA)
        txt = (f"bead_px {bp:.2f} (pairs {min(L):.1f}-{max(L):.1f})   rim R {R:.1f}   "
               f"red = drum rim, yellow = limit for bead centres")
        cv2.rectangle(o, (0, 0), (o.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(o, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        png = os.path.splitext(a.out)[0] + "_check.png"
        cv2.imwrite(png, o)
        print(f"saved {a.out} and {png}")
        print(f"  bead_px = {bp:.2f} px (from {len(L)} pairs, range {min(L):.1f}-{max(L):.1f})")
        print(f"  drum rim: centre ({cx:.1f}, {cy:.1f}), R = {R:.1f} px, fit scatter {np.std(res):.1f} px")
        if bp and abs(bp - 19) / 19 > 0.05:
            print(f"  NOTE: differs from the old 19 px - an old model will NOT work here, train a new one")
        st["saved"] = True
        plt.close(fig)

    def on_key(e):
        if e.key == "n" and st["step"] == 1:
            if len(st["wall"]) < 3:
                print("click at least 3 wall points first")
            else:
                st["step"] = 2
        elif e.key == "b" and st["step"] == 2:
            st["step"] = 1
        elif e.key in ("w", "enter") and st["step"] == 2:
            save(); return
        elif e.key == "q":
            plt.close(fig); return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    fig.tight_layout()
    plt.show()
    if not st["saved"]:
        print("not saved")


if __name__ == "__main__":
    main()