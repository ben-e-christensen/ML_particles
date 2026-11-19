# ML_particles: bead detection cheat sheet

Commands are for the MSYS / Git Bash shell, run from the repo folder with the venv active.
Paths use forward slashes (Python is fine with them on Windows).

## Every session: set these once, then copy-paste the commands below

```bash
source .venv/Scripts/activate

RUN="C:/Users/ben/Desktop/burst_test_2026-11-17_17_20/rpm06"   # the frame folder you're working on
M=bead_rf_uniform.joblib      # forest model (the trainer's guesser)
L=bead_labels_uniform         # labels folder (one subfolder per run, made automatically)
```

To switch rpm, just change `RUN` and rerun the commands.

## Labelling (bead_trainer.py)

```bash
# label new frames: every 40th frame starting at 20 (change --start for fresh ones: 10, 30, ...)
python bead_trainer.py "$RUN" --model $M --labels $L --step 40 --start 20

# fix one frame you already approved
python bead_trainer.py "$RUN" --model $M --labels $L --redo frame_000024.png

# step through every approved frame in this run
python bead_trainer.py "$RUN" --model $M --labels $L --review
```

Keys: left click = add, right click = remove, `u` undo, `a` approve, `s` skip, `r` reset, `q` quit.
Right-click the false hits (disks, etc.). They're saved as "not a bead" examples.

## Neural network (bead_cnn.py)

Uses every run in the labels folder. About 11 min on the RTX 3060.

```bash
# train the real model on all approved frames
python bead_cnn.py train --labels $L --out bead_cnn.pt

# fair test: hold 3 frames out, score network vs forest on them -> bead_cnn_test_report/
python bead_cnn.py train --labels $L --holdout 3 --compare-forest --out bead_cnn_test.pt

# how many steps? ONE long run that saves + scores held-out frames every 2500 steps.
# Everything goes in bead_cnn_30k_report/ (named after --out), so runs never overwrite each other:
#   progress.csv = held-out score at each checkpoint, *_stepNNNNNN.pt = the checkpoints
python bead_cnn.py train --labels $L --holdout 6 --eval-every 2500 --steps 30000 --out bead_cnn_30k.pt

# rerun that comparison without retraining (same --holdout)
python bead_cnn.py train --labels $L --holdout 3 --compare-forest --out bead_cnn_test.pt --skip-train

# run on one frame and look at it -> cnn_<frame>.png
python bead_cnn.py detect bead_cnn.pt "$RUN/frame_000100.png"
```

## One-time / rare

```bash
# new camera position or zoom: click the drum edge + touching bead pairs -> bead_setup.json
python bead_setup.py "$RUN"

# old labels sitting directly in the labels folder: move them into their run's subfolder
python bead_trainer.py --adopt-flat $L "C:/Users/ben/Desktop/burst_test_2026-11-17_17_20/rpm01"

# save the package list after installing something
python -m pip freeze > requirements.txt
```

## New machine

```bash
git clone <repo url> && cd ML_particles
py install 3.13
py -V:3.13 -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

PyTorch: with an NVIDIA GPU, install the CUDA build using the command from
https://pytorch.org/get-started/locally/ (a plain pip install on Windows is CPU-only). Check it:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Without a GPU, everything still runs on CPU, just slower.

**Copy these too:** git doesn't have them unless you committed them:

- `bead_setup.json`
- the labels folder (`bead_labels_uniform/`)
- the models (`*.joblib`, `*.pt`)
- the frame folders (burst_test_...)

**Frame folder paths matter:** each run's `run_info.json` (in `bead_labels_uniform/<run>/`) stores the
full path to its frames. If the frames live somewhere else on the new machine, either put them at
the same path, or edit `"frame_dir"` in each `run_info.json`. Otherwise those labels get skipped
(with a warning).

## Files

| file | what |
|---|---|
| `bead_setup.py` | drum circle + bead size for a camera position -> `bead_setup.json` |
| `bead_trainer.py` | label frames: it guesses, you fix, it retrains the forest |
| `bead_detector.py` | the forest detector (used by the trainer and the tracking scripts) |
| `bead_cnn.py` | neural-network detector trained from the same labels |