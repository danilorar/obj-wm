# Playground guide

**5-minute guide** to running single scenes and exploring WM predictions interactively.

> Dev / mini scenes only — never test scenes (`sim/scenarios/nuscenes_test150/`).

---

## Quick start

```bash
# 1. Set up both venvs (once)
make setup

# 2. Download nuScenes mini data and convert to ScenarioNet format
#    (see README § Reproduce → Quickstart on nuScenes mini for the data download)
make quickstart          # builds mini dataset, trains a mini model, converts scenes

# 3. Run one scene and get a GIF
make play SCENE=0 PLANNER=wm       # WM planner, scene index 0
make play SCENE=0 PLANNER=cv       # CV planner, same scene
make play SCENE=0 PLANNER=wm LIVE=1   # open the GIF when done
```

GIFs land in `outputs/play/`.

---

## Available scenes

Scenes are referenced by **integer index** into `sim/scenarios/nuscenes_mini/`.
After `make quickstart` (or `sim_check.py convert --scenes 10`) there are 10 scenes
(indices 0–9).  The nuScenes scene token for each index is printed by:

```bash
.venv_sim/bin/python - <<'EOF'
from pathlib import Path
import pickle
d = Path('sim/scenarios/nuscenes_mini')
s = pickle.load(open(d / 'dataset_summary.pkl', 'rb'))
for i, (k, v) in enumerate(s['scenario_summary'].items()):
    print(i, k)
EOF
```

---

## Playground notebook

Open `playground.ipynb` in a Jupyter server with the `.venv_sim` kernel:

```bash
.venv_sim/bin/pip install jupyter ipykernel
.venv_sim/bin/python -m ipykernel install --user --name venv_sim --display-name ".venv_sim"
.venv_sim/bin/jupyter notebook playground.ipynb
```

The notebook walks through:
1. Loading the v2 checkpoint.
2. Starting a mini scene and running 20 warm-up steps (2 s of history).
3. Generating all 35 candidate plans in one call.
4. Calling `predict()` once to get agent trajectories under every plan.
5. Plotting the 4 highlight plans (brake / keep speed / accelerate / left offset)
   side by side, with each agent's 6 predicted modes (opacity ∝ probability).
6. Printing how the top-mode probability and estimated speed change across plans.
7. An editable cell where you pick any (speed, offset) pair and see the result.

---

## `make play` internals

```
make play SCENE=<idx> PLANNER={cv,wm} [LIVE=1]
```

Calls:

```bash
.venv_sim/bin/python scripts/sim_check.py plan \
    --predictor <PLANNER> \
    --scenes <idx> \
    --gif <idx> \
    --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth \
    --out outputs/play/<PLANNER>_scene<idx>.json
```

Output: `outputs/play/plan_<PLANNER>_scene<idx>.gif` (top-down planning view,
one frame per 0.5 s replanning step) and a JSON with per-step metrics.

Pass `CK=<path>` to use a different checkpoint, e.g. one trained with `make quickstart`:

```bash
make play SCENE=0 PLANNER=wm CK=checkpoints/mini/v2_seed42_best.pth
```

---

## `make drive` internals

```
make drive SCENE=<idx>
```

Runs the existing `sim_check.py drive` function: a constant-action drive test
(fixed throttle 0.2, steering 0.0) for 10 s, saving `outputs/sim/drive_test.gif`.
This is a sanity check that the simulator loads and steps correctly.

**Keyboard-controlled interactive drive** (MetaDrive with WM predictions overlaid)
was considered but requires a live display and significant additional development —
it is not currently implemented.

---

## Planner v2.1 flags

You can pass the v2.1 experimental flags to `make play` via the `sim_check.py`
command directly:

```bash
.venv_sim/bin/python scripts/sim_check.py plan \
    --predictor wm --scenes 0 --gif 0 \
    --risk-smooth \
    --out outputs/play/wm_scene0_rsmooth.json
```

See `docs/EXPERIMENTS.md` § v2.1 for what each flag does and why none was adopted.
