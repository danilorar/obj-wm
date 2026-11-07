# CLAUDE.md

## Project
Trajectory prediction on nuScenes, used as an object-level world model for closed-loop
sampling-MPC planning (MetaDrive + ScenarioNet). v1 = GNN (tag v1-gnn), v2 = scene transformer; the official model is
v2_d_rerun (`checkpoints/v2_d_rerun/v2_seed42_best.pth`). Final branch: v3a-no-plan-dropout (v2 + the v3 ablations and
results). v3a/b/c were not confirmed; v3b/v3c code is on tags v3b-xattn / v3c-stand-in, WOMD/AV2 on tag womd-nogo.
See docs/CODE_GUIDE.md for file-by-file details.

## Environments
- `.venv` (Python 3.12, torch 2.15 cu132): data building, training, open-loop eval.
- `.venv_sim` (Python 3.10, torch 2.14 cu132): MetaDrive/ScenarioNet, closed-loop planning.
- RunPod pod (torch 2.8): long training runs only. Upload code with `git archive`, verify checksums, never
  `--eval-test`; stop the pod as soon as checkpoints are copied back.

## Key commands
- Train v2: `python scripts/train_model.py --model v2 ...` (see CODE_GUIDE for flags)
- Open-loop analysis (turn bins, CV baseline, follower / placebo test): `python scripts/analyze_v2.py --model v2 ...`
- Simulator: `.venv_sim/bin/python scripts/sim_check.py {match|follow|plan|evaluate} ...`

## Data splits — never break these
- train = model training; val = model selection and all tuning; test = final numbers only.
- Test scenes and the test split may be run only for the final evaluation or for reproducibility checks of an
  evaluation already run; never for tuning, planner changes or model selection.
- Closed-loop test set = sim/scenarios/nuscenes_test150 (146 scenes). The 10 mini scenes are
  the planner development set; cost weights are tuned only there and then frozen. dev100 (val-split scenes) is for
  planner development and diagnosis.
- Reserved, never run: `sim/scene_lists/test_v3.json` (only for a confirmed v3 winner, which does not exist) and
  `sim/scene_lists/womd_test300.json` (WOMD, NO-GO).

## Rules
- All training runs are listed in docs/EXPERIMENTS.md; add a row for every new run.
- Never commit or upload nuScenes data, images or processed samples (data/ is git-ignored).
- Ask before: deleting files, downloads > 5 GB, anything that costs money (pods, APIs).
- Launch every training run with `systemd-inhibit` (the laptop must not sleep mid-run).
- Run GPU jobs on the laptop sequentially, one at a time.
- Commit after each completed step, with a message that says what changed and why.
- Keep long commands short and with a timeout; run long jobs in the background with a log.
- When a result changes a decision, report it before continuing.

## Known pitfalls
- In log replay, other agents do not react: report rear-end collisions separately from at-fault ones.
- ScenarioNet drops lane-connector polygons: the adapter restores them (add_connector_polygons).
- The model predicts 16 agent slots; extra slots fall back to constant velocity in the planner.
- Accelerations from interpolated logs are noisy: compute them from 2 Hz positions.
- `--plan-drop` defaults to 0.5 (older runs rely on it); the official v2 config passes `--plan-drop 0.2` explicitly.
- Training noise: the same seed and flags on the laptop (torch 2.15) vs the pod (torch 2.8) differ by ~0.025 m val
  minADE₆, more than the 3-seed spread (0.006). Compare variants over several seeds, on the same machine if possible.