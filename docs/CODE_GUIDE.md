# Code guide

A map of the repository for review: what lives where, what each v2 file does, the flags, and how to run things.
v1 is the GATv2-GRU model from the course report (tag `v1-gnn`); v2 is the scene transformer, the official model is
`v2_d_rerun`. The v3 ablations (v3a/b/c) are recorded in docs/EXPERIMENTS.md; only v3a's flag (`--plan-drop 0`) is
in this code, v3b / v3c are on tags `v3b-xattn` / `v3c-stand-in`.

## 1. Repository structure

```
wm/
├── utils/                      library code (imported, never run directly)
│   ├── lane_tokens.py          v2: map lanes -> 20 m lane-segment tokens
│   ├── dataset_v2.py           v2: processed sample -> fixed-size agent / lane tensors
│   ├── models_v2.py            v2: encoders, scene transformer, decoder, losses, V2Predictor
│   └── training_utils.py       shared: seeds, manifests, multimodal_metrics
├── scripts/                    command-line entry points
│   ├── build_processed_dataset.py   nuScenes -> data/processed(_v2)/*.pkl
│   ├── make_anchors.py              endpoint anchors -> assets/anchors*.npy
│   ├── train_model.py               train / evaluate v2
│   ├── analyze_v2.py                open-loop analysis: turn bins, CV baseline, follower / placebo test, examples
│   └── sim_check.py                 simulator entry point: convert, replay, plan, evaluate, reactive checks
├── sim/                        planner.py (plans, tracker), closed_loop.py (episode loop, costs, metrics,
│   │                           reactive road users), adapter.py (sim -> model inputs)
│   └── scene_lists/            dev100.json; reserved, never run: test_v3.json
├── assets/                     anchors.npy (k-means), anchors_angle.npy (turn-spread), README figures and GIF
├── docs/                       CODE_GUIDE.md (this file), EXPERIMENTS.md (every run + pre-registrations),
│                               reactive_agents_design.md, womd_feasibility.md, architecture.png
├── results/                    summary JSONs behind the README tables, make_figures.py (no dataset needed)
├── data/                       local only (git-ignored except LICENSE)
│   ├── v1.0-trainval/          nuScenes metadata tables
│   ├── maps/                   map expansion v1.3 (+ prediction splits)
│   ├── nuscenes_mini/          v1.0-mini tables only (maps -> ../maps); used by the simulator test
│   └── processed_v2/           built v2 datasets: train / val / test .pkl
├── checkpoints/                local only: one folder per experiment, *_best.pth + *.json manifest
├── outputs/                    local only: training logs, plots, turn-bin JSONs, sim/ images
├── sim/metadrive, sim/scenarionet, sim/scenarios/   local only: simulator clones and converted scenes
├── .venv/                      main Python env (training)
└── .venv_sim/                  Python 3.10 env for the simulator (uv)
```

Splits: nuScenes prediction challenge `train` = train (25,711 samples), `train_val` = **val** (6,685),
`val` = **test** (7,242). Settings are chosen on val; test is looked at once, at the end.

## 2. The v2 pipeline in one paragraph

Each sample is one target vehicle at its current time, everything in the **target frame** (target at
the origin, heading along +x). `build_processed_dataset.py --lane-tokens` stores 2 s of history and
6 s of future for every agent, the ego, and the lanes within 50 m cut into ≤ 20 m segments.
`TrajectoryDatasetV2` turns that into fixed-size tensors (64 agent slots, 192 lane slots).
`V2Predictor` encodes every agent and lane segment into one token, runs a transformer over all tokens
with a relative-pose attention bias, and decodes 6 trajectories (modes) plus a probability per mode,
for the target (or, with `--predict-agents 16`, for the target and 15 neighbours).
Training uses winner-takes-all: only the best mode gets the regression loss.

## 3. v2 files, in plain words

### `utils/lane_tokens.py`
- `poses_to_local(poses, origin, yaw)`: world poses -> target-frame `[x, y, cos ψ, sin ψ]`.
- `split_lane(points, seg_len=20)`: cuts a lane polyline (1 m spacing) into segments of ≤ 20 m (21 points);
  neighbouring segments share their end point so they touch.
- `build_lane_tokens(nusc_map, origin, yaw, radius=50, seg_len=20)`: all lanes and lane connectors within
  50 m -> segment points, an "is connector" flag per segment, and successor edges between segments.

### `scripts/build_processed_dataset.py`
Builds the datasets from nuScenes. v1 output is unchanged by default; `--lane-tokens` adds the v2
fields and saves to `data/processed_v2/`. Main pieces: `get_map_for_sample` (cached map per location),
`get_agent_states_for_sample` (per agent: past (5 steps) and future (12 steps) of `[x, y, cos, sin]`
with validity masks, size, category; plus the ego's past and future), `validate_dataset`
(shape / finiteness checks; counts target history gaps instead of failing).
`--workers 6` runs it in parallel (~17 min); each split is saved as soon as it is done.

### `utils/dataset_v2.py`
- `TrajectoryDatasetV2(path, max_agents=64, max_lanes=192)`: one item = a dict of tensors.
  Agent slot order: **0 = target, 1 = ego, then neighbours by distance**, so truncation never drops
  the target or the ego. Keeps the 192 lane segments closest to the target and remaps their edges into
  `lane_adj`. Also returns the ego's planned future (`ego_future`) for the ego-plan token.
- `add_velocity`: appends `[vx, vy]` from consecutive positions (0.5 s apart).
- `agent_type`: nuScenes category -> one of 6 classes (car, truck/bus, pedestrian, cyclist, ego, other).
- `EGO_SIZE`: fixed ego size (Renault Zoe, 1.73 × 4.08 m).

### `utils/models_v2.py`
Encoders (one token each, 128-d):
- `AgentEncoder`: an agent's flattened 2 s history + validity flags, plus type and size embeddings.
- `LaneEncoder`: VectorNet-style: per-point MLP -> max-pool -> concat -> MLP -> max-pool, plus a connector flag.
- `EgoPlanEncoder` (block 2): the ego's planned 6 s future, flattened with validity flags, plus a learned
  "plan" embedding.

Scene encoder:
- `token_poses`: (x, y, heading) per token: agents at their last valid history step, lanes at their middle point.
- `rel_features`: for every token pair, where j is seen from i: `[dx, dy]` in i's frame, cos/sin of the
  heading difference, distance (positions / 50 m); with `--topology` also "j is successor / predecessor of i".
- `SceneEncoder` (+ `EncoderLayer`): 3 pre-LN transformer layers, 8 heads. The relative features go through a
  small MLP into **one scalar bias per head** added to the attention scores (computed once, shared by all layers).
  Padded tokens are masked out.

Decoder:
- `AnchorDecoder` (+ `DecoderLayer`): one query per (agent, mode): `MLP(anchor_k) + mode embedding + agent token`.
  2 layers of: self-attention among the 6 modes of the same agent -> cross-attention to all scene tokens -> FFN.
  Heads output a 12-step path (in 50 m units) and a score per mode.
- `to_target_frame`: paths are predicted in each agent's own frame (origin = its current pose, x = its heading)
  and rotated/translated into the target frame.

Losses:
- `v2_loss` (target only): SmoothL1 on the winning mode + cross-entropy on the scores. The winner `k*` is the
  predicted mode with the lowest ADE (`assign="predicted"`) or the anchor closest to the true endpoint (`"anchor"`).
- `multi_agent_loss`: the same per agent, for agents with ≥ 6 valid future steps, averaged over agents.
- `soft_score_ce` (block 3): score target `p_k = softmax(−ADE_k / 1 m)` instead of one-hot on the winner.

Full model:
- `V2Predictor(anchors, ..., predict_agents=1, ego_plan=False, soft_scores=False, topology=False)`:
  encoders -> scene encoder -> decoder. `forward(batch, plan=None)` returns
  `traj (B, 6, 12, 2), logits (B, 6)` for the target, or `(B, 16, 6, 12, 2), (B, 16, 6)` with
  `predict_agents=16` (target at index 0; ego slot skipped). `.loss(batch, traj, logits)` picks the right loss.
  The ego-plan token is dropped per sample with p = 0.5 in training; `plan=True/False` forces it on/off.
  With every option off, the model is the original target-only v2 (bit-identical outputs).

### `scripts/train_model.py`
- `evaluate_v2(model, loader, device, plan)`: returns a dict: target minADE₆ / minFDE₆ (same code as v1),
  miss rate (minFDE₆ > 2 m, endpoint definition), top-1 ADE / FDE (highest-score mode), mode-win shares;
  for multi-agent models also all-agent and **moving-agent** (> 1 m from the current position in history or
  future) minADE₆ / minFDE₆ / top-1. Use the moving-agent numbers: parked cars make all-agent errors look easy.
- `agent_min_errors`, `moving_agents`: helpers for the multi-agent metrics (ADE over valid steps, FDE at the
  last valid step).
- `run_v2`: AdamW (lr 1e-3, wd 0.01), 1 epoch linear warmup then cosine to 1e-5 over all epochs, grad clip 1.0,
  dropout 0.1, best-val checkpoint, early stopping. Logs per epoch: losses, lr, val metrics (plan on and off
  with `--ego-plan`), mode wins (flags a mode below 3%). After training: val on the best checkpoint (test only with
  `--eval-test`), written to the manifest JSON next to the checkpoint.

### `scripts/analyze_v2.py`
Errors binned by the absolute endpoint angle (0–15°, 15–30°, 30–60°, 60°+, only endpoints ≥ 5 m away; same
definition as the v1 report). Reads anchors and model options from the checkpoint and its manifest.
`--examples out.png` instead plots 8 val examples (2 per bin) with lanes, agents, ground truth and the 6 modes.
Note: the 60°+ bin also contains reversing agents (endpoint behind the target).
Renamed from `run_turn_angle_analysis.py` (2026-11-07). Other modes:
- `--follower-test DIR [--follower-group G]`: does the ego plan reach other agents? 20 val scenes, one agent per scene
  of group G: `follower` (2–15 m behind the ego in its lane, should react) or a placebo group that should not react:
  `lead` (2–30 m ahead), `oncoming`, `side` (next lane). Ego plans: real, brake to stop (−4 m/s²), accelerate
  (+2 m/s²), plan off. Prints and saves predicted 6 s travel per plan, the **brake reaction ratio** (drop in travel /
  drop needed to stay a car length behind the stopped ego) and the mean |travel change| (placebo measure).
- `--cv-baseline JSON`: constant-velocity baseline vs the model on val.

### `scripts/make_anchors.py`
`--method kmeans` (default): k-means (K = 6) on train endpoints -> `assets/anchors.npy`.
`--method angle`: 2 straight anchors + one each for right, sharp right, left, sharp left -> `assets/anchors_angle.npy`.

### `check_lane_tokens.py` (v1-era, on branch `workbench` under `legacy/`)
Builds the lane tokens of one sample directly from the map and plots segments, direction arrows and edges.

### `sim/adapter.py` (Phase D: simulator -> model inputs)
`HistoryBuffer` keeps the last 2 s of sim snapshots (10 Hz); `snapshot_from_scenario` (log) and `snapshot_from_env`
(log for the others + the live MetaDrive ego) fill it. `build_sample(buffer, map_features, target_id, ego_plan)` rebuilds
exactly what the dataset builder stores for that target (agents within 35 m, 5 keyframes of history, lanes in the 50 m box
as 20 m segments + edges, the ego plan as `ego_future`); `to_model_inputs` turns it into the tensors of
`TrajectoryDatasetV2`. `add_connector_polygons` restores lane-connector polygons (ScenarioNet drops them) from the
nuScenes map, which the lane selection needs. `sim_check.py match` checks all of this against the processed dataset.

### `scripts/sim_check.py` (simulator entry point)
Smoke tests: `metadrive`, `drive`, `replay`, `match` (adapter vs processed dataset), `follow` (tracker check).
`convert [--set mini|dev100|test150]`: nuScenes scenes -> ScenarioNet scenarios in `sim/scenarios/`.
`plan --predictor cv wm`: closed-loop planners + the logged ego ("expert") on a scenario folder; `--reactive idm` for
reactive road users, `--t0 S` to start at S seconds (the ego follows its log until then; metrics from t0),
`--no-shadow` to skip the shadow WM in CV runs (no GPU needed).
`evaluate`: the full CV-vs-WM evaluation with paired statistics (bootstrap CIs, Wilcoxon) and a report.
`reactive-check`: validation of the reactive road users (mini + dev100). `showcase`: the README GIF.
`hesitation results.json ...`: stop-and-go / chatter metrics from driven traces (no simulation).

### `scripts/demo_2x2.py` (README examples)
CV vs WM on development scenes, MetaDrive 3D + top-down: `grid` (assets/examples_grid.png), `videos`, `clip`
(example 1, ~8 s with slow motion and a freeze-frame -> assets/example1_clip.gif; MP4 in outputs/, not in git).
Uses the off-by-default `step_hook` of `run_scene`. Selection rule: docs/demo_selection.md.

## 4. v2 flags (`train_model.py --model v2`)

| Flag | Default | What it does |
|---|---|---|
| `--assign {predicted,anchor}` | predicted | Which mode is the winner in the loss: lowest-ADE predicted mode (winner-takes-all) or the anchor closest to the true endpoint. Anchor was clearly worse (val minADE₆ 1.84 vs 1.22). |
| `--predict-agents N` | 1 | 1 = target only. 16 = target + 15 nearest neighbours (ego skipped); adds all-agent and moving-agent metrics. |
| `--ego-plan` | off | Adds the ego's planned future as a scene token; val logged with plan on and off; checkpoint and early stopping use plan on. |
| `--plan-drop P` | **0.5** | Probability of dropping the plan per training sample. The official v2 uses **0.2** (pass it explicitly); v3a used 0. |
| `--plan-queries` | off | Gives each predicted agent the plan in its own frame (official v2: on). |
| `--target-split` | off | Target-weighted loss for multi-agent models (official v2: on). |
| `--patience-start N` | 0 | No early stopping before epoch N (official v2: 30). |
| `--eval-test` | off | Also score the test split. Final evaluation / reproducibility checks only. |
| `--eval-slots N` | | With `--evaluate-only`: agent metrics over the first N slots (compare different `--predict-agents`). |
| `--soft-scores` | off | Score loss against `softmax(−ADE_k / 1 m)` instead of the one-hot winner. |
| `--topology` | off | Adds successor / predecessor lane flags to the attention bias. Ablation only: no gain on val (seed 42). |
| `--scale-inputs` | off | Divides input positions by 50 m. Not used in any reported run. |
| `--anchors PATH` | assets/anchors.npy | Anchor file used to initialise the mode queries. |
| `--patience N` | 10 | Early stopping on val target minADE₆ (0 = off); the cosine schedule still spans `--epochs`. |
| `--epochs`, `--batch-size`, `--lr`, `--seed` | 30, 64, 1e-3, 42 | Reported runs use 60 epochs, batch 64, lr 1e-3, seeds 42 / 123 / 2026. |
| `--checkpoint-dir`, `--evaluate-only`, `--checkpoint` | | Where to save; evaluate an existing checkpoint without training. |

## 5. How to run

Training env: `.venv` (main). Simulator env: `.venv_sim`. Run everything from the repo root.

```bash
# Build the v2 data (once; ~17 min)
.venv/bin/python scripts/build_processed_dataset.py --lane-tokens --workers 6

# Official v2 (d_rerun)
FLAGS="--model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2"
.venv/bin/python scripts/train_model.py $FLAGS --epochs 60 --patience 15 --patience-start 30 --seed 42 \
    --checkpoint-dir checkpoints/v2_d_rerun

# Evaluate a checkpoint on val (model flags needed; add --eval-test only for the final evaluation)
.venv/bin/python scripts/train_model.py $FLAGS --evaluate-only --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth

# Follower test and a placebo group (val)
.venv/bin/python scripts/analyze_v2.py --model v2 --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth \
    --follower-test outputs/run_d_rerun --follower-group follower

# Turn-angle bins on val, and an example plot
.venv/bin/python scripts/analyze_v2.py --model v2 --checkpoint checkpoints/v2_predicted/v2_seed42_best.pth \
    --dataset data/processed_v2/nuscenes_val_prediction_samples_map.pkl --output outputs/turn_bins.json
.venv/bin/python scripts/analyze_v2.py --model v2 --checkpoint checkpoints/v2_predicted/v2_seed42_best.pth \
    --dataset data/processed_v2/nuscenes_val_prediction_samples_map.pkl --examples outputs/v2_examples_val.png

# Simulator smoke test (CPU)
.venv_sim/bin/python scripts/sim_check.py metadrive
.venv_sim/bin/python scripts/sim_check.py convert
.venv_sim/bin/python scripts/sim_check.py replay --index 0
```

Both scripts use **val** by default; test is scored only with `--eval-test` (final evaluation or reproducibility
checks). The turn-angle script refuses a test `--dataset` without `--eval-test`.

Simulator env setup (already done): `uv venv --python 3.10 .venv_sim`, then with `VIRTUAL_ENV=.venv_sim`:
`uv pip install -e sim/metadrive -e sim/scenarionet "nuscenes-devkit>=1.1.10"` and
`.venv_sim/bin/python -m metadrive.pull_asset`. Versions: MetaDrive 0.4.3, ScenarioNet 0.0.1
(needs MetaDrive ≥ 0.4.1.2), nuscenes-devkit 1.2.0, Python 3.10.

## 6. Where results and logs are

| What | Where |
|---|---|
| Every run, settings and results; pre-registrations | `docs/EXPERIMENTS.md` |
| Closed-loop reports | `outputs/sim/<run>/report.md` (local); final: `results/test150_final_report.md` |
| v3 checks (follower / placebo / val) | `outputs/v3_checks/`, references `outputs/v3a_prereg/` |
| Training logs, one per run | `outputs/train_v2_<run>_seed<N>.log` |
| Best checkpoint + manifest (settings, best / stopped epoch, val and test metrics) | `checkpoints/<run>/v2_seed<N>_best.pth` and `.json` |
| Turn-angle bins (val) | `outputs/turn_bins_v2_val_seed*.json`, `outputs/turn_bins_v2topo_val_seed42.json` |
| Example predictions | `outputs/v2_examples_val.png` |
| Anchor plots | `outputs/anchors.png`, `outputs/anchors_angle.png` |
| Simulator replay | `outputs/sim/replay_scenario0.png`, `.gif` |
| Dataset build log | `outputs/build_v2.log` |

Checkpoint folders follow the run names in docs/EXPERIMENTS.md (e.g. `v2_d_rerun`, `v2_d_seed123`, `v3a`,
`v3a_seed123`, `v2_d_rerun_current`). Older manifests (before `--eval-test`) also hold test metrics: don't use them
for any choice.

## 7. Things to know

- GPU training is not bit-reproducible: the same seed gives different results, and across machines (laptop torch 2.15
  vs pod torch 2.8) d_rerun differed by 0.025 m val minADE₆, more than the 3-seed spread (0.006). Compare over seeds.
- Miss rate here = minFDE₆ > 2 m at the endpoint (Argoverse style), not nuScenes' max-over-time definition.
- `predict_agents=1` with every flag off reproduces the original v2 outputs exactly, so old checkpoints still load.
