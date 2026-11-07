# v3 world model on Waymo Open Motion (WOMD): feasibility

Date: 2026-11-07. Branch `womd-feasibility`. Read-only research: nothing was downloaded, installed or run on data,
and no test data was used. Anything not confirmed by a source I read is marked **(unverified)**.

**Short answer: go, with one conditional gate first (step 0, below).** The conversion path exists and is
already in our ScenarioNet clone. The model can stay as it is if we take the "current" step at 2 s instead of
Waymo's 1.1 s. The real work is in five silent mismatches in `sim/adapter.py`. Each would leave the pipeline
running with wrong inputs instead of crashing.

---

## 1. ScenarioNet + WOMD with our setup

| Item | Finding | Source |
|---|---|---|
| Converter | `python -m scenarionet.convert_waymo -d <out> --raw_data_path <dir with tfrecords> --num_workers N`. Present in our clone (`sim/scenarionet`, commit `d4acdb5`, 2026-03-09). | `sim/scenarionet/scenarionet/convert_waymo.py`, [ScenarioNet Waymo docs](https://scenarionet.readthedocs.io/en/main/waymo.html) |
| What it needs | **Only `tensorflow`** (it uses `tf.data.TFRecordDataset` to read the files) plus `protobuf`. The Waymo protos ship with ScenarioNet (`converter/waymo/waymo_protos/*_pb2.py`), so the `waymo-open-dataset` package is **not** needed. CPU only: the converter sets `CUDA_VISIBLE_DEVICES=""`. | `converter/waymo/utils.py` lines 14–22, 464 |
| Documented pins | `tensorflow==2.11.0`, `protobuf==3.20` (docs: "you may fail to install protobuf with pip, use conda"). | [waymo.rst](https://github.com/metadriverse/scenarionet/blob/main/documentation/waymo.rst) |
| **Known issue: the pins conflict** | PyPI: `tensorflow 2.11.0` requires `protobuf<3.20`. The bundled `_pb2.py` files import `google.protobuf.internal.builder`, which first appeared in protobuf 3.20. This explains the docs' install warning. | [PyPI tensorflow 2.11.0](https://pypi.org/project/tensorflow/2.11.0/) (requires_dist), `waymo_protos/scenario_pb2.py` line 8 |
| Proposed working combo | Python 3.10 (= `.venv_sim`) + `tensorflow-cpu==2.15.1` (cp310 wheel, 236 MB; needs `protobuf>=3.20.3,<5`, `numpy<2`) + `protobuf==4.25.*`. `.venv_sim` has numpy 1.26.4, which fits. **(unverified: not installed or tested)** | [PyPI tensorflow-cpu 2.15.1](https://pypi.org/project/tensorflow-cpu/2.15.1/) |
| Where to install | A separate env `.venv_womd` (Python 3.10: TF + protobuf + scenarionet/metadrive in editable mode). `.venv_sim` stays untouched. The converted `.pkl` files need no TF, so the simulator env never needs it. | proposal |
| No-TF fallback | A TFRecord file is just records of the form `[uint64 length][crc][bytes][crc]`, so about 15 lines of Python could replace `tf.data`. That would drop TF entirely. **(unverified, not tried)** | proposal |
| `waymo-open-dataset` pkg (not needed) | Latest is `waymo-open-dataset-tf-2-12-0==1.6.7` (2025-04-29). It pins `tensorflow==2.13` and `numpy==1.23.5`. | [PyPI](https://pypi.org/project/waymo-open-dataset-tf-2-12-0/) |
| MetaDrive | Waymo is MetaDrive's native scenario source (`MetaDriveType.COORDINATE_WAYMO`). Our MetaDrive 0.4.3 and ScenarioNet 0.0.1 handle it. | `sim/metadrive/metadrive/type.py` line 80 |
| Lane polylines after conversion? | **Yes.** Each lane is `{type, polyline, entry_lanes, exit_lanes, left/right_neighbor, left/right_boundaries, interpolating, speed_limit}`. Road lines, road edges, crosswalks, stop signs, speed bumps and driveways are also kept. **But:** `polyline` is `(x, y, z)`, not `(x, y, yaw)` as in our nuScenes conversion, and lanes have **no polygons**. | `converter/waymo/utils.py` `extract_poly`, `extract_center` |
| Known issues list | The ScenarioNet docs say "Known Issues: Waymo: N/A". I did not find a GitHub issue about conversion crashes (search of [metadriverse/scenarionet issues](https://github.com/metadriverse/scenarionet/issues)). | docs |

## 2. WOMD facts

The waymo.com dataset pages redirect to a Google login, so I could not read them. The facts below come from the
paper, the protos and third-party pages.

| Fact | Value | Source |
|---|---|---|
| Segments | 103,354 segments of 20 s at 10 Hz, mined for interactions (unprotected turns, merges, lane changes, intersections) | [paper](https://arxiv.org/abs/2104.10133), [GitHub README](https://github.com/waymo-research/waymo-open-dataset) |
| Scenario = 9 s window | 91 steps at 10 Hz: 11 history steps (1.1 s, incl. current; `current_time_index` = 10) + 80 future steps (8 s) | [WOSAC paper](https://arxiv.org/pdf/2305.12032), `scenario.proto` |
| Splits | 70 / 15 / 15 train / val / test by segment. Train **486,995**, val **44,097**, test **44,920** scenarios (v1.2.0) | [WOSAC paper](https://arxiv.org/pdf/2305.12032) |
| Interactive splits | `validation_interactive` and `testing_interactive`: same idea as val/test, but focused on the 2 mined interacting objects. **There is no `training_interactive`.** In training, `objects_of_interest` marks the "interactive group". Sizes ~43.5k / ~44.2k **(unverified, from memory)**. Whether `validation_interactive` shares scenario IDs with `validation`: **(unverified)**. | paper, `scenario.proto` comment on `objects_of_interest` |
| `training_20s` | The full 20 s segments of the training split (1000 shards per the ScenarioNet docs). Same segments as `training`, so it overlaps with training. | [waymo.rst](https://github.com/metadriverse/scenarionet/blob/main/documentation/waymo.rst) |
| `testing` split | Has history only (futures hidden for the leaderboard), so it is useless for us | paper |
| Shards and size | training 1000 shards of ~380–490 MB (≈ 487 scenarios per shard); validation 150 shards of ~235–280 MB (≈ 294 per shard). Totals ≈ 440 GB / 40 GB. Sizes come from a third-party mirror listing **(unverified for the current version)**. | [IEEE DataPort listing](https://ieee-dataport.org/documents/pnpt-dataset), counts ÷ shards |
| Agent types | VEHICLE, PEDESTRIAN, CYCLIST, OTHER (+ UNSET = error). **No car / truck / bus split.** | `scenario.proto` `Track.ObjectType` |
| Agents per scenario | Up to 128 in the tf.Example format **(unverified, from memory)**. All tracks span all 91 steps, with a `valid` flag per step. | `scenario.proto` |
| Map | Lane centres (types FREEWAY / SURFACE_STREET / BIKE_LANE, with an `interpolating` flag), entry / exit lanes, neighbours, boundaries, road lines, road edges, crosswalks, speed bumps, stop signs, driveways (v1.2+), and traffic-light states per step. Polyline spacing ≈ 0.5 m **(unverified, from memory)**. | `map.proto` |
| Version | Bucket in the ScenarioNet docs: `waymo_open_dataset_motion_v_1_2_0`. The Waymo About page says the Motion data was last updated in Oct 2025. The current version number is **(unverified)**. | [Waymo About](https://waymo.com/open/about/) |
| Licence | Waymo Dataset License Agreement for **Non-Commercial Use**. You accept it on waymo.com, then use `gsutil` with your Google account. | [GitHub README](https://github.com/waymo-research/waymo-open-dataset), [terms](https://waymo.com/open/terms/) |

## 3. Gap vs our v2 pipeline

Note on the brief: our v2 pipeline uses **6 s** of future (12 steps at 2 Hz), not 8 s
(`utils/dataset_v2.py`: T = 5 past, F = 12 future, DT = 0.5).

**Key idea:** take the model's "current" step at WOMD step **20** (t = 2 s) instead of Waymo's step 10. Then
history = 2 s (5 keyframes: steps 0, 5, 10, 15, 20) and future = 6 s (steps 25…80 ≤ 90). This is exactly what
v2 expects, so the **model needs no change**. The cost is that our numbers cannot be compared with the official
WOMD leaderboard, which we don't need.

| # | Our pipeline expects | WOMD gives | Change needed |
|---|---|---|---|
| 1 | 2 Hz keyframes | 10 Hz | Take every 5th step (the adapter already does this: `STRIDE = 5`). |
| 2 | 5 history steps (2 s) | 11 steps (1.1 s) at the official current step | Current = step 20 (see above). Optionally also step 25 and 30, which gives up to 3 samples per scenario. |
| 3 | 12 future steps (6 s) | 80 steps (8 s) | Use 12 of the 16 available. Going to 8 s would be a separate change (decoder output, anchors, ego plan, planner horizon); not proposed for v3. |
| 4 | 6 agent classes (car, truck/bus, ped, cyclist, ego, other) from nuScenes category strings | 4 types, no truck class | Map VEHICLE→car (or truck/bus if length > 6 m; a heuristic, write the rule down before training), PEDESTRIAN→ped, CYCLIST→cyclist, OTHER→other, SDC→ego. |
| 5 | **Adapter keeps only `vehicle.*` / `human.pedestrian*` categories** (`sim/adapter.py:203`) | `VEHICLE`, `PEDESTRIAN`, `CYCLIST` | **Silent failure:** with no change, every agent is dropped. Needs a type map per dataset. |
| 6 | **`LANE_TYPES` = {SURFACE_STREET, SURFACE_UNSTRUCTURE}** (`sim/adapter.py:40`) | LANE_SURFACE_STREET, **LANE_FREEWAY**, LANE_BIKE_LANE | **Silent failure:** freeway lanes are dropped. Add FREEWAY (and decide whether bike lanes count). |
| 7 | "Is connector" flag (nuScenes lane connectors) | No connector concept; `interpolating` flag (lanes that join two other lanes, roughly the same idea) | is_connector := `interpolating` **(unverified how close the two meanings are)**. We retrain from scratch, so the flag only has to be consistent between training and the simulator. |
| 8 | **Polyline = (x, y, yaw) at 1 m** (`lane_tokens` reads `[:, :3]` as x, y, yaw) | (x, y, **z**) at ~0.5 m | **Silent failure:** z would be read as yaw. Compute yaw from point differences and resample to 1 m before `split_lane` (21 points per 20 m token). |
| 9 | Connector polygons from the nuScenes map (`add_connector_polygons`, used in `closed_loop.py:344`) | No polygons at all | Skip it for WOMD. `LaneSelector` already falls back to polylines when a lane has no polygon. |
| 10 | 192 lane slots, 50 m box | Waymo maps are dense (more lanes, plus bike lanes) | Measure the token count per sample on 1 shard. If p95 > 192, raise `max_lanes` or shrink the box. **(unverified)** |
| 11 | 64 agent slots, 16 predicted, 35 m radius | Often more agents than nuScenes **(unverified)** | Keep 64 / 16 for v3 (same as the current WM). Report coverage (share of nearby agents that get a WM slot) as before. |
| 12 | Ego = Renault Zoe, fixed 1.73 × 4.08 m (`EGO_SIZE`, `EGO_SIZE_ZOE`) | SDC (Waymo vehicle) with its own box per step | Use the SDC track's length and width. |
| 13 | Data builder reads the nuScenes devkit | Protos / ScenarioNet pkl | New builder. **Proposal:** build training samples from the ScenarioNet `.pkl` with the same `sim/adapter.build_sample` used in closed loop, so training and simulator inputs are identical by construction (like `sim_check.py match`). |
| 14 | Velocity from 0.5 s position differences | WOMD also gives velocity_x / velocity_y | Keep our differences (consistent). The "noisy accelerations" pitfall is about interpolated nuScenes logs; WOMD is natively 10 Hz. |
| 15 | Closed-loop scenes of 20 s | Val scenarios are **9 s**; after 2 s of history, **7 s remain** | Fewer events per scene, so more scenes are needed (see 5). The planner's 6 s horizon runs past the log end in the last seconds; score the first ~5 s, or replay the remaining agents as constant velocity **(decision)**. |
| 16 | Reactive-agent rule uses ScenarioNet type VEHICLE | Same string for Waymo | Works as is. |
| 17 | Traffic lights not used | Given per step | Ignore in v3 (same as v2). |

## 4. Minimal plan

| Item | Proposal |
|---|---|
| Training data | **50 `training` shards ≈ 24k scenarios**. With 1–3 targets per scenario this gives ≈ 50–75k samples, 2–3× nuScenes (25.7k). Oversample scenarios where `objects_of_interest` contains the SDC or a vehicle right behind it (the follower case). |
| Model-selection data | 10 `validation` shards (see 5). |
| Raw size | 50 × ~0.45 GB ≈ **22 GB** + val 20 × ~0.26 GB ≈ **5 GB**. Above the 5 GB rule: **you download** (licence + gsutil). |
| Converted size | ScenarioNet `.pkl`: about the same as raw **(unverified)**. Processed samples: ~41 KB per sample (nuScenes: 1.07 GB / 25.7k), so ≈ 2–3 GB. Raw tfrecords can be deleted after conversion (ask first). Laptop has 315 GB free. |
| Where | Download and convert **on the laptop** (CPU, ~1–2 h with 6 workers **(unverified)**). Only the processed train/val `.pkl` (~3 GB) goes to the pod. Whether the WOMD licence allows data on a rented pod: **your call**. |
| GPU time | Reference: e32/e63 on nuScenes ≈ 80 min on an RTX 4090 for ~45 epochs (2 runs sharing one GPU). At 3× the data and 40 epochs, expect **≈ 3–5 h on a 4090 per seed**. On the laptop 4060, likely 2–3× slower **(unverified)**. Plan: 1 seed first, 3 seeds only if the follower test passes. |
| Gate before all of this (step 0) | Check the hypothesis cheaply first, using **1 val shard (~0.27 GB)** and the nuScenes train set. Count "strong follower" events (a vehicle behind the ego that decelerates > 2 m/s² within 2 s of the ego braking) per 1000 samples. If WOMD has fewer than 2× nuScenes's rate, the data hypothesis is weak: stop and rethink (e.g. the plan-dropout setting, or oversampling inside nuScenes). |

## 5. Fresh test design (fixed now, before any training)

**Source:** WOMD `validation` only. It is segment-disjoint from `training` by Waymo's 70/15/15 split
([paper](https://arxiv.org/abs/2104.10133)). We never use `training_20s` (it is the training segments) or
`testing` (no futures).

| Set | Shards | Use |
|---|---|---|
| **womd_val** | `validation.tfrecord-00000…00009-of-00150` (~2.9k scenarios) | Model selection, all tuning, open-loop follower test |
| **womd_dev** | 100 scenes from `validation-00010…00019`, selected by the rule below with seed 1 | Planner development and cost tuning (replaces mini + dev100 for WOMD) |
| **womd_test300** | 300 scenes from `validation-00140…00149`, selected by the rule below with seed 0 | Final closed-loop numbers only |

**Selection rule (applied mechanically, written here before any download):** the SDC is valid at all 91 steps
and travels ≥ 10 m between steps 20 and 90. Then stratify: 150 scenes where `objects_of_interest` contains the
SDC (interactive) + 150 drawn at random from the rest. Use `numpy.random.default_rng(seed)` on the scenario IDs
sorted alphabetically. 300 instead of 146 because each scene is only 7 s long.

**Keeping it separate:**
1. Write the test scenario IDs (IDs only, no data) to `sim/scene_lists/womd_test300.json`. Commit it, and record its
   sha256 in `docs/EXPERIMENTS.md` before training starts.
2. The data builder never reads validation shards ≥ 140. The builder and `closed_loop.py` refuse any ID in the
   test list unless `--eval-test` is given (same rule as today).
3. If `validation_interactive` is used for extra val data, first drop every scenario ID that is in the test list
   (the overlap between the two is **unverified**).
4. Converted test scenes go to their own folder, `sim/scenarios/womd_test300`, which is converted only at final
   evaluation time.

## 6. Effort estimate (agent working time, not wall-clock GPU time)

| Step | Work | Estimate |
|---|---|---|
| 0. Env + smoke + gate | `.venv_womd`, convert 1 val shard, replay 1 scene in MetaDrive, follower-event count (needs 1 shard from you) | 0.5 day |
| 1. Loader / builder | Type map, lane types, yaw from polyline + 1 m resampling, no polygons, SDC size, current = step 20, build via `adapter.build_sample`; checks: lane-token plot, token-count stats, match test (builder vs live sim on 3 scenes) | 1.5–2 days |
| 2. Training | Anchors from WOMD, 5-epoch check on the laptop, then 1 seed on the pod (~3–5 h, ~$2–4 **(unverified price)**); EXPERIMENTS.md rows | 0.5 day + GPU |
| 3. Sim adaptation | `closed_loop.py` dataset switch (no connector polygons, start at step 20, 7 s episodes, end-of-log handling), reactive agents on Waymo types, scene lists, re-run the reactive validations (a)/(b)/(d) on womd_dev | 1.5–2 days |
| 4. Evaluation | Open-loop follower test + val metrics; planner tuning on womd_dev; pre-registration amendment; then the final womd_test300 run (CV vs WM, replay + reactive) and report | 1–1.5 days + CPU |
| **Total** | | **≈ 5–6.5 days** |

---

## Recommendation

**Go, conditional on step 0.** Conversion is low-risk: the converter is in our clone, MetaDrive's native format
is Waymo, and the TF/protobuf conflict has a clear fix. The model is unchanged if the current step is taken at
t = 2 s. Run step 0 first (1 val shard, ~0.27 GB). If WOMD does not show clearly more strong follower
interactions than nuScenes, stop before downloading the 22 GB.

**Biggest risk:** the weak follower reaction may not be a data problem. It may come from the model or the training
setup instead: the ego-plan token is dropped in 20–50% of training samples, and WTA plus soft scores reward the
average reaction. In that case WOMD costs a week and the follower test stays weak. The step 0 gate partly tests
this. A one-line nuScenes ablation (plan dropout 0) would test it more directly, but that is outside this brief.
Second risk: the 9 s scenes give only 7 s of closed loop, so at-fault events will be even rarer than on test150.

## Decisions I need from you

1. **Download:** accept the WOMD licence and fetch **1 validation shard** (`validation.tfrecord-00000-of-00150`,
   ~0.27 GB) for step 0. The 50 training + 20 val shards (~27 GB) come later, after the gate.
2. **Environment:** a new `.venv_womd` (TF 2.15.1-cpu + protobuf 4.25) vs the no-TF TFRecord reader. I recommend
   `.venv_womd`: it is the tested path for ScenarioNet, and the pure-Python reader is a fallback.
3. **Pod + licence:** may the processed WOMD samples go to the RunPod pod, or do we train only on the laptop
   (~2–3× slower)?
4. **Test design:** approve section 5 as written (validation shards 140–149, 300 scenes, the stratified rule), or
   change it before any download.
5. **End of log in closed loop:** score only the first 5 s of each 7 s episode, or continue the missing agents as
   constant velocity.
6. **Truck class:** map every VEHICLE to "car", or split by length > 6 m.
