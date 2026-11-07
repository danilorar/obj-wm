# v3 world model on Waymo Open Motion (WOMD): feasibility

> Public branch note: the test-scene list `sim/scene_lists/womd_test300.json` (IDs and file checksums only) is kept on
> branch `workbench`; no WOMD or Argoverse 2 data is in this repository.

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

---

## Decisions (2026-11-07, from Danilo)

1. Step 0 uses 1 validation file, **not** one of files 140–149 (reserved for test). Chosen: `validation.tfrecord-00000-of-00150` (part of womd_val).
2. New env `.venv_womd` (Python 3.10, tensorflow-cpu 2.15.1, protobuf 4.25.9, numpy 1.26.4, MetaDrive + ScenarioNet
   editable). Built 2026-11-07; the ScenarioNet Waymo protos import fine, so the protobuf conflict is solved (verified at import level).
3. Processed WOMD data stays on the laptop for now and is never committed (`data/` is git-ignored).
4. Test design (section 5) is approved. The 300 test scene IDs are committed (`sim/scene_lists/womd_test300.json`)
   **before** step 0 or any training.
5. Closed loop: score only the first 5 s of each episode; no constant-velocity extension.
6. All WOMD vehicles map to "car"; keep the real box sizes.

## Step 0: pre-registered definition (written before any count was run)

Same code for both datasets (`scripts/womd.py follower`), on a 2 Hz grid (nuScenes keyframes; WOMD steps 0, 5, …, 90).
- Speed v[k] = displacement from k−1 to k, projected on the heading at k, ÷ 0.5 s. Acceleration a[k] = (v[k] − v[k−2]) / 1 s.
  The 1 s window reduces the annotation noise of nuScenes (accelerations from 2 Hz positions).
- **Exposure:** keyframes t where the ego's a[t] is defined, the ego moved at ≥ 2 m/s at t − 1 s, and the log
  continues to t + 2 s. Reported as ego-moving hours.
- **Ego brake:** a_ego[t] ≤ −2 m/s². Consecutive brake keyframes form one **episode**.
- **Follower at t:** a vehicle (nuScenes `vehicle.*` except bicycles; WOMD TYPE_VEHICLE), moving at ≥ 2 m/s, with its
  centre 3–30 m behind the ego (ego frame), |lateral| ≤ 2 m, heading within 30° of the ego.
- **Strong follower event:** a brake episode with at least one follower that reaches a ≤ −2 m/s² within t … t + 2 s.
- **Primary number:** strong events per ego-moving hour, R. **nuScenes side:** the official `train` scenes (700 =
  our train + val); the official `val` scenes (our test) are never read.
- **Gate (from section 4):** go if R_WOMD ≥ 2 × R_nuScenes **and** WOMD has ≥ 10 strong events. With < 10 events the
  result is "inconclusive": ask for 2 more files (00001, 00002) and rerun on the 3 files together.
- Also reported (not used for the gate): brake episodes per hour, and the share of followed brake episodes where the
  follower also braked hard.
- Known bias: nuScenes 2 Hz annotations are noisier than WOMD's 10 Hz tracks. This can only inflate the nuScenes
  counts, so it makes the gate harder to pass (conservative).

### Step 0 results (2026-11-07)
Test IDs were committed first (`1900f41`). WOMD = v1.3.1 `validation.tfrecord-00000-of-00150` (286 scenarios).

| Dataset | Scenes | Ego-moving h | Ego brake episodes / h | With follower | Strong events | **R (strong / h)** | 95% CI (Poisson) |
|---|---|---|---|---|---|---|---|
| nuScenes official train | 700 | 2.40 | 88.2 | 72 | 37 | **15.4** | 10.8 – 21.2 |
| WOMD val 00000 | 286 | 0.26 | 30.3 | 2 | 2 | **7.6** | 0.9 – 27.4 |

**Gate (pre-registered):** R_WOMD ≥ 30.8 / h and ≥ 10 events. **Not met.** By the written rule the outcome is
"inconclusive" (2 < 10 events), which calls for 2 more files. But the data already speak against it: if WOMD's true
rate were 2 × nuScenes, the chance of seeing ≤ 2 of the 39 combined events in WOMD is **p = 0.02** (exact
conditional binomial test). The WOMD point estimate is half the nuScenes rate, not double.

**Base-rate check (reported only, not part of the gate).** (follower, t) pairs over the same exposure keyframes:

| | nuScenes | WOMD 00000 |
|---|---|---|
| P(follower brakes hard within 2 s \| ego brakes) | 0.51 (56 / 109) | 1.00 (6 / 6) |
| P(follower brakes hard within 2 s \| ego does not brake) | 0.065 (497 / 7,680) | 0.099 (82 / 824) |
| Lift (ego brakes vs not) | 7.9× | ~10× (n = 6) |
| Any moving vehicle, share of keyframes with a ≤ −2 m/s² (noise floor) | 0.040 | 0.040 |

The 51% from nuScenes is real co-braking, not annotation noise: it is 8× the no-brake rate, and the
noise floor is the same in both datasets (4.0%). The concern raised earlier is withdrawn.

**Why WOMD is lower:** WOMD's interactions are mined between *agents*, not around the ego. The Waymo SDC brakes hard
3× less often than the nuScenes ego (30 vs 88 episodes / h), and it is stationary more often (0.26 ego-moving hours
in 286 × 9 s scenes ≈ 37% of the time).

### Step 0 final (2026-11-07): files 00000–00002, same code and gate → **NO-GO**

| Dataset | Scenes | Ego-moving h | Ego brake episodes / h | With follower | Strong events | **R (strong / h)** | 95% CI (Poisson) |
|---|---|---|---|---|---|---|---|
| nuScenes official train | 700 | 2.40 | 88.2 | 72 | 37 | **15.4** | 10.8 – 21.2 |
| WOMD val 00000–00002 | 879 | 0.81 | 40.6 | 11 | 9 | **11.1** | 5.1 – 21.0 |

- Gate: R_WOMD ≥ 30.8 / h and ≥ 10 events. **Not met** (11.1 / h, 9 events). The rule's extra-files step is now
  used up, so the decision is final: **no-go** for the full WOMD download and the v3-on-WOMD plan as designed.
- If WOMD's true rate were 2 × nuScenes, WOMD would get ~18.5 of the 46 combined events; it got 9 (exact
  conditional binomial **p = 0.002**). The 95% CI upper bound for WOMD (21.0 / h) is below the 30.8 threshold.
- Base rates (reported only): P(follower brakes hard | ego brakes) 0.82 (31 / 38) vs 0.075 when not; vehicle noise
  floor 0.041 (nuScenes 0.51 / 0.065 / 0.040). The reaction is just as clear in WOMD, but the ego rarely creates it.
- Raw output: `outputs/womd_step0/womd_val00000-00002.json` (local).

## Future work
- **(c) Agent–agent stand-in ego.** WOMD's interactions are mined between agents, not around the SDC. Re-centre
  training samples on another vehicle that has a leader–follower partner. Treat that vehicle as the "ego": its
  logged future becomes the plan token, and the follower is the target to predict. This would turn WOMD's many
  agent–agent interactions into ego-plan training data. It needs its own spec (sample selection rule, how to
  handle the real SDC, a check that the follower test still measures the right thing). Not planned yet.
- Cheaper first test of the follower weakness, on nuScenes: retrain with the ego-plan token never dropped and rerun
  the follower test.

---

## Argoverse 2 (AV2): same step-0 check (pre-registered 2026-11-07, before any AV2 data was read)

Why look at it: the AV2 paper says its 250k motion-forecasting scenarios were "mined for interesting and challenging
interactions between the autonomous vehicle and other actors"
([paper](https://arxiv.org/abs/2301.00493)). That is ego-centric mining, which WOMD lacks. Licence CC BY-NC-SA 4.0
(non-commercial). Public S3 bucket `s3://argoverse/datasets/av2/motion-forecasting/{train,val,test}/`, with no login.
One folder per scenario (`scenario_<id>.parquet` ~70–270 KB + `log_map_archive_<id>.json` ~50–200 KB). ScenarioNet
has `convert_argoverse2` (needs the `av2` package; ego = track `"AV"`; box sizes are fixed estimates because AV2 has
none).

- **Sample:** the first 2,000 scenario folders of AV2 `val` in S3 key order (the IDs are random UUIDs, so the order
  carries no selection), trajectories (`.parquet`) only. The IDs are saved locally. **Any future AV2 test set must
  exclude them.**
- **Same code and definition** as the WOMD / nuScenes step 0 (`scripts/womd.py follower`, new `--av2` reader).
  2 Hz = AV2 steps 0, 5, …, 105 (110 steps at 10 Hz). Ego = track `AV`. Vehicles = object_type vehicle, bus,
  motorcyclist (as nuScenes `vehicle.*` without bicycles). Valid = a row exists for that track and step.
- **Same gate:** R_AV2 ≥ 2 × R_nuScenes = **30.8 strong events / h** and ≥ 10 events → go for an AV2 v3 spec.
  Otherwise no-go. Base rates are reported only.

### AV2 result (2026-11-07) → **NO-GO**

| Dataset | Scenes | Ego-moving h | Ego brake episodes / h | With follower | Strong events | **R (strong / h)** | 95% CI |
|---|---|---|---|---|---|---|---|
| nuScenes official train | 700 | 2.40 | 88.2 | 72 | 37 | **15.4** | 10.8 – 21.2 |
| WOMD val 00000–00002 | 879 | 0.81 | 40.6 | 11 | 9 | **11.1** | 5.1 – 21.0 |
| AV2 val, first 2,000 | 2,000 | 2.62 | 57.2 | 35 | 17 | **6.5** | 3.8 – 10.4 |

- Gate (≥ 30.8 / h, ≥ 10 events): **not met**. AV2 has *fewer* strong follower events per hour than nuScenes
  (whose CI is 10.8–21.2). Ratio ≥ 2 rejected at p < 1e-7 (17 of 54 combined events vs 37 expected).
- Base rates (reported only): P(follower brakes hard | ego brakes) 0.56 (38 / 68) vs 0.052 when not, a lift of 11×.
  The vehicle noise floor is 0.073, about twice nuScenes / WOMD (AV2 tracks look noisier), which if anything
  inflates the AV2 count.
- Reading: despite the paper's wording, AV2's mining around the AV does not produce more "ego brakes, follower
  reacts" moments per hour of driving than nuScenes. Neither public dataset has the ego–follower density we hoped for.
- Sample: deleted after the check (2026-11-07). The IDs can be rebuilt: the first 2,000 folders of the S3 `val/` listing in key order. Raw output:
  `outputs/womd_step0/av2_val2000.json` (local).
