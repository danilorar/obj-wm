# obj-wm

Object-level, plan-conditioned world model for sampling-based MPC planning on nuScenes.

![mini scene-1094: CV planner hits a rider, WM planner keeps 0.83 m clearance](assets/clip_1094.gif)

*mini scene-1094 (development). Rider enters from the left. CV: at-fault collision. WM keeps 0.83 m — it decelerates 3.5 s earlier.*

## How it works

- **World model:** scene transformer predicting road-user futures conditioned on our candidate plan; trained with 20% plan dropout so it also works without one.
- **Planner:** sampling-based MPC — 35 candidate plans scored against predicted futures for risk, progress, comfort and lane offset; re-planned every 0.5 s.
- **Simulator:** nuScenes scenes in MetaDrive/ScenarioNet; other road users replay the log or react with IDM.
- **Evaluation:** pre-registered closed-loop test on 146 held-out nuScenes scenes with reactive road users.

## Results: closed loop on 146 test scenes

| Planner | At-fault | Within 1 m of others | Progress |
|---|---|---|---|
| CV | 1 | 8.3% | 0.917 |
| **WM** | **0** | **3.6%** | **0.917** |

Reactive IDM agents; 146 held-out test scenes. The 4.7 pp difference in near-miss rate is significant (97.5% CI [−8.0, −2.0], p = 1e-4); at-fault events are too rare to compare statistically.

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) · [results/](results/)

## Quickstart

Download first (free [nuScenes account](https://nuscenes.org/nuscenes#download), non-commercial): the *mini* split (`v1.0-mini.tgz`, ~500 MB) and *map expansion v1.3* (~200 MB).

```bash
mkdir -p data/nuscenes_mini && tar xf v1.0-mini.tgz -C data/nuscenes_mini/
mkdir -p data/maps && unzip nuScenes-map-expansion-v1.3.zip -d data/maps/
ln -s ../maps data/nuscenes_mini/maps
```

```bash
make setup       # create Python environments (~5 min)
make data        # process nuScenes mini into training data
make train-mini  # train the world model (30 epochs, ~30 min on a GPU)
make demo        # run one scene closed-loop, write outputs/demo/scene1094_wm_vs_cv.gif
```

The mini-only model is weak (val minADE₆ ≈ 3.5 m vs 1.24 m on full data) — the quickstart shows the pipeline works, not the numbers above. Full reproduction: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#full-reproduction-nuscenes-v10-trainval).

## Limitations

- Only 1 at-fault collision in 146 test scenes, so the result rests on near-miss rate. That measure became a co-primary endpoint after seeing an earlier run (disclosed in EXPERIMENTS.md).
- 16 road users predicted; ~15% of nearby agents fall back to constant velocity.
- The world model reacts to our plan too weakly — the car behind barely brakes when we do (brake ratio 0.3–0.4 of a real driver). Three variants were tested; none confirmed (see EXPERIMENTS.md).
- WM planner chatters: 1.4× more accel sign-flips per minute than CV. Four planner-side fixes all traded smoothness for safety. Root cause: risk flicker in WM predictions between replanning steps.

## Future work

- A physically plausible reaction to our plan, checked with the same follower / placebo test.
- Reactive road users, stage 2: swept ego footprint, merge anticipation, reacting pedestrians and cyclists.
- More predicted road users than 16 (32 / 63 slots were tested; see EXPERIMENTS.md).
- Several seeds per variant on one machine before any val threshold is applied.
- Smoother planning via temporally consistent WM predictions; planner-side fixes (switching cost, veto hysteresis, risk smoothing, risk-aware hold) were tested and each traded chattering for a safety cost.

## Data

**nuScenes** is [CC BY-NC-SA 4.0](https://www.nuscenes.org/terms-of-use) (non-commercial). This repository contains no nuScenes data, processed samples, or trained weights. MetaDrive and ScenarioNet are installed from their own repos. Code: MIT ([LICENSE](LICENSE)).

v2 builds on v1, a GNN trajectory predictor made as a course project for SSY340 at Chalmers (Group 33) with Ashok Ramasamy, Yanran Xie and Kaushik Rathnakar Shetty: [gnn-trajectory](https://github.com/danilorar/gnn-trajectory) (tag `v1-gnn` here).
