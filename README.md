# A trajectory-prediction world model for closed-loop planning

A scene transformer, trained on [nuScenes](https://www.nuscenes.org/nuscenes), predicts where nearby road users will
go in the next 6 s, given the plan our car is considering. A planner in [MetaDrive](https://github.com/metadriverse/metadrive)
uses those predictions to pick a safe plan every 0.5 s. On 146 held-out test scenes, this world-model planner spends
less time close to other road users than the same planner with constant-velocity predictions, with the same progress.
Three attempts to make the model react more strongly to our plan (v3) were not confirmed.

### Same street scene, two planners

Our car (green) re-plans every 0.5 s.
- **Left:** it assumes everyone keeps going straight at the same speed.
- **Right:** it asks the learned world model.

Watch the rider (purple).

<p align="center">
  <img src="assets/closed_loop_demo.gif" width="960" alt="The same scene driven twice: with constant-velocity predictions the car hits a rider; with the world model it slows down and passes safely">
</p>

*Left: the conflict is seen only 0.2 s ahead, and the car hits the rider. Right: the rider's move is predicted early;
the car slows down and passes. (nuScenes mini scene-1094, a development scene; other road users react to our car.)*

<p align="center">
  <img src="docs/architecture.png" width="960" alt="Model: agent, lane and ego-plan encoders, scene transformer, anchor decoder with 6 paths and probabilities for 16 agents">
</p>

## Examples (illustration)

Development scenes, not test scenes.

<p align="center"><img src="assets/example1_clip.gif" width="960" alt="Example 1: CV passes a turning trailer at 0.03 m, WM keeps 1.30 m"></p>

*Example 1 (dev100 scene-0263): CV passes a turning trailer at 0.03 m, WM slows and keeps 1.30 m (~8 s, 0.5× slow
motion and a freeze-frame at the closest moment).*

<p align="center"><img src="assets/examples_grid.png" width="960" alt="Three development scenes, CV vs WM at the CV planner's closest moment"></p>

*Picked by a fixed rule among 66 eligible development scenes (WM more distance in 25, CV in 8, equal in 33); rules 1
and 3 amended post hoc: [docs/demo_selection.md](docs/demo_selection.md).*

## Results (v2, the official model)

On **146 held-out test scenes**, the same planner is run with constant-velocity predictions (CV) and with the world
model (WM). Other road users either replay the log or react to our car.

| Planner | Other road users | At-fault collisions | Rear-ended | Progress | Time within 1 m of others |
|---|---|---|---|---|---|
| CV | replay the log | 1 | 14 | 0.921 | 10.7% |
| WM | replay the log | 0 | 16 | 0.923 | 6.0% |
| CV | react | 1 | 2 | 0.917 | 8.3% |
| WM | react | **0** | 2 | 0.917 | **3.6%** |

The decision rule was pre-registered. **The world-model planner is safer:** it spends 4.7 percentage points less time
within 1 m of others (97.5% CI [−8.0, −2.1] pp, p = 1e-4), with the same progress.

Open-loop prediction error (val, target minADE₆) is **1.239 ± 0.006 m** over 3 seeds. Training the same seed with the
same flags on two machines (laptop vs cloud GPU, different torch versions) gave 1.243 vs 1.268 m: **noise between
machines (~0.025 m) is larger than the spread over seeds**, so small val differences need several seeds to mean
anything. Every run is listed in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

<details>
<summary>More figures</summary>

<p align="center"><img src="assets/prediction_examples_val.png" width="960" alt="Val examples: 6 predicted modes vs ground truth"></p>

*Open-loop predictions on val. Green: what happened; orange: the 6 predicted futures (number = probability).*

<p align="center"><img src="assets/follower_test_val.png" width="960" alt="Follower predictions under real, braking and accelerating ego plans"></p>

*The car behind ours is predicted to slow down when we plan to brake (red), and to speed up when we accelerate (blue).*

</details>

## v3 ablations: a stronger reaction to our plan? (not confirmed)

The v2 model reacts to our plan in the right direction, but too weakly: if we brake to a stop, the car behind is still
predicted to drive into us. Three single changes were tested against criteria written down before training
(open loop, val only):
- **Brake ratio:** how much of the needed slow-down of the following car is predicted (1 = enough to stay behind us).
- **Placebo:** how much cars that should *not* react (ahead, oncoming, next lane) change when only our plan changes.
- **Val accuracy:** target minADE₆ with the plan on.

| Model | Change | Brake ratio (≥ 0.53) | Placebo (≤ 2.9 m) | Val minADE₆ (≤ 1.259 m) | Result |
|---|---|---|---|---|---|
| v2 (3 seeds) | – | 0.30 – 0.41 | 1.44 – 2.38 | 1.232 – 1.243 | reference |
| v3a (3 seeds) | plan never dropped in training | 0.545 / 0.558 / 0.609 | 2.40 / 2.42 / **3.18** | 1.205 / 1.235 / **1.267** | **not confirmed** (seed 2026 fails) |
| v3b | decoder cross-attention to the plan | 0.408 | 2.02 | 1.206 | fails the brake check |
| v3c | other cars stand in as "ego" in training | 0.382 | 1.90 | 1.208 | fails the brake check |

What it shows: always giving the model our plan (v3a) makes the reaction to it consistently stronger, but it also
leaks into cars that should not react, and accuracy does not improve on average. Even the strongest v3a seed leaves
18 of 20 following cars driving into a stopped ego. So v2 stays the official model and no v3 closed-loop test was run.

**More data?** Waymo Open Motion (WOMD) and Argoverse 2 were checked first for how often the ego brakes hard with a car
close behind, the situation that teaches this reaction. Both have fewer such events per hour than nuScenes (WOMD
11.1, AV2 6.5 vs nuScenes 15.4; the pre-set gate was 2× nuScenes), so switching datasets would not help
([docs/womd_feasibility.md](docs/womd_feasibility.md)).

## Limitations

- **Few at-fault events.** There is only 1 at-fault collision in 146 scenes, so the safety result rests on the
  close-call measure. That measure was made a primary endpoint after an earlier test run had been seen; this is
  disclosed in docs/EXPERIMENTS.md.
- **Limited coverage.** The model predicts 16 road users; the remaining ~15% nearby fall back to constant velocity.
- **Simple reactive road users.** Only cars and trucks react, there is no merge anticipation, and pedestrians and
  cyclists replay the log.
- **Weak reaction to our plan.** The predicted reaction to our plan is weaker than real drivers' (see v3).
- **Chatter (dev100, existing runs, `sim_check.py hesitation`):** acceleration sign flips per minute CV 9.7 / WM 13.4
  (replay), 8.6 / 12.0 (reactive), logged driver 8.0; stop-go events per minute 0.56 / 0.90 and 0.37 / 0.74, driver
  0.31. Each 0.5 s re-plan picks the best of 35 plans with no switching cost on a coarse speed grid.
- **Planner hesitation.** The WM planner can stop and go: in one development scene (dev100 scene-0019) it stopped for
  about 4 s in the middle of a junction while the logged driver kept moving. Being cautious lowers close calls but can
  cost smoothness; the progress measure does not capture this.
- **Small open-loop differences are noisy.** See the cross-machine note above.

## Future work

- A physically plausible reaction to our plan: losses or data that teach "stay behind a braking car", checked with
  the same follower / placebo test.
- Reactive road users, stage 2: swept ego footprint, merge anticipation, reacting pedestrians and cyclists.
- More predicted road users than 16 (tested with 32 / 63 slots, see EXPERIMENTS.md), and a cost term against
  stop-and-go.
- Several seeds per variant on one machine before any val threshold is applied.

<details>
<summary>Reproduce</summary>

Put the nuScenes v1.0-trainval metadata and the map expansion v1.3 in `data/`. For the simulator, add the `v1.0-mini`
tables in `data/nuscenes_mini/v1.0-mini/` and link the maps with `ln -s ../maps data/nuscenes_mini/maps`.

```bash
# environments: .venv (training), .venv_sim (simulator)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt scikit-learn scipy openpyxl
uv venv --python 3.10 .venv_sim
git clone --depth 1 https://github.com/metadriverse/metadrive.git sim/metadrive
git clone --depth 1 https://github.com/metadriverse/scenarionet.git sim/scenarionet
VIRTUAL_ENV=.venv_sim uv pip install -e sim/metadrive -e sim/scenarionet "nuscenes-devkit>=1.1.10" ijson
VIRTUAL_ENV=.venv_sim uv pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv_sim/bin/python -m metadrive.pull_asset

# dataset (~20 min) and training of the official v2 model
.venv/bin/python scripts/build_processed_dataset.py --lane-tokens --workers 6
FLAGS="--model v2 --predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2"
.venv/bin/python scripts/train_model.py $FLAGS --assign predicted --epochs 60 --patience 15 --patience-start 30 \
    --batch-size 64 --lr 0.001 --seed 42 --checkpoint-dir checkpoints/v2_d_rerun

# validation (expect 1.243 m with plan on for the released checkpoint), and the follower test
.venv/bin/python scripts/train_model.py $FLAGS --evaluate-only --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth
.venv/bin/python scripts/analyze_v2.py --model v2 --checkpoint checkpoints/v2_d_rerun/v2_seed42_best.pth \
    --follower-test outputs/run_d_rerun

# closed loop: convert scenes, check on the mini dev scenes, then the final test (replay and reactive)
.venv_sim/bin/python scripts/sim_check.py convert --scenes 10
.venv_sim/bin/python scripts/sim_check.py convert --set test150 --scenes 146
.venv_sim/bin/python scripts/sim_check.py evaluate --data sim/scenarios/nuscenes_mini --out outputs/sim/mini_check
.venv_sim/bin/python scripts/sim_check.py evaluate --data sim/scenarios/nuscenes_test150 --out outputs/sim/test150_v4
.venv_sim/bin/python scripts/sim_check.py evaluate --data sim/scenarios/nuscenes_test150 --reactive idm \
    --out outputs/sim/test150_reactive
```

**Flags and defaults.** Pass `--plan-drop 0.2` explicitly: the default is 0.5, which older runs in EXPERIMENTS.md
rely on. With this code the released v2 checkpoint evaluates to exactly 1.243 m (plan on) / 1.281 m (plan off). The
v3b / v3c options (`--plan-xattn`, `--stand-in-ego`) are not part of this code; they live on tags `v3b-xattn` and
`v3c-stand-in`. v3a is the same code with `--plan-drop 0`.

</details>

## Repository

- `utils/`: the dataset and the model (`models_v2.py`).
- `scripts/`: data building, training, open-loop analysis (`analyze_v2.py`) and the simulator entry point
  (`sim_check.py`).
- `sim/`: the planner, the simulator-to-model adapter and the reactive road users; `sim/scene_lists/` holds the scene
  lists (two reserved test sets that were never run).
- `docs/`: [code guide](docs/CODE_GUIDE.md), [every run and pre-registration](docs/EXPERIMENTS.md), the reactive-agent
  design and the WOMD feasibility study.
- `legacy/`, `plots/`: v1-era scripts.

Tags: `v1-gnn` (the earlier GNN model, a final project for SSY340 at Chalmers), `v3b-xattn`, `v3c-stand-in`,
`womd-nogo`.
