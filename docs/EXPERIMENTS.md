# Experiments

One row per training run. **Add a row for every new run** (including pod runs, while they are still running).
Metrics: best-epoch **val** target minADE₆ (plan on for ego-plan runs). Test numbers do not go here
(test is looked at once, at the end; see CLAUDE.md).
Checkpoints: `checkpoints/<run>/v2_seed<N>_best.pth` + `.json`; logs: `outputs/train_v2_<run>_seed<N>.log`.

| Run | Date | Change vs previous | Key flags | Best / stop epoch | Val minADE₆ [m] | Verdict |
|---|---|---|---|---|---|---|
| v1 GRU baseline | 2026-10-19 | GATv2 + GRU decoder (branch feature/trajectory-prediction) | see README | n/a | n/a (only test numbers recorded) | Reference model from the v1 report |
| v2_check | 2026-11-05 | First v2 scene transformer, 5-epoch check | `--epochs 5` | 5 / – | 1.852 | Pipeline works; still improving |
| v2_anchor (k*) | 2026-11-05 | Anchor-based mode assignment | `--assign anchor` | 25 / 38 | 1.840 | Much worse than predicted assignment; dropped |
| angle anchors (stopped) | 2026-11-05 | Angle-based anchors | log: `train_v2_angle_anchors_seed42_stopped.log` | stopped early | – (2.377 at ep 5) | Dead mode (<3%); stopped |
| v2_predicted (k*), seed 42 | 2026-11-05 | Predicted mode assignment, 60 epochs | `--assign predicted` | 30 / – | 1.222 | Baseline v2; overfits after ep 30 |
| v2_predicted, seed 123 | 2026-11-05 | Same, seed 123 | | 28 / 38 | 1.220 | Seed spread |
| v2_predicted, seed 2026 | 2026-11-05 | Same, seed 2026 | | 30 / 40 | 1.252 | Seed spread: 1.231 ± 0.015 over 3 seeds |
| v2_topology | 2026-11-05 | + lane topology in attention bias | `--topology` | 33 / 43 | 1.226 | No gain (within seed spread) |
| v2_scale25 | 2026-11-06 | Data scaling: 25% of train | train_fraction 0.25 | 56 / – | 1.278 | Data-scaling point (first attempt cut by power-off) |
| v2_scale50 | 2026-11-06 | Data scaling: 50% of train | train_fraction 0.5 | 39 / 49 | 1.255 | Data-scaling point |
| v2_multi_check | 2026-11-05 | Block 1: multi-agent prediction, 5-epoch check | `--predict-agents 16 --epochs 5` | 5 / – | 1.621 | Check only |
| v2_b123 (a) | 2026-11-06 | Blocks 1+2+3: 16 agents + ego-plan token + soft scores | `--predict-agents 16 --ego-plan --soft-scores` (plan_drop 0.5) | 19 / 29 | 1.350 | Soft scores: minADE₆ ≈ (b), but top-1 ADE worse (3.50 vs 3.36); dropped |
| v2_b12 (b) | 2026-11-06 | Blocks 1+2, no soft scores | `--predict-agents 16 --ego-plan` (plan_drop 0.5) | 22 / 32 | 1.367 | Target accuracy drops vs single-agent v2 (1.22) |
| v2_c (c) | 2026-11-06 | Target-weighted loss | `--predict-agents 16 --ego-plan --target-split` | 28 / – (interrupted at ep 33, laptop suspend) | 1.265 | Recovers most target accuracy; plan has no effect (follower test) |
| v2_d (d) interrupted | 2026-11-06 | + plan in each agent's frame, plan dropout 0.2 | `... --plan-queries --plan-drop 0.2` | 28 / – (interrupted at ep 31, display freeze) | 1.251 (plan off 1.296; from the log) | World model v1 (used in the mini planner development); rerun below |
| v2_d_rerun (d) | 2026-11-06 | Rerun of (d) | `--predict-agents 16 --ego-plan --target-split --plan-queries --plan-drop 0.2` | 40 / 55 | 1.243 (plan off 1.281) | Current reference model (follower test: `outputs/run_d_rerun/`) |
| e32 | 2026-11-06 (pod) | (d) with 32 predicted agents | (d) flags + `--predict-agents 32` | 36 / 51 | 1.222 | Moving top-1 ADE 2.520 (16 shared slots); follower direction ✓ |
| e63 | 2026-11-06 (pod) | (d) with 63 predicted agents (all slots but ego) | (d) flags + `--predict-agents 63` | 26 / 41 | 1.228 | Moving top-1 ADE 2.606 (16 shared slots); follower direction ✓ |
| **Model selection** | 2026-11-06 | Val, rule pre-stated (Amendment 1 (2)) | `train_model.py --evaluate-only --eval-slots 16` | – | – | **d_rerun chosen**: best val moving-agent top-1 ADE (2.462 vs e32 2.520, e63 2.606); all pass the follower direction check |
| v2_d_seed123 | 2026-11-07 (pod) | (d) rerun, seed 123 | d_rerun flags + `--seed 123` (no `--eval-test`; torch 2.8.0+cu128) | 34 / 49 | 1.232 (plan off 1.260) | Variance seeds for d_rerun; not used for selection |
| v2_d_seed2026 | 2026-11-07 (pod) | (d) rerun, seed 2026 | d_rerun flags + `--seed 2026` (no `--eval-test`; torch 2.8.0+cu128) | 21 / 36 | 1.241 (plan off 1.271) | Variance seeds for d_rerun; not used for selection. 3 seeds (42/123/2026): 1.239 ± 0.006 (plan off 1.271 ± 0.011) |
| v3a | 2026-11-07 | Plan always on (pre-registered) | d_rerun flags with `--plan-drop 0` (no `--eval-test`) | 22 / 37 | 1.205 (plan off 1.502) | **Passes all three checks; selected** (brake ratio 0.545, placebo 2.40 m). 1 seed, preliminary. `outputs/v3_checks/` |
| v3b | 2026-11-07 (pod) | Plan cross-attention (branch `v3b-plan-xattn`) | d_rerun flags + `--plan-xattn` (no `--eval-test`; torch 2.8.0+cu128) | 24 / 39 | 1.206 (plan off 1.230) | Fails main check (brake ratio 0.408 < 0.53); placebo and val pass |
| v3c | 2026-11-07 (pod) | Stand-in ego (branch `v3c-stand-in-ego`) | d_rerun flags + `--stand-in-ego 1.0` (no `--eval-test`; torch 2.8.0+cu128) | 24 / 39 | 1.208 (plan off 1.233) | Fails main check (brake ratio 0.382 < 0.53); placebo and val pass |
| v3a_seed123 | 2026-11-07 (pod) | v3a, seed 123 (amendment 3) | v3a flags + `--seed 123` (no `--eval-test`; torch 2.8.0+cu128; code e489b34) | 26 / 41 | 1.235 (plan off 1.594) | Brake ratio 0.558, placebo 2.42 m: passes |
| v3a_seed2026 | 2026-11-07 (pod) | v3a, seed 2026 (amendment 3) | v3a flags + `--seed 2026` (same) | 30 / 45 | 1.267 (plan off 1.666) | Brake ratio 0.609; **fails** placebo (3.18 > 2.9 m) and val (1.267 > 1.259) |
| v2_d_rerun_current | 2026-11-07 (pod) | Control: d_rerun flags unchanged, seed 42, current code | d_rerun flags (no `--eval-test`; torch 2.8.0+cu128; code e489b34) | 22 / 37 | 1.268 (plan off 1.295) | Same seed and flags as d_rerun (laptop, torch 2.15): 1.243. Not a selection candidate |

Notes
- Follower test (val, 20 scenes, 6 s travel real / brake / accel): d_rerun 30.7 / 25.3 / 36.7 m, e32 29.5 / 23.3 / 35.5 m,
  e63 29.8 / 20.7 / 34.7 m (`outputs/run_d_rerun/`, `outputs/e32/`, `outputs/e63/`). Direction is right, but with brake-to-stop (ego 3.4 m, follower ~12 m behind) all
  still predict ~21–25 m of travel: the reaction is too weak to be physically plausible.
- e32 / e63 exist to raise planner coverage: with 16 slots, busy scenes (~22 nearby agents) get only ~68% world-model
  coverage; the rest falls back to constant velocity.
- "All agents minADE₆" is not comparable across different `--predict-agents` values (different agent sets).
  Compare runs on target minADE₆.

## Closed-loop evaluations
Planner weights frozen on the 10 mini dev scenes (col 500, progress 1, comfort 0.1, offset 0.5, veto early risk > 0.3
within 2 s). WM = v2_d_rerun unless noted. Results: `outputs/sim/<run>/report.md`.

| Run | Date | Scenes | Setting | Status |
|---|---|---|---|---|
| mini dev runs | 2026-11-06 | 10 mini | v2_d (ep 28), then v2_d_rerun | Development set (weights chosen here) |
| test150 (v1) | 2026-11-06 | 146 test | MetaDrive ego 4.52 x 1.85 m, stopped-ego collisions at fault | Superseded (classifier false positives) |
| test150_v2 | 2026-11-06 | 146 test | Zoe ego 4.08 x 1.73 m, ego < 0.5 m/s not at fault, near-miss metrics | Superseded: order-dependent (MetaDrive physics state across resets; 1 of 15 re-rendered scenes changed) |
| test150_v3 | 2026-11-06 | 146 test | v2 + fresh env per scene (deterministic), initial-overlap class, at-fault diagnosis | Superseded by v4 (footprint changed) |
| test150_v4 | 2026-11-07 | 146 test | Final replay: covering footprints, frozen setup | At fault CV 1 / WM 0; rear-ended 14 / 16; close-call WM − CV −4.7 pp [−7.6, −2.1] |
| test150_reactive | 2026-11-07 | 146 test | **Pre-registered reactive run** (two-state rule) | **Decision: WM is safer** (close-call 97.5% CI [−8.0, −2.1] pp, p = 1e-4; at fault −0.007 [−0.027, 0.000], p = 0.32; progress CI [−0.005, +0.006]). Report: `outputs/sim/test150_final_report.md` |
| dev100 | 2026-11-06 | 100 val-split scenes (`sim/scene_lists/dev100.json`) | footprint = circles covering each box (real sizes), tracking error | Planner development / crash diagnosis (never test). At fault CV 3 / WM 1, all 'no safe plan existed' (risk first seen 0.4–0.8 s before); 0 'zero predicted risk'; tracking p95 0.24 m |
| reactive_dev (stage 1) | 2026-11-07 | mini + dev100 (never test) | `--reactive idm`: log-path IDM agents; IDM tuned on dev100 (3 attempts; final s0 1.0, T 0.8, a_max 3.0, b 3.0, v0 x1.5) | Validation (a) fails (8 s median 0.65 m / p95 3.2 m; tracking law lags, not IDM), (b) passes (0/20 vs replay 18/20); dev100 reactive: rear-ended 0.23 -> 0.02 (CV), 0.24 -> 0.01 (WM); WM close-call -5.1 pp (p 3e-5). Report: `outputs/sim/reactive_dev/report.md`. Pre-registered reactive test NOT run (needs (a) to pass first) |
| reactive_dev (deviation rule) | 2026-11-07 | mini + dev100 (never test) | IDM only when the leader deviates > 0.3 m from its log; design-doc IDM | Failed on mini: 8 s p95 9.2 m (start-up transients > 0.3 m trigger chain reactions); replaced by the two-state rule |
| reactive_dev (two-state rule) | 2026-11-07 | mini + dev100 (never test) | REPLAY exactly until a deviating leader (off-log ego / REACTIVE agent) with a_IDM < a_ff; REACTIVE = min(a_track, a_IDM), 1 s smoothed feed-forward clipped to ±3; back after 1 s, \|Δs\| < 0.3 m, \|Δv\| < 0.3 m/s; design-doc IDM | Mini: (a) 0.00 m, 0 new overlaps, expert 0 at fault; (b) pass; (d) ego 10% slower: 0 rear-ends, 0 new overlaps, max decel 3.4 m/s². dev100: (a) 0.00 m, 0 new overlaps, expert 0 at fault; (b) 0/20 (replay 18/20); (d): 1 rear-end by a non-eligible creeping car (by design), 2 new overlaps explained (replayed motorcycle merge; two reactive cars merging; stage-1 limitation), max decel at b_max in 13 / 100 scenes; meets the (d) targets recorded 2026-11-07. Reactive test waits for the user's go |
| t0 option check | 2026-11-07 | 10 mini (never test) | `plan --t0 {0,4,8} --predictor cv --no-shadow --device cpu`: ego follows its log until t0, then the planner drives; metrics from t0 | t0 = 0 identical to the code before (all fields incl. traces and chosen plans, expert and CV; timing excluded). Expert at-fault 0 at t0 = 4 / 8 s. CV at-fault 1 / 1 / 0 at t0 = 0 / 4 / 8 (scene-1094 rider, as before); smooth hand-over (speed continuous, max \|a\| 3.5 vs 3.2 at t0 = 0). Outputs `outputs/sim/t0_check/` |

## Pre-registration: closed-loop test with reactive agents
Written 2026-11-06, before any reactive run on the test scenes. Nothing below may change after the first reactive
test run; deviations are reported as such.

**Question.** With agents that react to the ego (instead of log replay), does the world-model planner (WM) cause fewer
at-fault collisions than the constant-velocity planner (CV), without less progress?

**Agents.** The log-path IDM agents of `docs/reactive_agents_design.md` (vehicles keep their logged path, speed =
min(log tracking, IDM)), not MetaDrive's built-in `reactive_traffic` (too narrow: only vehicles spawning behind the ego).
Their parameters are tuned and frozen on the 10 mini dev scenes using the design doc's validation (a) and (b), before the
test run. If validation fails its targets (median along-track error < 0.2 m and p95 < 1 m at 8 s with the ego driving
the log; the follower brakes without collision), the test run does not start and this is reported.

**Fixed for the test run.** Scenes: `sim/scenarios/nuscenes_test150` (146 = official nuScenes val minus the 4 dev mini
scenes). Planner, cost weights, veto, plan grid, tracker: as in test150_v3. WM checkpoint: see Amendment 1, (2). Ego:
Zoe size. Fresh MetaDrive env per scene. One run per planner (CV, WM) plus the expert (logged ego, agents reactive).

**Collision classes** (as in test150_v3): initial overlap (first contact within 0.5 s); not at fault (the other object
hits the ego from behind, or the ego is below 0.5 m/s); at fault (all others).

**Primary endpoint.** At-fault collisions per scene, paired WM − CV over the 146 scenes: mean difference with a
bootstrap 95% CI (10,000 resamples over scenes, seed 0) and a two-sided Wilcoxon signed-rank p-value.

**Secondary endpoints** (same paired analysis): progress ratio (scenes where the logged ego moves), not-at-fault
collisions per scene, min distance, min TTC (capped at 5 s), share of steps with distance < 1 m. Also reported, not
tested: off-road, max |a| and |jerk|, per-map breakdown, WM coverage, planning time, and the at-fault diagnosis (no safe
plan / cost picked a risky plan / prediction miss).

**Decision rules.**
- "WM is safer" only if the primary CI lies entirely below 0 and progress is not worse (progress CI lower bound > −0.02).
- If the primary CI contains 0, the result is "no detectable difference", not "equal": with ~5–10 at-fault events per
  planner the test can only detect large effects; the event counts are reported with it.
- Secondary endpoints are reported with their CIs but are not used to claim the primary result.
- The reactive vs replay comparison (same planner, test150_v3 vs reactive) is descriptive: rear-ended collisions are
  expected to drop; at-fault counts are compared per planner with the same paired analysis.

**Reported regardless of outcome.** The full per-scene table (JSON), the 5 worst scenes per planner with GIFs, and a
random spot-check of 5 at-fault and 5 not-at-fault events with frames.

### Amendment 1 (2026-11-06, made before any reactive run)
Made before any reactive run on any scene (mini or test). The non-reactive test results test150_v2 had been seen when
it was written: there, the close-call share was the only significant paired difference (WM − CV −2.4 pp, 95% CI
[−4.3, −0.7], Wilcoxon p = 0.02). This amendment overrides the corresponding parts above.

**(1) Two co-primary endpoints, Bonferroni-corrected.**
- At-fault collisions per scene, and the close-call share (share of steps with ego–agent box distance < 1 m), both
  paired WM − CV over the 146 scenes.
- Each is tested at α = 0.025: a bootstrap **97.5%** CI (10,000 resamples over scenes, seed 0) and a two-sided Wilcoxon
  signed-rank p-value compared with 0.025.
- Decision: "WM is safer" if **either** 97.5% CI lies entirely below 0 **and** progress is not worse by more than 0.02
  (lower bound of the progress 95% CI > −0.02). Otherwise "no detectable difference".
- Secondary endpoints are unchanged and are no longer co-primary: progress ratio, not-at-fault collisions per scene, min
  distance, min TTC.

**(2) Checkpoint = the final model chosen on val, fixed before the reactive run.** It is not necessarily v2_d_rerun.
The candidates are the multi-agent ego-plan models in the run table, including e32 and e63 when they finish. Only val
(open loop) and the mini dev scenes (closed loop) may be used, never test. The rule is applied in order:
1. **Follower test passed** (open-loop `--follower-test`, 20 val scenes). The braking plan must lower the follower's
   predicted 6 s travel below the real plan's, and the accelerating plan must raise it above the real plan's.
2. Among those, the **best val moving-agent top-1 ADE**.
3. Tie-break: **higher coverage** (more predicted agent slots). Within 0.02 m counts as a tie.

The chosen checkpoint, the numbers behind the choice, and the date are recorded here before the reactive run starts.

**Recorded 2026-11-06, before any reactive run:** chosen checkpoint = `checkpoints/v2_d_rerun/v2_seed42_best.pth`.
Val selection by the pre-stated rule: all three candidates pass the follower test (brake < real < accel); val
moving-agent top-1 ADE on the 16 shared slots is d_rerun 2.462, e32 2.520, e63 2.606 m, so d_rerun is chosen (step 2;
no tie). This checkpoint is fixed for the reactive run.

The pass criterion in step 1 is written out here so that it is fixed. Earlier, the follower test was only read by eye.


### Validation targets for check (d) (recorded 2026-11-07, before any reactive test result)
Check (d): the logged ego driven 10% slower along its logged route, agents in `--reactive idm` mode, mini + dev100.
- 0 rear-ends by an **eligible** reactive agent that had the ego as its leader.
- 0 new agent–agent overlaps (not in the log), or each one explained and shown not to be caused by the reaction rule.
- Max agent deceleration ≤ 6 m/s² (b_max) is allowed. The number of scenes that hit the cap is reported as a limitation.

## Pre-registration: v3a "plan always on" (recorded 2026-11-07, before training)
**Question:** does the WM react more strongly to the ego plan if the plan is never dropped in training?
**Change (one only):** v2_d_rerun flags with `--plan-drop 0` instead of 0.2. Seed 42, branch `v3a-no-plan-dropout`.
No `--eval-test`; no test split and no test scenes are used. 1 seed = preliminary result.

**Metrics** (open loop, `analyze_v2.py --follower-test DIR --follower-group G`, the same 20 val scenes as
before, ego plan re-timed to brake to a stop at −4 m/s² or accelerate at +2 m/s²):
- *Brake reaction ratio* (followers 2–15 m behind the ego in its lane): (predicted travel at 6 s with the real plan −
  with the brake plan) / (drop needed to stay one car length, 4.5 m centre to centre, behind the stopped ego). The
  denominator is floored at 2 m and each ratio is clipped to [−0.5, 1.5]. Mean over the 20 followers; 1 = fully plausible.
- *Placebo |change|* (agents that should not react: `lead` 2–30 m ahead in the ego lane, `oncoming` heading
  difference ≥ 150°, `side` next lane 2.5–6 m lateral): mean over agents of ½(|travel brake − real| + |travel accel −
  real|) at 6 s, averaged over the three groups.

**Reference: the three v2 d seeds (val, computed 2026-11-07 before v3a exists; `outputs/v3a_prereg/`)**

| v2 seed | follower brake ratio | stays behind (brake) | lead | oncoming | side | placebo mean |
|---|---|---|---|---|---|---|
| 42 (d_rerun) | 0.351 | 1/20 | 3.00 | 0.91 | 3.23 | 2.38 m |
| 123 | 0.414 | 0/20 | 3.16 | 0.63 | 3.11 | 2.30 m |
| 2026 | 0.302 | 1/20 | 1.87 | 1.17 | 1.27 | 1.44 m |
| mean ± SD | 0.356 ± 0.056 | – | – | – | – | 2.04 ± 0.52 m |

**Pass criteria (all three must hold for "v3a is better"):**
1. **Main:** follower brake ratio ≥ **0.53** (= v2 mean 0.356 + max(0.15, 3 SD = 0.17)), and the direction check
   still passes (brake < real < accel in mean follower travel).
2. **Placebo:** placebo mean ≤ **2.9 m** (= highest v2 seed 2.38 + 0.5 m), and no single group above its highest v2
   seed + 1.0 m (lead ≤ 4.2, oncoming ≤ 2.2, side ≤ 4.2 m).
3. **Accuracy:** val target minADE₆ with the plan on ≤ **1.259 m** (3-seed v2 1.239 + 0.02). Plan-off numbers are
   reported but not judged (the model never sees "plan off" in training).

Reported as well, not judged: stays-behind count, best epoch, moving-agent top-1 ADE, and *specificity* (added
2026-11-07, before training) = follower mean |travel change| ½(|brake − real| + |accel − real|) / placebo mean. Higher =
the plan reaches the follower more than the agents that should ignore it. v2: seed 42 6.86 / 2.38 = **2.88**,
seed 123 6.96 / 2.30 = **3.03**, seed 2026 3.59 / 1.44 = **2.50**.
If 1 passes but 2 fails, the model has become sensitive to the plan everywhere, not only where it should be:
reported as "fails". Any pass is preliminary until it is repeated with seeds 123 and 2026.

### Addendum (recorded 2026-11-07, before v3a training): v3b, v3c, selection and a new closed-loop test set
- **v3b** (plan cross-attention) and **v3c** (stand-in ego) are judged with the **same three pass criteria and the same
  v2 references** as v3a above (follower brake ratio ≥ 0.53; placebo ≤ 2.9 m and per-group caps; val minADE₆ plan on
  ≤ 1.259 m). Specificity is reported for each, not judged.
- **Selection rule:** winner = the variant that passes all three checks with the largest follower brake ratio. Only the
  winner gets a closed-loop test, **once**, against v2_d_rerun, on the new test set below. If none passes, there is
  no closed-loop test.
- **New closed-loop test set** `sim/scene_lists/test_v3.json`: the 146 test150 scenes with new start times t0 = 4 s and
  8 s, kept only if ≥ 7 s of log remain (all scenes are 19.0–20.0 s long, so 292 episodes). Built from the scene and
  sample tables only; nothing has been run on it.
  - Caveat: the scenes are the same as test150, and the windows overlap the seconds already driven there (test150
    episodes ran the whole ~19.5 s log). The start states and the plans are new; the traffic is not unseen. No unused
    nuScenes scenes remain (train/val went into training, all test scenes into test150).
  - Start-time option added 2026-11-07 (`sim_check.py plan --t0`, checked on the mini scenes only; see the
    closed-loop table, row "t0 option check").

### Amendment 2 (recorded 2026-11-07, before v3c training): v3c stand-in probability
- v3c is trained with `--stand-in-ego 1.0` (every qualifying sample uses the stand-in ego), not 0.5.
- Reason: only 12.6% of the training samples qualify for a stand-in ego, so p = 0.5 would give the model a stand-in
  in only ~6% of samples, likely too few to learn from.
- Nothing else changes: same three pass criteria, same v2 references, same selection rule and test set.

### v3 results and selection (2026-11-07, val only; `outputs/v3_checks/v3_selection.json`)
Sanity check first: all three v3 variants gained ~0.035 m val minADE₆ with three different changes. With the current
code, v2_d_rerun and v2_d_seed123 re-evaluate to exactly 1.243 / 1.232 (plan off 1.281 / 1.260), and the data,
anchors and training code are unchanged since d_rerun (the only edit to `train_model.py` makes test scoring opt-in). The eval is not the
cause. The common cause of the gain is **not explained** (v3b / v3c also improve plan off, so it is not plan use).

| Model | Brake ratio (≥ 0.53) | Direction | Lead (≤ 4.2) | Oncoming (≤ 2.2) | Side (≤ 4.2) | Placebo (≤ 2.9 m) | Val minADE₆ (≤ 1.259) | Specificity | Stays behind |
|---|---|---|---|---|---|---|---|---|---|
| v2 seed 42 | 0.351 | ✓ | 3.00 | 0.91 | 3.23 | 2.38 | 1.243 | 2.88 | 1/20 |
| v2 seed 123 | 0.414 | ✓ | 3.16 | 0.63 | 3.11 | 2.30 | 1.232 | 3.03 | 0/20 |
| v2 seed 2026 | 0.302 | ✓ | 1.87 | 1.17 | 1.27 | 1.44 | 1.241 | 2.50 | 1/20 |
| **v3a** | **0.545 ✓** | ✓ | 3.14 | 1.37 | 2.69 | 2.40 ✓ | 1.205 ✓ | 3.03 | 2/20 |
| v3b | 0.408 ✗ | ✓ | 3.37 | 0.57 | 2.12 | 2.02 ✓ | 1.206 ✓ | 2.68 | 1/20 |
| v3c | 0.382 ✗ | ✓ | 2.36 | 0.65 | 2.70 | 1.90 ✓ | 1.208 ✓ | 3.03 | 0/20 |

**Selection rule applied: winner = v3a** (the only variant passing all three). Caveats: it passes the main check by
0.015 (0.545 vs 0.53, well inside the v2 seed spread 0.30–0.41 → about a quarter of one SD); 1 seed = preliminary;
18 of 20 followers still do not stay behind the stopped ego (the reaction is stronger but still too weak).
No closed-loop test has been run.

### Amendment 3 (recorded 2026-11-07, before the v3a seed runs): confirmation rule for v3a
- v3a is **CONFIRMED** only if, over seeds 42 / 123 / 2026 (same flags, `--seed` only, no `--eval-test`):
  1. the **3-seed mean** follower brake ratio is ≥ 0.53, and
  2. **each seed** passes the placebo check (mean ≤ 2.9 m; no group more than 1 m above its worst v2 seed: lead ≤ 4.2,
     oncoming ≤ 2.2, side ≤ 4.2 m), and
  3. **each seed** passes val (target minADE₆ with the plan on ≤ 1.259 m).
- Only if confirmed: one closed-loop test on `sim/scene_lists/test_v3.json`, v3a (seed 42) vs v2_d_rerun. If not confirmed: no
  closed-loop test.
- Control run, same time: d_rerun flags unchanged, seed 42, current code (pod). It explains the shared ~0.035 m v3 val
  gain if it also reaches ~1.21 m; it is not a selection candidate.

### v3a confirmation (2026-11-07, amendment 3; val only; `outputs/v3_checks/v3a_confirmation.json`)
| v3a seed | Brake ratio | Lead | Oncoming | Side | Placebo (≤ 2.9 m) | Val minADE₆ (≤ 1.259) | Specificity | Stays behind |
|---|---|---|---|---|---|---|---|---|
| 42 (laptop) | 0.545 | 3.14 | 1.37 | 2.69 | 2.40 ✓ | 1.205 ✓ | 3.03 | 2/20 |
| 123 (pod) | 0.558 | 2.68 | 1.02 | 3.54 | 2.42 ✓ | 1.235 ✓ | 3.21 | 2/20 |
| 2026 (pod) | 0.609 | 4.13 | 1.87 | 3.55 | **3.18 ✗** | **1.267 ✗** | 3.11 | 1/20 |
| mean | **0.571** (≥ 0.53 ✓) | | | | | 1.236 | | |

**Decision: v3a is NOT CONFIRMED** (seed 2026 fails placebo and val). By the pre-stated rule there is **no closed-loop
test**; `test_v3.json` stays unused.
- What did hold: the stronger follower reaction is consistent over seeds (0.545–0.609 vs v2 0.302–0.414), with the
  direction right in all seeds. But in seed 2026 the plan also moves agents that should not react (lead 4.13 m), and
  even then 18–19 of 20 followers still do not stay behind the stopped ego.
- Val: the 3-seed v3a mean (1.236) equals the v2 mean (1.239); the 1.205 of seed 42 was a lucky seed.
- Control: d_rerun retrained with the same seed, flags and code on the pod gives 1.268 vs 1.243 on the laptop (torch 2.8
  vs 2.15, non-deterministic GPU kernels). So run-to-run noise is ~0.025 m, larger than the 3-seed SD (0.006)
  suggested, and the "shared ~0.035 m v3 gain" is within that noise. The 0.02 m val margin is tight for this noise.

## State and next steps (handoff, 2026-11-07, end of the v3 round)
- **Official model: v2_d_rerun** (`checkpoints/v2_d_rerun/v2_seed42_best.pth`). Final closed-loop test done
  (test150 replay v4 + pre-registered reactive run): WM is safer on the close-call co-primary endpoint; at-fault shows
  no detectable difference (1 event). Report: `outputs/sim/test150_final_report.md`.
- **v3 round closed: no confirmed improvement.** v3b / v3c fail the follower check; v3a reacts more strongly on all
  seeds (brake ratio mean 0.571) but seed 2026 fails placebo and val (amendment 3). No closed-loop test; `test_v3.json`
  is unused and stays reserved.
- **Noise:** same seed and flags differ by ~0.025 m val minADE₆ between laptop and pod (d_rerun 1.243 vs 1.268), more
  than the 3-seed spread. Future val thresholds should use several seeds per variant, on one machine.
- **Data scale closed:** WOMD and AV2 have fewer strong ego–follower braking events per hour than nuScenes (NO-GO).
- Branch structure: final branch `v3a-no-plan-dropout` (to become `main` of a new repo); tags `v1-gnn`, `v3b-xattn`,
  `v3c-stand-in`, `womd-nogo`.
- Open: a follower reaction that is physically plausible (even v3a leaves 18–19 of 20 followers driving into a stopped
  ego); reactive stage 2 (swept ego footprint, merge anticipation); coverage beyond 16 slots; planner hesitation
  (stop-go at junctions, seen in a demo selection on dev100).

## WOMD / Argoverse 2 feasibility (branch womd-feasibility, 2026-11-07; code `scripts/womd.py` on tag `womd-nogo`)
- Feasibility: `docs/womd_feasibility.md`. Decisions recorded there; `.venv_womd` built in ~/Desktop/wm.
- Step 0 (follower-event gate, pre-registered in the doc): nuScenes done (15.4 strong events / h).
  Waiting for Danilo to download WOMD validation files 00000 and 00140–00149 into ~/Desktop/wm/data/womd/validation/.
- **womd_test300 fixed (2026-11-07, before step 0 and before any training):** `sim/scene_lists/womd_test300.json`,
  sha256 `9ce970748867e6193f5064e5035e336d80d8c36d9a89af27bcb573468b32cce8`. Source: v1.3.1
  `uncompressed/scenario/validation` files 140–149 (sha256 of each file is in the JSON). 2,927 scenarios read, 1,827
  eligible (223 with the SDC in objects_of_interest); picked 150 interactive + 150 others (rng seed 0).
- Step 0 result: WOMD val 00000 R = 7.6 strong follower events / h (2 events) vs nuScenes 15.4 (37). Gate (≥ 30.8 / h,
  ≥ 10 events) not met; formally "inconclusive", but ratio ≥ 2 is rejected at p = 0.02. Recommendation: no-go for the
  full download as planned. Base rates (reported only): follower hard-brake 51% vs 6.5% (nuScenes) when the ego brakes
  vs not; noise floor 4.0% in both. Details: docs/womd_feasibility.md "Step 0 results".
- **Step 0 final (files 00000–00002, same code and gate): NO-GO.** WOMD 11.1 strong follower events / h (9 events,
  95% CI 5.1–21.0) vs nuScenes 15.4; the gate needed ≥ 30.8 / h and ≥ 10 events. Ratio ≥ 2 rejected at p = 0.002.
  WOMD full download and v3-on-WOMD as designed are stopped.
- Future work: (c) agent–agent stand-in ego on WOMD (needs its own spec); nuScenes ego-plan no-dropout ablation
  as the cheaper first test. See docs/womd_feasibility.md "Future work".
- **Argoverse 2 check (2026-11-07, same code and gate, pre-registered in `cae567e`): NO-GO.** 2,000 val scenarios
  (public S3): 6.5 strong follower events / h (17 events, CI 3.8–10.4), below nuScenes 15.4. Neither WOMD nor
  AV2 is denser in ego–follower braking than nuScenes, so the data-scale route is closed for this question. Next
  candidate: the nuScenes ego-plan no-dropout ablation, or (c) the stand-in ego (works on any dataset).
