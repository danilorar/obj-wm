# Demo examples: how they were picked

The README examples are an **illustration**, not evidence: the test results are in `results/`. They are development
scenes (dev100 and mini, never test), driven by the frozen planner with the CV predictor and with the v2 world model
(`checkpoints/v2_d_rerun`), other road users reacting (`--reactive idm`). Code: `scripts/demo_2x2.py`.

The rule was written before rendering; two amendments were made **after** seeing numbers or images (post hoc) and are
kept here as written.

## Selection rule for the 3 examples (written 2026-11-07, before rendering any example)

Data: the closed-loop sweep already run on the current demo-3d code (outputs/demo_2x2/sel_dev100.json,
sel_mini.json): CV and WM planners (checkpoints/v2_d_rerun/v2_seed42_best.pth), reactive idm agents.
Scenes: dev100 (100) + mini (10). Never test150.

Eligible scene (all must hold):
- no initial overlap in either run, and a road user within 30 m at some point in both runs (finite min distance);
- at-fault collisions = 0 for both planners;
- the WM planner never stops while the log driver keeps moving: at no 0.5 s sample (2 Hz speeds from the driven
  positions) is the WM speed < 0.5 m/s while the logged ego speed is > 2.0 m/s.

gain = min distance (WM) - min distance (CV)   [m, closed-loop minimum box-to-box distance to any road user]

1. Clear win: the eligible scene with the largest gain.
2. Typical: the eligible scene (other than 1) whose gain is closest to the median gain over all eligible scenes.
3. Different situation: the largest-gain eligible scene whose type differs from the types of both 1 and 2.
   Type = (road user class, our manoeuvre):
   - road user class: category of the road user closest to our car at the CV planner's closest moment
     (CV re-run without the 3D camera, nearest by box-to-box distance): "vulnerable" for human.* and
     vehicle.bicycle / vehicle.motorcycle, else "vehicle";
   - our manoeuvre, from the logged ego route: heading change between the start and the end of the route
     (headings from positions 2 m apart): |change| < 30 deg = straight, else left / right turn by sign;
     route shorter than 5 m = standing.
   If no eligible scene has a different type, take the largest-gain scene not yet chosen (and say so).
Ties: higher gain first, then dev100 before mini, then lower index.
The closest moment shown in the grid = the step with the CV planner's minimum distance.

### Amendment (2026-11-07, POST-HOC: written after seeing the selection numbers, before any rendering)
The rule above gave a weak set: example 1 = scene-0021 (CV 4.93 m vs WM 6.31 m, no close call, closest moment = last
frame); example 2 = mini scene-0553 (gain 0.00 m, our car standing; median gain over all 66 eligible scenes = 0.00 m).
Split of the 66 eligible scenes: WM keeps more distance in 25, CV in 8, equal (|gain| < 0.05 m) in 33.
Changed (eligibility, gain, type and tie rules unchanged):
1. Clear win: largest gain among eligible scenes where the CV planner gets within 1 m (CV min distance < 1.0 m).
2. Typical: among eligible scenes where the planners differ (|gain| >= 0.05 m, 33 scenes; median over these),
   the scene with gain closest to that median, excluding example 1.
3. Different situation: unchanged (largest gain, type different from both 1 and 2), excluding 1 and 2.
Examples 1-3 must be three different scenes. Disclosed as post-hoc in the README caption notes.

### Amendment 2 (2026-11-07, POST-HOC: written after seeing the grid with example 3 = scene-0021)
scene-0021 (right turn, CV 4.93 m vs WM 6.31 m) had no close call; its closest moment was the last frame.
Changed example 3 only: the largest-gain eligible scene whose type differs from both example 1 and example 2,
among scenes where the CV planner gets within 1 m (CV min distance < 1.0 m); must differ from examples 1 and 2.
Examples 1 and 2 unchanged (scene-0263, scene-0150; both type vehicle / straight). Disclosed in the caption notes.

## Rendering notes
- In scene-0019 the WM planner hesitates (stop-go, then a ~4 s stop mid-junction while the logged driver kept moving):
  multimodal pedestrian predictions leave only "stop" plans risk-free, and the speed grid (0, 2, 4, … 12 m/s) has no
  slow-creep option between 0 and 2 m/s. Safe but jerky. Future work: a finer speed set or temporal smoothing of the plan choice.
- MetaDrive draws some large vehicles with smaller models (e.g. the 16 m trailer in example 1 as a ~6 m box truck);
  the top-down view and all distances use the real nuScenes sizes.
- render_pipeline (MetaDrive's high-quality mode) crashes offscreen in our version; the 3D uses the default renderer
  (PBR, shadows, terrain, sky, sidewalks). The 3D view updates at the 10 Hz sim rate; the top-down is interpolated to 20 fps.
