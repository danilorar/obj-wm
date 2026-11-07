# status/sim.md (sim agent)

Updated 2026-11-07.

## Done
- v3a confirmation (amendment 3): NOT confirmed (seed 2026 fails placebo 3.18 > 2.9 m and val 1.267 > 1.259); mean brake
  ratio 0.571. No closed-loop test. Control d_rerun retrain (pod): 1.268 vs 1.243.
- Pod checkpoints + logs copied back, checksums verified: `checkpoints/{v3a_seed123,v3a_seed2026,v2_d_rerun_current}`,
  `outputs/pod/v3seeds/`.

## Running
- Nothing.

## Needs
- Stop the pod. Network volume: `/workspace/wm` 3.5 GB, `wm-v3b` / `wm-v3c` / `wm-v3seeds` ~90 MB (all copied);
  `hf_cache` 6.3 GB and `detector_failure` 7.8 GB are not from this project.
- Go for the approved cleanup (below).

## Cleanup plan: APPROVED 2026-11-07 with changes (execute only after the pod results are reported)
1. Old repo `gnn_trajectory` is shared: no PR into its `main`, no deleting teammates' branches (Ashok, xieyanran,
   results/attack-baselines-and-modes). Only add tag `v1-gnn` there (on `origin/feature/trajectory-prediction`).
2. Final branch = `v3a-no-plan-dropout` after merging `pod-runs` and the womd-feasibility docs (feasibility study,
   NO-GO result, `womd_test300.json`). It becomes `main` of a NEW repo (Danilo creates it).
3. Tags `v3b-xattn`, `v3c-stand-in`, `womd-nogo`; then delete those branches locally; remove worktrees `../wm-pod`,
   `../wm-womd`.
4. Delete (git-ignored): `data/womd`, `data/av2`, `checkpoints/v2_check`, `checkpoints/v2_multi_check`, superseded
   `outputs/sim/test150*` GIFs (keep all JSON).
5. Scripts: `run_interaction_analysis.py`, `check_lane_tokens.py` → `legacy/`; `run_turn_angle_analysis.py` →
   `analyze_v2.py`, docs updated.
6. Docs (CLAUDE.md, CODE_GUIDE.md, EXPERIMENTS.md handoff, README): after the v3a results.
7. Pod: after copy-back, report which `/workspace` data is on the network volume (Danilo deletes it in RunPod Storage).
Rule: ask before deleting any tracked file.

<details><summary>Original proposal</summary>

## Cleanup proposal (for review, nothing changed yet)

### Branches
| Branch | State vs v3a-no-plan-dropout | Proposal |
|---|---|---|
| `v3a-no-plan-dropout` | active | After the v3a decision: merge into `v2-transformer`, then open a PR into `main` |
| `v2-transformer` | 0 ahead (fully contained in v3a) | Fast-forward to v3a, then it is the source for the PR into `main`; delete after `main` is updated |
| `main` | 9 commits not in v3a (poster PR #25, attack-baseline results), 89 behind | PR `v2-transformer` → `main` (merge, not rebase; teammates' commits stay). `main` becomes the final branch |
| `pod-runs` | 2 ahead (status/pod.md updates) | Merge into v3a, delete; remove worktree `../wm-pod` |
| `v3b-plan-xattn`, `v3c-stand-in-ego` | 1 commit each; failed the main check | Tag (`v3b-xattn`, `v3c-stand-in`) so results stay reproducible, delete the branches. Alternative: merge, as both flags default to off |
| `womd-feasibility` | 9 ahead; WOMD and AV2 step 0: NO-GO | Merge only its docs (the feasibility study + NO-GO result) or tag `womd-nogo` and delete; remove worktree `../wm-womd` |
| `origin/feature/trajectory-prediction` (v1) | fully contained | Tag `v1-gnn`, delete the branch (CLAUDE.md then refers to the tag) |
| `origin/Ashok`, `origin/xieyanran` | fully contained, Sept/Oct | Teammates' branches: ask them, then delete |
| `origin/results/attack-baselines-and-modes` | 4 commits not in v3a (but in `main`?) | Teammate's branch: leave, or delete after checking it is merged into `main` |

Final structure: `main` only (+ one working branch at a time); tags `v1-gnn`, `v3b-xattn`, `v3c-stand-in`, `womd-nogo`.

### Worktrees
- `../wm-pod` (pod-runs), `../wm-womd` (womd-feasibility): remove after the merges above. (`../wm-v3b` already removed.)

### Files and disk (all git-ignored unless noted; deleting needs your OK)
| What | Size | Proposal |
|---|---|---|
| `data/womd/` | 3.4 GB | Delete (WOMD NO-GO); the branch notes how to re-download |
| `checkpoints/v2_check`, `v2_multi_check` | 10 MB | Delete (smoke tests, not in the run table) |
| Other `checkpoints/v2_*`, `v3*` | 100 MB | Keep (all in EXPERIMENTS.md) |
| `outputs/sim/` | 318 MB | Keep reports/JSON; delete superseded runs `test150`, `test150_v2`, `test150_v3` GIFs (~keep JSON) |
| Pod network volume: `/workspace/wm`, `wm-v3b`, `wm-v3c`, `wm-v3seeds` | ? | Delete after the copy-back; the volume costs money while it exists |
| Untracked files in git | none | – |
| Tracked: `assets/closed_loop_demo.gif` 1.4 MB | | Keep (README). `.git` is 113 MB from old binaries; no history rewrite proposed |

### Outdated docs
- `CLAUDE.md`: mentions only v1/v2 branches; add v3a and the tags; `test_v3.json` rule.
- `docs/CODE_GUIDE.md`: missing `--follower-group`, `plan --t0`, `--no-shadow`; (after merge) `--plan-xattn`, `--stand-in-ego`.
- `README.md`: results still v2 only; update after the v3a decision.
- `docs/EXPERIMENTS.md`: "Next steps (handoff, after the final test)" is outdated; replace with the v3 state.
- `status/pod.md`: add the v3a seed runs.

### Scripts
- `scripts/run_interaction_analysis.py`, `scripts/check_lane_tokens.py`: v1-era, referenced only in CODE_GUIDE. Move to
  `plots/`-style `legacy/` or delete (tag `v1-gnn` keeps them).
- `plots/` (v1 plotting): keep, or move with the above.
- `scripts/run_turn_angle_analysis.py` now also holds the follower/placebo tests; the name is misleading. Optional
  rename (e.g. `analyze_v2.py`), with README/CODE_GUIDE updated.
- `scripts/pod/check_v3_flags_off.py` (on v3b/v3c branches): keep with the tags.

</details>
