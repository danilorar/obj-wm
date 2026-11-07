# status/sim.md (sim agent)

Updated 2026-11-07.

## Done
- v3a trained (laptop); v3a/v3b/v3c checked on val: v3a passes all three pre-registered checks (1 seed, margin 0.015).
- Closed-loop start time `plan --t0` (checked on mini only). New test set `sim/scene_lists/test_v3.json` (not run).
- Amendment 3 (v3a confirmation rule) recorded before the seed runs.

## Running
- Pod (RTX 4090), `/workspace/wm-v3seeds` @ e489b34: v3a seed 123, v3a seed 2026, d_rerun control (seed 42).
  Logs `/workspace/wm-v3seeds/logs/`. ETA ~1–1.5 h.

## Needs
- Stop the pod when the checkpoints are copied back (I will say so).
- Review the cleanup proposal below. Nothing in it has been done.

---

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
