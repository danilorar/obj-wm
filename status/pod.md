# Pod runs

## 2026-11-06: e32 / e63 (RunPod RTX 4090, ~80 min)

Code: branch `v2-transformer` @ b672485. Torch 2.8.0+cu128 (the pod's preinstalled build, not the laptop's 2.15.0+cu132).

Shared flags, the (d) config:

```
--model v2 --seed 42 --epochs 60 --batch-size 64 --ego-plan --plan-drop 0.2 --plan-queries --target-split --patience 15 --patience-start 30
```

| Run | Extra flag | Stopped at epoch | Best epoch | Val minADE₆ (plan on) |
|---|---|---|---|---|
| e32 | `--predict-agents 32` | 51 | 36 | 1.222 m |
| e63 | `--predict-agents 63` (all real agent slots, ego skipped) | 41 | 26 | 1.228 m |

Both runs trained at the same time on the GPU, using about 13 GB together. e63 fit at batch 64.

On the laptop:
- Checkpoints: `~/Desktop/wm/checkpoints/v2_e32/`, `~/Desktop/wm/checkpoints/v2_e63/` (`v2_seed42_best.pth` + `.json`)
- Logs: `~/Desktop/wm/outputs/pod/train_v2_e32_seed42.log`, `train_v2_e63_seed42.log` (plus `smoke/`, a 1-epoch e63 test)

Cost: ~80 min of pod time on an RTX 4090.

Note: these logs and manifests also include test-split metrics, because `train_model.py` evaluated on the test split at the end of training. Don't use those numbers for model selection.

## Rule for future runs

Training must not evaluate on the test split. Once the sim agent has added the `--eval-test` flag (off by default), always run without it.

## 2026-11-07: v3b / v3c implemented, NOT trained (waiting for pod approval)

| Branch | Commit | Flag (default off) | Change vs d_rerun |
|---|---|---|---|
| v3b-plan-xattn | dea6255 | `--plan-xattn` | decoder cross-attention: each agent's mode queries attend to 12 ego-plan step tokens in that agent's frame (`--plan-queries` kept) |
| v3c-stand-in-ego | 97b339d | `--stand-in-ego 0.5` | training only: a vehicle with a follower becomes the ego, its logged future = the plan; effective rate ~6% (12.6% of samples qualify) |

Both branch from 60db661 (v3a tip). Checks: `scripts/pod/check_v3_flags_off.py` (flags off identical to d_rerun code;
v3c `--plot-stand-in` plots + plan-frame check), all pass.
Needs: pod approval; no v3b/v3c pass criteria recorded yet; branches not pushed.
