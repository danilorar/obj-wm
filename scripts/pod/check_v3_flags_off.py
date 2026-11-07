"""Pre-training checks for the v3 ablations (v3b --plan-xattn, v3c --stand-in-ego), on CPU.

1. Flags off = d_rerun: the current code with the new flag off is compared with the code at --base
   (the commit the ablation branches start from, loaded with `git show`):
   a. same parameter init under the same seed (so a flags-off training run is the same run),
   b. identical eval outputs with the d_rerun weights (plan on and off),
   c. identical train-mode loss and gradients for one step under the same seed (dropout, plan dropout),
   d. identical dataset items (train split) when the dataset has the new option and it is off.
2. Flag on (sanity): forward + backward are finite; v3b: with the plan off, the output does not depend on the plan.
3. --plot-stand-in DIR (v3c): plots of stand-in-ego samples and a plan-frame check, stand-in vs real ego.

Usage: python scripts/pod/check_v3_flags_off.py [--checkpoint ...] [--plot-stand-in DIR]
"""

import argparse
import inspect
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.dataset_v2 import TrajectoryDatasetV2  # noqa: E402
from utils.models_v2 import V2Predictor  # noqa: E402
from utils.training_utils import set_seed  # noqa: E402

D_FLAGS = dict(predict_agents=16, ego_plan=True, plan_drop=0.2, plan_queries=True, target_split=True)
DATA = str(ROOT / "data/processed_v2/nuscenes_{}_prediction_samples_map.pkl")


def old_module(base, path, name):
    """Module `path` as it is at git commit `base`."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:{path}"], check=True,
                         capture_output=True, text=True).stdout
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / path)
    exec(compile(src, f"{base}:{path}", "exec"), mod.__dict__)
    return mod


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def flags_off(args, anchors, val_batch, train_ds_new, train_ds_old):
    old = old_module(args.base, "utils/models_v2.py", "old_models_v2")
    results = []

    # a. init
    set_seed(42); new_m = V2Predictor(anchors, **D_FLAGS)
    set_seed(42); old_m = old.V2Predictor(anchors, **D_FLAGS)
    sn, so = new_m.state_dict(), old_m.state_dict()
    results.append(check("same parameter names and init under seed 42",
                         sn.keys() == so.keys() and all(torch.equal(sn[k], so[k]) for k in sn)))

    # b. eval outputs with the d_rerun weights
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    new_m.load_state_dict(state, strict=True); old_m.load_state_dict(state, strict=True)
    new_m.eval(); old_m.eval()
    with torch.no_grad():
        for plan in (True, False):
            tn, ln = new_m(val_batch, plan=plan)
            to, lo = old_m(val_batch, plan=plan)
            results.append(check(f"d_rerun weights, eval, plan {'on' if plan else 'off'}: identical traj and logits "
                                 f"(max |diff| {max((tn - to).abs().max().item(), (ln - lo).abs().max().item()):.1e})",
                                 torch.equal(tn, to) and torch.equal(ln, lo)))

    # c. one training step: loss and gradients under the same RNG state
    new_m.train(); old_m.train()
    tb = collate([train_ds_new[i] for i in range(32)])
    grads = []
    for m in (new_m, old_m):
        m.zero_grad()
        torch.manual_seed(7)
        traj, logits = m(tb)
        loss, _ = m.loss(tb, traj, logits)
        loss.backward()
        grads.append((loss.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}))
    (l1, g1), (l2, g2) = grads
    results.append(check(f"train mode, one step: identical loss ({l1.item():.6f} vs {l2.item():.6f}) and gradients",
                         torch.equal(l1, l2) and g1.keys() == g2.keys() and all(torch.equal(g1[k], g2[k]) for k in g1)))

    # d. dataset items
    if train_ds_old is not None:
        same = True
        for i in range(0, len(train_ds_new), max(1, len(train_ds_new) // 200)):
            a, b = train_ds_new[i], train_ds_old[i]
            same &= a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        results.append(check("train dataset, new option off: identical items (200 samples)", same))
    return all(results)


def flag_on(anchors, val_batch, train_batch):
    results = []
    if "plan_xattn" in inspect.signature(V2Predictor).parameters:
        set_seed(42)
        m = V2Predictor(anchors, **D_FLAGS, plan_xattn=True)
        m.train()
        traj, logits = m(train_batch)
        loss, _ = m.loss(train_batch, traj, logits)
        loss.backward()
        finite = torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        used = m.decoder.layers[0].plan_attn.in_proj_weight.grad.abs().sum() > 0
        results.append(check(f"--plan-xattn on: finite loss {loss.item():.3f} and gradients; plan attention gets "
                             f"gradient", bool(finite and used)))
        m.eval()
        moved = {**val_batch, "ego_future": val_batch["ego_future"] + 5.0}
        with torch.no_grad():
            off_a, off_b = m(val_batch, plan=False)[0], m(moved, plan=False)[0]
            on_a, on_b = m(val_batch, plan=True)[0], m(moved, plan=True)[0]
        results.append(check("--plan-xattn on, plan off: output independent of the plan; plan on: it changes",
                             torch.equal(off_a, off_b) and not torch.equal(on_a, on_b)))
    return all(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="60db661", help="commit the ablation branches start from (= d_rerun code)")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path.home() / "Desktop/wm/checkpoints/v2_d_rerun/v2_seed42_best.pth")
    parser.add_argument("--plot-stand-in", type=Path, metavar="DIR", help="v3c: stand-in-ego plots + frame check")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    anchors = np.load(ROOT / "assets/anchors.npy")
    val = TrajectoryDatasetV2(DATA.format("val"))
    val_batch = collate([val[i] for i in range(0, 640, 20)])
    train_new = TrajectoryDatasetV2(DATA.format("train"))
    train_old = None
    if "stand_in_prob" in inspect.signature(TrajectoryDatasetV2).parameters:
        old_ds = old_module(args.base, "utils/dataset_v2.py", "old_dataset_v2")
        train_old = old_ds.TrajectoryDatasetV2.__new__(old_ds.TrajectoryDatasetV2)
        train_old.samples, train_old.max_agents, train_old.max_lanes, train_old.max_points = train_new.samples, 64, 192, 21

    print(f"1. Flags off vs the code at {args.base} (d_rerun weights: {args.checkpoint})")
    ok = flags_off(args, anchors, val_batch, train_new, train_old)
    print("2. Flag on (sanity)")
    ok &= flag_on(anchors, val_batch, collate([train_new[i] for i in range(32)]))
    if args.plot_stand_in:
        if "stand_in_report" not in globals():
            parser.error("--plot-stand-in needs the v3c branch (stand-in-ego dataset option)")
        ok &= stand_in_report(args.plot_stand_in, train_new.samples)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
