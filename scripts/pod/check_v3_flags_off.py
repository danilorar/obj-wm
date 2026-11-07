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


def plan_frame_stats(items):
    """Plan seen from its plan holder (slot 1): first plan step in the holder's frame vs its current velocity."""
    rows = []
    for it in items:
        now = it["agent_hist"][1, -1].numpy()
        if not it["agent_hist_valid"][1, -1] or not it["ego_future_valid"][0]:
            continue
        th = np.arctan2(now[3], now[2])
        c, s = np.cos(th), np.sin(th)
        p = it["ego_future"][0].numpy()
        d = p[:2] - now[:2]
        x, y = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        dth = np.degrees(np.angle(np.exp(1j * (np.arctan2(p[3], p[2]) - th))))
        speed = float(np.hypot(now[4], now[5]))
        rows.append((x - 0.5 * speed, y, dth, abs(np.hypot(p[2], p[3]) - 1)))
    r = np.abs(np.array(rows))
    return {"n": len(r), "x_minus_v_dt": np.median(r[:, 0]), "lateral": np.median(r[:, 1]),
            "heading_deg": np.median(r[:, 2]), "p95_lateral": np.percentile(r[:, 1], 95),
            "unit_norm_err": r[:, 3].max()}


def stand_in_report(out_dir, samples, n_scan=3000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.dataset_v2 import EGO_TYPE, is_follower

    out_dir.mkdir(parents=True, exist_ok=True)
    real = TrajectoryDatasetV2.__new__(TrajectoryDatasetV2)
    real.samples, real.max_agents, real.max_lanes, real.max_points = samples, 64, 192, 21
    real.stand_in_prob, real.predict_agents = 0.0, 16
    sub = TrajectoryDatasetV2.__new__(TrajectoryDatasetV2)
    sub.__dict__.update(real.__dict__)
    sub.stand_in_prob = 1.0
    idx = range(0, len(samples), max(1, len(samples) // n_scan))
    torch.manual_seed(0)
    items = [(i, sub[i]) for i in idx]
    swapped = [(i, it) for i, it in items if it["stand_in_slot"] >= 0]
    rate = len(swapped) / len(items)
    print(f"3. Stand-in ego (v3c), {len(items)} train samples scanned")
    print(f"  samples with a stand-in candidate: {rate:.1%} -> effective stand-in rate at p = 0.5: {rate / 2:.1%}")

    results = []
    # Structure: stand-in in slot 1 with type ego and no label; real ego in slot j as a car, no label; target kept
    ok = True
    for i, it in swapped:
        j = int(it["stand_in_slot"])
        base = real[i]
        ok &= int(it["agent_type"][1]) == EGO_TYPE and not it["agent_future_valid"][1].any()
        ok &= int(it["agent_type"][j]) == 0 and not it["agent_future_valid"][j].any()
        ok &= torch.equal(it["agent_hist"][1], base["agent_hist"][j]) and torch.equal(it["agent_hist"][j], base["agent_hist"][1])
        ok &= torch.equal(it["target_future"], base["target_future"]) and torch.equal(it["lane_points"], base["lane_points"])
        ok &= bool(it["ego_future_valid"].all())
    results.append(check("swap: stand-in in slot 1 (type ego, no label), real ego in its slot (car, no label), "
                         "target / lanes unchanged, full 12-step plan", ok))

    # Plan = the stand-in's own logged future, in the target frame (same as before the swap, from the raw sample)
    ok = True
    for i, it in swapped[:300]:
        j = int(it["stand_in_slot"])
        base = real[i]
        ok &= torch.allclose(it["ego_future"][:, :2], base["agent_future"][j])
    results.append(check("plan xy = the stand-in's logged future (target frame)", ok))

    # Plan frame: the first plan step seen from the plan holder, real ego vs stand-in
    s_real = plan_frame_stats([real[i] for i, _ in items])
    s_sub = plan_frame_stats([it for _, it in swapped])
    print(f"  plan step 1 in the holder's frame (medians of |.|):   {'real ego':>9} {'stand-in':>9}")
    for key, label in (("n", "samples"), ("x_minus_v_dt", "x - v*0.5 s [m]"), ("lateral", "lateral y [m]"),
                       ("p95_lateral", "lateral y p95 [m]"), ("heading_deg", "heading diff [deg]"),
                       ("unit_norm_err", "max | |(cos,sin)| - 1 |")):
        print(f"    {label:<28} {s_real[key]:>9.3f} {s_sub[key]:>9.3f}")
    results.append(check("plan frame matches: stand-in medians within 0.3 m / 3 deg of the real ego's, unit headings",
                         abs(s_sub["lateral"] - s_real["lateral"]) < 0.3
                         and abs(s_sub["x_minus_v_dt"] - s_real["x_minus_v_dt"]) < 0.3
                         and abs(s_sub["heading_deg"] - s_real["heading_deg"]) < 3
                         and max(s_sub["unit_norm_err"], s_real["unit_norm_err"]) < 1e-3))

    # Plots: 3 samples, target frame
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (i, it) in zip(axes, swapped[:: max(1, len(swapped) // 3)][:3]):
        j = int(it["stand_in_slot"])
        lp, lv = it["lane_points"].numpy(), it["lane_point_valid"].numpy()
        for seg, v in zip(lp, lv):
            if v.any():
                ax.plot(seg[v, 0], seg[v, 1], color="0.85", lw=1, zorder=0)
        h, hv, m = it["agent_hist"].numpy(), it["agent_hist_valid"].numpy(), it["agent_mask"].numpy()
        sp = h[1, -1]
        c, s = sp[2], sp[3]
        for a in np.flatnonzero(m):
            ax.plot(h[a, hv[a], 0], h[a, hv[a], 1], color="0.6", lw=1)
            ax.scatter(h[a, -1, 0], h[a, -1, 1], s=10, color="0.5")
            d = h[a, -1, :2] - sp[:2]
            dth = np.degrees(np.angle(np.exp(1j * (np.arctan2(h[a, -1, 3], h[a, -1, 2]) - np.arctan2(s, c)))))
            if (a in [0] + list(range(2, 17)) and a != j and hv[a, -1] and int(it["agent_type"][a]) in (0, 1)
                    and it["agent_future_valid"][a].sum() >= 6 and is_follower(c * d[0] + s * d[1], -s * d[0] + c * d[1], dth)):
                f = it["agent_future"][a].numpy()[it["agent_future_valid"][a].numpy()]
                ax.plot(f[:, 0], f[:, 1], color="tab:blue", lw=1.5, ls="--")
                ax.scatter(*h[a, -1, :2], s=60, color="tab:blue", zorder=3, label="follower (+ its future)")
        p = it["ego_future"].numpy()
        ax.plot(p[:, 0], p[:, 1], "-o", color="tab:red", ms=3, lw=2, label="stand-in plan = its logged future")
        ax.quiver(*sp[:2], c, s, color="tab:red", scale=12, width=0.006)
        ax.scatter(*sp[:2], s=90, marker="*", color="tab:red", zorder=4, label="stand-in ego (slot 1)")
        ax.scatter(*h[j, -1, :2], s=70, marker="s", color="k", zorder=4, label=f"real ego (moved to slot {j})")
        ax.scatter(*h[0, -1, :2], s=60, marker="^", color="tab:green", zorder=4, label="target (origin)")
        cx, cy = np.mean([sp[0], p[-1, 0]]), np.mean([sp[1], p[-1, 1]])
        r = max(35, 0.6 * np.hypot(p[-1, 0] - sp[0], p[-1, 1] - sp[1]) + 25)
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
        ax.set_aspect("equal"); ax.set_title(f"train sample {i}: stand-in from slot {j}")
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower left")
    fig.suptitle("v3c stand-in ego (training only): a vehicle with a follower becomes the ego; its logged future is "
                 "the plan (target frame, m)")
    fig.tight_layout()
    fig.savefig(out_dir / "stand_in_examples.png", dpi=110)
    print(f"  saved {out_dir / 'stand_in_examples.png'}")
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
