"""Planner smoothness ("chattering") metrics from closed-loop results.json files (sim_check.py plan / evaluate).

Per scene, from the ego trace at 2 Hz (every 5th 0.1 s position; interpolated logs make 0.1 s differences noisy):
    speed v = |dp| / 0.5 s, acceleration a = dv / 0.5 s, jerk = da / 0.5 s
  - accel sign flips / min: |a| > 1 m/s^2 samples whose sign is opposite to the previous |a| > 1 m/s^2 sample
    (hysteresis: small accelerations in between do not reset or create a flip)
  - stop-go events / min: moving (v > 2 m/s) -> stopped (v < 0.5 m/s) -> moving again (v > 2 m/s) = one event
  - mean |jerk| [m/s^3]
  minutes = driven time from t0 (len(trace) - 1) * 0.1 s / 60.
Per run (planner): means over scenes, plus the closed-loop numbers used by the v2.1 selection rule
(close-call share = share of steps with distance < 1 m, progress ratio, at-fault count).

    python scripts/planner_metrics.py summary RUN.json [RUN.json ...] [--ref W0.json] [--out out.json]
    python scripts/planner_metrics.py identical A.json B.json      # equal except timing / diagnosis fields?
    python scripts/planner_metrics.py changes RUN.json [--out out.json]
    python scripts/planner_metrics.py flipsplit RUN.json [--out out.json]
    python scripts/planner_metrics.py select BASE.json S0.5=RUN.json ... R=RUN.json [RS=RUN.json ...] [U2=RUN.json ...]

select (Amendment B, docs/EXPERIMENTS.md): per planner, a variant qualifies vs the baseline if
    flips/min <= max(1.1 x expert flips/min, 0.7 x baseline flips/min)   (the easier of the two)
    close-call share mean paired difference <= +1 pp, progress ratio mean paired difference >= -0.02,
    at-fault total <= baseline.
    S = the smallest qualifying w among the names S<w>; w for R+S = that w, else the S w with the largest flips cut
    among those passing the other three conditions. Selected variant = the qualifying one (S, R, RS) with the largest
    flips cut; S or R wins over RS if within 10 percentage points of RS's cut (between S and R: the larger cut).
    Amendment C adds U<N> (risk-aware switching): within U the smallest qualifying N; U counts as single-mechanism.
    The same name may be given twice (e.g. RS for CV and RS for WM from separate runs); each planner uses the file
    that contains it.

changes: every planning step whose chosen cell (target speed, offset) differs from the previous step's, classified
    (a) cost-driven  - the previous cell is still allowed at the new step but is no longer the cheapest; the driving
                       term = the weighted cost term with the largest saving (new - previous plan)
    (b) veto-driven  - the previous cell is vetoed at the new step
    (c) other        - every plan is vetoed (least-early-risk fallback) or a tie
    Also: speed reversals = consecutive speed changes in opposite directions <= 2 s apart (a swing up-down / down-up).
"""

import argparse
import json
from pathlib import Path

import numpy as np

FLIP_A, STOP_V, GO_V = 1.0, 0.5, 2.0
TIMING = {"plan_ms", "plan_ms_p95", "predict_ms", "cost_ms",
          "prev_vetoed", "prev_total", "prev_terms", "all_vetoed", "n_held", "hold_applied"}        # timing + diagnosis-only fields


def scene_metrics(trace, dt=0.1, every=5):
    p = np.asarray(trace, float)[::every]
    minutes = max((len(trace) - 1) * dt / 60.0, 1e-9)
    h = every * dt
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / h
    a = np.diff(v) / h
    jerk = np.diff(a) / h
    flips, last = 0, 0
    for x in a:
        if abs(x) > FLIP_A:
            s = int(np.sign(x))
            flips += last != 0 and s != last
            last = s
    events, state = 0, None                       # state: None (not yet moving), "moving", "stopped"
    for x in v:
        if x > GO_V:
            events += state == "stopped"
            state = "moving"
        elif x < STOP_V and state == "moving":
            state = "stopped"
    return dict(flips_per_min=flips / minutes, stop_go_per_min=events / minutes,
                mean_abs_jerk=float(np.abs(jerk).mean()) if len(jerk) else 0.0, minutes=minutes)


def run_rows(rows):
    out = []
    for m in rows:
        r = scene_metrics(m["trace"])
        r.update(scene=m["scene"], t0_s=m.get("t0_s", 0.0), close_call=m["share_dist_below_1m"],
                 progress=m["progress_ratio"], at_fault=m["at_fault"])
        out.append(r)
    return out


def summarize(rows):
    pr = np.array([r["progress"] for r in rows], float)
    return dict(n=len(rows), flips_per_min=float(np.mean([r["flips_per_min"] for r in rows])),
                stop_go_per_min=float(np.mean([r["stop_go_per_min"] for r in rows])),
                mean_abs_jerk=float(np.mean([r["mean_abs_jerk"] for r in rows])),
                close_call=float(np.mean([r["close_call"] for r in rows])), progress=float(np.nanmean(pr)),
                at_fault=int(sum(r["at_fault"] for r in rows)))


def paired(rows, ref):
    """Mean paired differences (this run - ref) over the same scenes / start times."""
    key = lambda r: (r["scene"], r["t0_s"])
    rb = {key(r): r for r in ref}
    assert set(rb) == {key(r) for r in rows}, "runs cover different episodes"
    d = lambda k: [r[k] - rb[key(r)][k] for r in rows if np.isfinite(r[k]) and np.isfinite(rb[key(r)][k])]
    base = summarize(ref)
    return dict(d_close_call=float(np.mean(d("close_call"))), d_progress=float(np.mean(d("progress"))),
                flips_ratio=summarize(rows)["flips_per_min"] / max(base["flips_per_min"], 1e-9))


def summary(paths, ref, out):
    ref_res = json.loads(Path(ref).read_text())["results"] if ref else None
    report = {}
    head = f"{'run':40s} {'planner':14s} {'n':>4s} {'flips/min':>9s} {'stop-go/min':>11s} {'|jerk|':>7s} " \
           f"{'close-call':>10s} {'progress':>8s} {'at-fault':>8s}" + ("  vs ref: flips x, d close-call pp, d progress"
                                                                     if ref else "")
    print(head)
    for path in paths:
        res = json.loads(Path(path).read_text())
        for label, rows in res["results"].items():
            rr = run_rows(rows)
            s = summarize(rr)
            line = (f"{Path(path).parent.name + '/' + Path(path).name:40s} {label:14s} {s['n']:4d} {s['flips_per_min']:9.2f} "
                    f"{s['stop_go_per_min']:11.3f} {s['mean_abs_jerk']:7.2f} {100 * s['close_call']:9.2f}% "
                    f"{s['progress']:8.3f} {s['at_fault']:8d}")
            if ref_res is not None and label in ref_res:
                p = paired(rr, run_rows(ref_res[label]))
                s["vs_ref"] = p
                line += f"  {p['flips_ratio']:.2f}x, {100 * p['d_close_call']:+.2f} pp, {p['d_progress']:+.4f}"
            report.setdefault(str(path), {})[label] = dict(summary=s, w_switch=res.get("w_switch", 0.0), scenes=rr)
            print(line)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1))
        print(f"saved {out}")
    return report


def classify(chosen, weights):
    """Plan changes of one scene: list of dicts (t, class, driver, dv, doffset)."""
    w = dict(risk=weights["col"], progress=-weights["progress"], comfort=weights["comfort"],
             offset2=weights["offset"], switch=1.0)
    out = []
    for prev, c in zip(chosen, chosen[1:]):
        if (c["speed"], c["offset"]) == (prev["speed"], prev["offset"]):
            continue
        dv, do = c["speed"] - prev["speed"], c["offset"] - prev["offset"]
        if c.get("all_vetoed"):
            cls, drv = "c", "all vetoed"
        elif c["prev_vetoed"]:
            cls, drv = "b", "veto"
        elif c["total"] < c["prev_total"]:
            cls = "a"
            gains = {n: wt * (c[n] - c["prev_terms"][n]) for n, wt in w.items() if n in c and n in c["prev_terms"]}
            drv = min(gains, key=gains.get)
        else:
            cls, drv = "c", "tie"
        out.append(dict(t=c["t"], cls=cls, driver=drv, dv=dv, doffset=do))
    return out


def changes(path, out):
    res = json.loads(Path(path).read_text())
    report = {}
    print(f"{'planner':14s} {'scenes':>6s} {'min':>6s} {'changes/min':>11s} {'(a) cost':>9s} {'(b) veto':>9s} "
          f"{'(c) other':>9s} {'speed reversals/min':>19s}  cost-driven by term")
    for label, rows in res["results"].items():
        if label.startswith("expert"):
            continue
        cnt, drivers, minutes, rev, per_scene = {"a": 0, "b": 0, "c": 0}, {}, 0.0, 0, []
        rev_cls = {}
        for m in rows:
            ch = classify(m["chosen_plans"], res["weights"])
            mins = (len(m["trace"]) - 1) * 0.1 / 60
            minutes += mins
            for e in ch:
                cnt[e["cls"]] += 1
                if e["cls"] == "a":
                    drivers[e["driver"]] = drivers.get(e["driver"], 0) + 1
            sp = [e for e in ch if e["dv"] != 0]
            r = 0
            for e1, e2 in zip(sp, sp[1:]):
                if e1["dv"] * e2["dv"] < 0 and e2["t"] - e1["t"] <= 2.0 + 1e-9:
                    r += 1
                    key = e1["cls"] + e2["cls"]
                    rev_cls[key] = rev_cls.get(key, 0) + 1
            rev += r
            per_scene.append(dict(scene=m["scene"], minutes=mins, n_changes=len(ch), reversals=r,
                                  **{k: sum(e["cls"] == k for e in ch) for k in "abc"}, changes=ch))
        n = sum(cnt.values())
        print(f"{label:14s} {len(rows):6d} {minutes:6.1f} {n / minutes:11.2f} "
              + " ".join(f"{cnt[k]:4d} ({cnt[k] / minutes:4.2f})" for k in "abc") + f" {rev / minutes:19.2f}  "
              + ", ".join(f"{k} {v}" for k, v in sorted(drivers.items(), key=lambda x: -x[1])))
        print(f"{'':14s} reversal pairs by class (first, second change): {dict(sorted(rev_cls.items()))}")
        report[label] = dict(minutes=minutes, counts=cnt, per_min={k: cnt[k] / minutes for k in "abc"},
                             changes_per_min=n / minutes, cost_drivers=drivers, speed_reversals=rev,
                             speed_reversals_per_min=rev / minutes, reversal_pairs=rev_cls, scenes=per_scene)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1))
        print(f"saved {out}")
    return report


def flip_events(trace, dt=0.1, every=5):
    """Accel sign flips with times: list of (i, t_center, t_start, t_end). trace[k] = pose at (k + 1) * dt."""
    p = np.asarray(trace, float)[::every]
    h = every * dt
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / h
    a = np.diff(v) / h
    out, last = [], 0
    for i, x in enumerate(a):                       # a[i] uses positions at dt + h * (i, i + 1, i + 2)
        if abs(x) > FLIP_A:
            s_ = int(np.sign(x))
            if last != 0 and s_ != last:
                t0 = dt + h * i
                out.append((i, t0 + h, t0, t0 + 2 * h))
            last = s_
    return out


def flipsplit(path, out):
    """Split accel flips into (a) plan-change related and (b) same plan (tracking), and describe (b)."""
    res = json.loads(Path(path).read_text())
    report = {}
    for label, rows in res["results"].items():
        if label.startswith("expert"):
            continue
        n = dict(total=0, a_loose=0, b_loose=0, a_strict=0, b_strict=0)
        b_desc = dict(overshoot=0, near_target=0, ramp=0, target_zero=0, low_speed=0)
        yaw_b, yaw_all, gap_b = [], [], []
        for m in rows:
            tr = np.asarray(m["trace"], float)
            t10 = (np.arange(len(tr)) + 1) * 0.1 + m.get("t0_s", 0.0)
            v10 = np.r_[np.nan, np.linalg.norm(np.diff(tr, axis=0), axis=1) / 0.1]     # speed at t10 (backward diff)
            hd = np.unwrap(np.arctan2(*np.diff(tr, axis=0).T[::-1]))
            yr = np.r_[np.nan, np.nan, np.abs(np.diff(hd)) / 0.1]
            yr[v10 < 1.0] = np.nan                                                     # heading undefined at standstill
            yaw_all += list(yr[np.isfinite(yr)])
            ch = m["chosen_plans"]
            tc = [c["t"] for p_, c in zip(ch, ch[1:]) if (c["speed"], c["offset"]) != (p_["speed"], p_["offset"])]
            tp = np.array([c["t"] for c in ch])
            for i, tm, ts, te in flip_events(m["trace"]):
                tm, ts, te = tm + m.get("t0_s", 0.0), ts + m.get("t0_s", 0.0), te + m.get("t0_s", 0.0)
                n["total"] += 1
                loose = any(ts - 0.5 - 1e-9 <= t <= te + 1e-9 for t in tc)        # change in the window or 0.5 s before
                strict = any(tm - 0.5 - 1e-9 <= t <= tm + 1e-9 for t in tc)        # change within 0.5 s before the flip
                n["a_loose" if loose else "b_loose"] += 1
                n["a_strict" if strict else "b_strict"] += 1
                if loose:
                    continue
                j = int(np.searchsorted(tp, tm + 1e-9) - 1)
                tgt = ch[max(j, 0)]["speed"]
                w = (t10 >= ts - 1e-9) & (t10 <= te + 1e-9)
                err = v10[w] - tgt
                err = err[np.isfinite(err)]
                if tgt == 0:
                    b_desc["target_zero"] += 1
                if np.nanmean(v10[w]) < 2.0:
                    b_desc["low_speed"] += 1
                if len(err) and err.min() < -0.3 and err.max() > 0.3:
                    b_desc["overshoot"] += 1                                       # speed crosses the plan's target
                elif len(err) and np.abs(err).max() <= 1.0:
                    b_desc["near_target"] += 1                                     # oscillation around the target
                else:
                    b_desc["ramp"] += 1                                            # still far from the target
                gap_b += list(np.abs(err))
                yaw_b += list(yr[w][np.isfinite(yr[w])])
        nb = max(n["b_loose"], 1)
        report[label] = dict(counts=n, share_b_loose=n["b_loose"] / max(n["total"], 1),
                             share_b_strict=n["b_strict"] / max(n["total"], 1),
                             b_description={k: v / nb for k, v in b_desc.items()}, b_counts=b_desc,
                             yaw_rate_b=float(np.mean(yaw_b)) if yaw_b else None,
                             yaw_rate_all=float(np.mean(yaw_all)), speed_gap_b=float(np.mean(gap_b)) if gap_b else None)
        r = report[label]
        print(f"{label}: flips {n['total']}; (b) same plan: {n['b_loose']} = {100 * r['share_b_loose']:.0f}% "
              f"(loose: no change in the accel window or 0.5 s before) / {n['b_strict']} = {100 * r['share_b_strict']:.0f}% "
              f"(strict: no change within 0.5 s before the flip)")
        print(f"  (b) loose: crosses target {b_desc['overshoot']}, within 1 m/s of target {b_desc['near_target']}, "
              f"far from target {b_desc['ramp']}; target 0: {b_desc['target_zero']}, mean speed < 2 m/s: "
              f"{b_desc['low_speed']}; mean |v - target| {r['speed_gap_b']}; |yaw rate| (b) {r['yaw_rate_b']} vs all "
              f"{r['yaw_rate_all']:.3f} rad/s")
    if out:
        Path(out).write_text(json.dumps(report, indent=1))
        print(f"saved {out}")
    return report


def select(base_path, specs, out):
    base = json.loads(Path(base_path).read_text())["results"]
    expert = summarize(run_rows(base["expert (log)"]))["flips_per_min"]
    files = {}
    for sp in specs:
        name, path = sp.split("=", 1)
        files.setdefault(name, []).append(json.loads(Path(path).read_text())["results"])
    report = dict(expert_flips_per_min=expert, planners={})
    for label in ("CV planner", "WM planner"):
        b_rows = run_rows(base[label])
        b = summarize(b_rows)
        thr = max(1.1 * expert, 0.7 * b["flips_per_min"])
        table = {}
        for name, results in files.items():
            res = next((r for r in results if label in r), None)
            if res is None:
                continue
            rows = run_rows(res[label])
            s_ = summarize(rows)
            p_ = paired(rows, b_rows)
            c = dict(flips=s_["flips_per_min"] <= thr, close_call=p_["d_close_call"] <= 0.01 + 1e-12,
                     progress=p_["d_progress"] >= -0.02, at_fault=s_["at_fault"] <= b["at_fault"])
            table[name] = dict(summary=s_, cut=1 - p_["flips_ratio"], d_close_call=p_["d_close_call"],
                               d_progress=p_["d_progress"], checks=c, qualifies=all(c.values()))
        s_names = sorted((n for n in table if n.startswith("S")), key=lambda n: float(n[1:]))
        s_q = [n for n in s_names if table[n]["qualifies"]]
        s_sel = s_q[0] if s_q else None
        side = [n for n in s_names if all(v for k, v in table[n]["checks"].items() if k != "flips")]
        w_rs = float(s_sel[1:]) if s_sel else (float(max(side, key=lambda n: table[n]["cut"])[1:]) if side else None)
        u_names = sorted((n for n in table if n.startswith("U")), key=lambda n: int(n[1:]))
        u_sel = next((n for n in u_names if table[n]["qualifies"]), None)          # Amendment C: smallest N
        cand = {k: v for k, v in (("S", s_sel), ("R", "R" if table.get("R", {}).get("qualifies") else None),
                                  ("U", u_sel), ("RS", "RS" if table.get("RS", {}).get("qualifies") else None)) if v}
        chosen = None
        if cand:
            single = [cand[k] for k in ("S", "R", "U") if k in cand]
            best_single = max(single, key=lambda n: table[n]["cut"]) if single else None
            if "RS" in cand and (best_single is None or table[best_single]["cut"] < table["RS"]["cut"] - 0.10):
                chosen = "RS"
            else:
                chosen = best_single
        report["planners"][label] = dict(baseline=b, flips_threshold=thr, variants=table, s_selected=s_sel,
                                         w_for_rs=w_rs, selected=chosen or "baseline")
        print(f"{label}: baseline flips/min {b['flips_per_min']:.2f}, threshold {thr:.2f} "
              f"(1.1 x expert {1.1 * expert:.2f} / 0.7 x baseline {0.7 * b['flips_per_min']:.2f}), at fault {b['at_fault']}")
        print(f"  {'variant':8s} {'flips/min':>9s} {'cut':>6s} {'stop-go':>7s} {'|jerk|':>6s} {'d close-call':>12s} "
              f"{'d progress':>10s} {'at fault':>8s}  checks")
        for n, t in table.items():
            print(f"  {n:8s} {t['summary']['flips_per_min']:9.2f} {100 * t['cut']:5.1f}% "
                  f"{t['summary']['stop_go_per_min']:7.3f} {t['summary']['mean_abs_jerk']:6.2f} "
                  f"{100 * t['d_close_call']:+11.2f}pp {t['d_progress']:+10.4f} {t['summary']['at_fault']:8d}  "
                  + " ".join(f"{k}{'+' if v else '-'}" for k, v in t["checks"].items())
                  + ("  QUALIFIES" if t["qualifies"] else ""))
        report["planners"][label]["u_selected"] = u_sel
        print(f"  S selected: {s_sel}; U selected: {u_sel}; w for R+S: {w_rs}; SELECTED: {chosen or 'baseline'}")
    report["test_v3"] = report["planners"]["WM planner"]["selected"] != "baseline"
    print(f"test_v3 gate (WM planner qualifies): {report['test_v3']}")
    if out:
        Path(out).write_text(json.dumps(report, indent=1))
        print(f"saved {out}")
    return report


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def identical(a, b):
    ra, rb = (strip(json.loads(Path(p).read_text())["results"]) for p in (a, b))
    same = ra == rb
    print(f"identical (timing fields {sorted(TIMING)} excluded): {same}")
    if not same:
        for label in ra:
            for m1, m2 in zip(ra[label], rb.get(label, [])):
                if m1 != m2:
                    diff = [k for k in m1 if m1.get(k) != m2.get(k)]
                    print(f"  {label} {m1['scene']}: differs in {diff}")
    return same


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("summary")
    p.add_argument("runs", nargs="+")
    p.add_argument("--ref", help="results.json of the w = 0 run for paired differences")
    p.add_argument("--out")
    p = sub.add_parser("changes")
    p.add_argument("run")
    p.add_argument("--out")
    p = sub.add_parser("flipsplit")
    p.add_argument("run")
    p.add_argument("--out")
    p = sub.add_parser("select")
    p.add_argument("base")
    p.add_argument("variants", nargs="+", help="NAME=results.json (S<w>, R, RS)")
    p.add_argument("--out")
    p = sub.add_parser("identical")
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "summary":
        summary(args.runs, args.ref, args.out)
    elif args.cmd == "flipsplit":
        flipsplit(args.run, args.out)
    elif args.cmd == "select":
        select(args.base, args.variants, args.out)
    elif args.cmd == "changes":
        changes(args.run, args.out)
    else:
        raise SystemExit(0 if identical(args.a, args.b) else 1)
