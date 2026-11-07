"""Waymo Open Motion (WOMD) tools for the v3 world model. See docs/womd_feasibility.md.

Subcommands
  test-ids   pick the held-out closed-loop test scenes (fixed rule, validation files 140-149) -> JSON of IDs only
  follower   step-0 gate: count "strong follower" events (same definition for WOMD and nuScenes)

Envs: WOMD needs .venv_womd (tensorflow-cpu 2.15.1 + protobuf 4.25); nuScenes runs in .venv (nuscenes-devkit).

  .venv_womd/bin/python scripts/womd.py test-ids --files data/womd/validation/validation.tfrecord-001{40..49}-of-00150 \
      --out sim/scene_lists/womd_test300.json
  .venv_womd/bin/python scripts/womd.py follower --womd data/womd/validation/validation.tfrecord-00000-of-00150 --out ...
  .venv/bin/python      scripts/womd.py follower --nuscenes data --out ...
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

DT = 0.5           # 2 Hz keyframes for both datasets
A_BRAKE = -2.0     # m/s^2: "strong" braking (ego and follower)
V_MIN = 2.0        # m/s: ego moving before braking; follower moving
X_BEHIND = (-30.0, -3.0)   # follower centre in the ego frame (m, x forward)
Y_LAT = 2.0                # |y| <= 2 m: same lane (approx.)
YAW_MAX = np.deg2rad(30.0)
REACT = 4                  # follower brakes within t ... t + 2 s (4 keyframes)
TEST_FILES = range(140, 150)


# ---------------------------------------------------------------- WOMD reading
def womd_scenarios(paths):
    """Yield scenario_pb2.Scenario from uncompressed scenario tfrecords."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    from scenarionet.converter.waymo.waymo_protos import scenario_pb2
    for p in paths:
        for rec in tf.data.TFRecordDataset(str(p), compression_type="").as_numpy_iterator():
            s = scenario_pb2.Scenario()
            s.ParseFromString(rec)
            yield s


def _track_arrays(track):
    st = track.states
    pos = np.array([[x.center_x, x.center_y] for x in st], np.float64)
    yaw = np.array([x.heading for x in st], np.float64)
    valid = np.array([x.valid for x in st], bool)
    return pos, yaw, valid


# ---------------------------------------------------------------- test IDs
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_test_ids(args):
    files = [Path(f) for f in args.files]
    idx = sorted(int(f.name.split("-")[1]) for f in files)
    assert idx == list(TEST_FILES), f"test-ids needs exactly validation files 140..149, got {idx}"
    inter, rest, n_all = [], [], 0
    for s in womd_scenarios(files):
        n_all += 1
        sdc = s.tracks[s.sdc_track_index]
        pos, _, valid = _track_arrays(sdc)
        if len(valid) != 91 or not valid.all():
            continue
        if np.linalg.norm(np.diff(pos[20:91], axis=0), axis=1).sum() < 10.0:
            continue
        (inter if sdc.id in set(s.objects_of_interest) else rest).append(s.scenario_id)
    inter, rest = sorted(inter), sorted(rest)
    rng = np.random.default_rng(0)
    pick_i = sorted(rng.choice(inter, size=min(150, len(inter)), replace=False).tolist())
    left = [r for r in rest]
    pick_r = sorted(rng.choice(left, size=300 - len(pick_i), replace=False).tolist())
    ids = sorted(pick_i + pick_r)
    out = {
        "name": "womd_test300",
        "rule": "WOMD validation files 140-149; SDC valid at all 91 steps and path length >= 10 m over steps 20-90; "
                "150 with SDC in objects_of_interest + rest random (numpy default_rng(0), IDs sorted); "
                "fill from the rest if fewer than 150 interactive",
        "source": args.source,
        "files": sorted(f.name for f in files),
        "file_sha256": {f.name: _sha256(f) for f in sorted(files)},
        "counts": {"scenarios_read": n_all, "eligible_interactive": len(inter), "eligible_other": len(rest),
                   "picked_interactive": len(pick_i), "picked_other": len(pick_r)},
        "interactive_ids": pick_i,
        "scenario_ids": ids,
    }
    text = json.dumps(out, indent=1)
    Path(args.out).write_text(text + "\n")
    print(json.dumps(out["counts"]), "sha256", hashlib.sha256((text + "\n").encode()).hexdigest())


# ---------------------------------------------------------------- follower events
def kinematics(pos, yaw, valid):
    """2 Hz arrays -> speed along heading v[k] (from k-1, k) and accel a[k] = (v[k] - v[k-2]) / 1 s."""
    T = len(valid)
    v, vv = np.zeros(T), np.zeros(T, bool)
    vv[1:] = valid[1:] & valid[:-1]
    d = pos[1:] - pos[:-1]
    v[1:] = (d[:, 0] * np.cos(yaw[1:]) + d[:, 1] * np.sin(yaw[1:])) / DT
    a, av = np.zeros(T), np.zeros(T, bool)
    av[2:] = vv[2:] & vv[:-2]
    a[2:] = (v[2:] - v[:-2]) / (2 * DT)
    return v * vv, vv, a * av, av


def scene_events(ego, agents):
    """ego / agents: (pos (T,2), yaw (T,), valid (T,)) at 2 Hz; agents = vehicles only.
    Returns exposure keyframes, brake episodes, episodes with a follower, strong follower episodes."""
    epos, eyaw, evalid = ego
    T = len(evalid)
    ev, evv, ea, eav = kinematics(epos, eyaw, evalid)
    kin = [kinematics(*ag) for ag in agents]
    exposure, moments = 0, []          # moments: (t, has_follower, strong)
    for t in range(2, T - REACT):
        if not (eav[t] and evv[t - 2] and ev[t - 2] >= V_MIN):
            continue
        exposure += 1
        if ea[t] > A_BRAKE:
            continue
        c, s = np.cos(eyaw[t]), np.sin(eyaw[t])
        has_f = strong = False
        for (apos, ayaw, avalid), (v, vv, a, av) in zip(agents, kin):
            if not (avalid[t] and vv[t] and v[t] >= V_MIN):
                continue
            dx, dy = apos[t] - epos[t]
            x, y = c * dx + s * dy, -s * dx + c * dy
            dyaw = abs((ayaw[t] - eyaw[t] + np.pi) % (2 * np.pi) - np.pi)
            if not (X_BEHIND[0] <= x <= X_BEHIND[1] and abs(y) <= Y_LAT and dyaw <= YAW_MAX):
                continue
            has_f = True
            w = slice(t, t + REACT + 1)
            if np.any(av[w] & (a[w] <= A_BRAKE)):
                strong = True
        moments.append((t, has_f, strong))
    # merge consecutive brake keyframes into one episode
    episodes = []
    for t, f, s in moments:
        if episodes and episodes[-1][0] == t - 1:
            episodes[-1] = (t, episodes[-1][1] or f, episodes[-1][2] or s)
        else:
            episodes.append((t, f, s))
    return dict(exposure=exposure, brakes=len(episodes), with_follower=sum(e[1] for e in episodes),
                strong=sum(e[2] for e in episodes))


def base_rates(ego, agents):
    """Reported only (not part of the gate). Over the same exposure keyframes as scene_events:
    (follower, t) pairs split by whether the ego brakes at t, and whether that follower reaches a <= A_BRAKE within
    t ... t + 2 s; plus how often any moving vehicle has a <= A_BRAKE at a keyframe (the noise floor)."""
    epos, eyaw, evalid = ego
    T = len(evalid)
    ev, evv, ea, eav = kinematics(epos, eyaw, evalid)
    kin = [kinematics(*ag) for ag in agents]
    out = dict(pairs_brake=0, hard_brake=0, pairs_nobrake=0, hard_nobrake=0, veh_keyframes=0, veh_hard=0)
    for v, vv, a, av in kin:
        m = av & vv & (np.abs(v) >= V_MIN)
        out["veh_keyframes"] += int(m.sum())
        out["veh_hard"] += int((m & (a <= A_BRAKE)).sum())
    for t in range(2, T - REACT):
        if not (eav[t] and evv[t - 2] and ev[t - 2] >= V_MIN):
            continue
        key = "brake" if ea[t] <= A_BRAKE else "nobrake"
        c, s = np.cos(eyaw[t]), np.sin(eyaw[t])
        for (apos, ayaw, avalid), (v, vv, a, av) in zip(agents, kin):
            if not (avalid[t] and vv[t] and v[t] >= V_MIN):
                continue
            dx, dy = apos[t] - epos[t]
            x, y = c * dx + s * dy, -s * dx + c * dy
            dyaw = abs((ayaw[t] - eyaw[t] + np.pi) % (2 * np.pi) - np.pi)
            if not (X_BEHIND[0] <= x <= X_BEHIND[1] and abs(y) <= Y_LAT and dyaw <= YAW_MAX):
                continue
            w = slice(t, t + REACT + 1)
            out["pairs_" + key] += 1
            out["hard_" + key] += int(np.any(av[w] & (a[w] <= A_BRAKE)))
    return out


def womd_scenes(paths):
    for s in womd_scenarios(paths):
        arr = {tr.id: _track_arrays(tr) for tr in s.tracks}
        sub = lambda x: tuple(y[0:91:5] for y in x)          # 10 Hz -> 2 Hz, steps 0, 5, ..., 90
        sdc = s.tracks[s.sdc_track_index].id
        agents = [sub(arr[tr.id]) for tr in s.tracks if tr.object_type == 1 and tr.id != sdc]   # TYPE_VEHICLE
        yield s.scenario_id, sub(arr[sdc]), agents


AV2_VEHICLES = {"vehicle", "bus", "motorcyclist"}


def av2_scenes(paths):
    """Argoverse 2 motion-forecasting parquet files (110 steps at 10 Hz) -> 2 Hz steps 0, 5, ..., 105. Ego = track 'AV'."""
    import pandas as pd
    for p in paths:
        d = pd.read_parquet(p, columns=["track_id", "object_type", "timestep", "position_x", "position_y", "heading"])
        T = int(d.timestep.max()) + 1
        keep = np.arange(0, T, 5)
        tracks = {}
        for tid, g in d.groupby("track_id"):
            if tid != "AV" and g.object_type.iloc[0] not in AV2_VEHICLES:
                continue
            pos, yaw, valid = np.zeros((T, 2)), np.zeros(T), np.zeros(T, bool)
            ts = g.timestep.to_numpy()
            pos[ts] = g[["position_x", "position_y"]].to_numpy()
            yaw[ts], valid[ts] = g.heading.to_numpy(), True
            tracks[tid] = (pos[keep], yaw[keep], valid[keep])
        ego = tracks.pop("AV")
        yield Path(p).stem.replace("scenario_", ""), ego, list(tracks.values())


def nuscenes_scenes(dataroot):
    """Official nuScenes 'train' scenes only (= our train + val); official 'val' (= our test) is never read."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    from pyquaternion import Quaternion
    nusc = NuScenes("v1.0-trainval", dataroot=str(dataroot), verbose=False)
    train = set(create_splits_scenes()["train"])
    assert not train & set(create_splits_scenes()["val"])
    yaw_of = lambda q: Quaternion(q).yaw_pitch_roll[0]
    for scene in nusc.scene:
        if scene["name"] not in train:
            continue
        samples, tok = [], scene["first_sample_token"]
        while tok:
            samples.append(nusc.get("sample", tok))
            tok = samples[-1]["next"]
        T = len(samples)
        epos, eyaw = np.zeros((T, 2)), np.zeros(T)
        tracks = {}
        for k, smp in enumerate(samples):
            pose = nusc.get("ego_pose", nusc.get("sample_data", smp["data"]["LIDAR_TOP"])["ego_pose_token"])
            epos[k], eyaw[k] = pose["translation"][:2], yaw_of(pose["rotation"])
            for at in smp["anns"]:
                ann = nusc.get("sample_annotation", at)
                cat = ann["category_name"]
                if not cat.startswith("vehicle.") or cat == "vehicle.bicycle":
                    continue
                tr = tracks.setdefault(ann["instance_token"], (np.zeros((T, 2)), np.zeros(T), np.zeros(T, bool)))
                tr[0][k], tr[1][k], tr[2][k] = ann["translation"][:2], yaw_of(ann["rotation"]), True
        yield scene["name"], (epos, eyaw, np.ones(T, bool)), list(tracks.values())


def cmd_follower(args):
    if args.womd:
        name, scenes = "womd", womd_scenes([Path(f) for f in args.womd])
    elif args.av2:
        name, scenes = "av2", av2_scenes(sorted(Path(args.av2).glob("scenario_*.parquet")))
    else:
        name, scenes = "nuscenes_train", nuscenes_scenes(args.nuscenes)
    rows = []
    for sid, ego, agents in scenes:
        rows.append({"scene": sid, **scene_events(ego, agents), **(base_rates(ego, agents) if args.base_rate else {})})
    tot = {k: int(sum(r[k] for r in rows)) for k in ("exposure", "brakes", "with_follower", "strong")}
    if args.base_rate:
        b = {k: int(sum(r[k] for r in rows)) for k in ("pairs_brake", "hard_brake", "pairs_nobrake", "hard_nobrake",
                                                      "veh_keyframes", "veh_hard")}
        share = lambda n, d: round(n / d, 4) if d else None
        b.update(p_follower_hard_if_ego_brakes=share(b["hard_brake"], b["pairs_brake"]),
                 p_follower_hard_if_ego_not=share(b["hard_nobrake"], b["pairs_nobrake"]),
                 p_vehicle_keyframe_hard=share(b["veh_hard"], b["veh_keyframes"]))
    hours = tot["exposure"] * DT / 3600
    summary = {"dataset": name, "scenes": len(rows), "ego_moving_hours": round(hours, 3), **tot,
               "strong_per_hour": round(tot["strong"] / hours, 2) if hours else None,
               "brakes_per_hour": round(tot["brakes"] / hours, 2) if hours else None,
               "strong_share_of_followed_brakes": round(tot["strong"] / tot["with_follower"], 3)
               if tot["with_follower"] else None}
    if args.base_rate:
        summary["base_rate"] = b
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_scene": rows}, indent=1))
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test-ids")
    t.add_argument("--files", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--source", default="gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation")
    f = sub.add_parser("follower")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--womd", nargs="+", help="validation tfrecords (never files 140-149)")
    g.add_argument("--nuscenes", help="nuScenes dataroot (with v1.0-trainval)")
    g.add_argument("--av2", help="folder of Argoverse 2 scenario_*.parquet files")
    f.add_argument("--out", required=True)
    f.add_argument("--base-rate", action="store_true", help="also report follower / vehicle hard-brake base rates")
    args = p.parse_args()
    if args.cmd == "follower" and args.womd:
        bad = [w for w in args.womd if "validation" in Path(w).name and int(Path(w).name.split("-")[1]) in TEST_FILES]
        if bad or any("validation" not in Path(w).name for w in args.womd):
            sys.exit(f"refusing: follower count only on validation files outside 140-149 ({args.womd})")
    {"test-ids": cmd_test_ids, "follower": cmd_follower}[args.cmd](args)


if __name__ == "__main__":
    main()
