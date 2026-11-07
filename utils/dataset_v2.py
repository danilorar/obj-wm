"""v2 dataset: turns one processed_v2 sample into fixed-size agent and lane-segment tokens.

Agent order: target (0), ego (1), then neighbours sorted by current distance to the target,
so the "closest max_agents" cut never drops the target or the ego.
All coordinates are in the target frame (target at the origin, heading along +x).
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

DT = 0.5  # seconds between keyframes (2 Hz)

AGENT_TYPES = ["car", "truck_bus", "pedestrian", "cyclist", "ego", "other"]
EGO_TYPE = AGENT_TYPES.index("ego")

# nuScenes ego vehicle is a Renault Zoe; size from the manufacturer's spec
# (width 1.73 m without mirrors, length 4.08 m), as used in the nuScenes devkit.
EGO_SIZE = (1.73, 4.08)  # (width, length) in metres

VEHICLE_TYPES = (0, 1)   # car, truck_bus


def is_follower(x, y, dth_deg):
    """Follower of an ego, in the ego frame: 2-15 m behind, |lateral| <= 1.5 m, heading within 20 deg
    (the follower-test definition, scripts/run_turn_angle_analysis.py)."""
    return -15 <= x <= -2 and abs(y) <= 1.5 and abs(dth_deg) <= 20


def agent_type(category):
    """Map a nuScenes category name to an index in AGENT_TYPES."""
    if category == "vehicle.car" or category == "vehicle.emergency.police":
        return 0
    if category.startswith(("vehicle.truck", "vehicle.bus", "vehicle.trailer",
                            "vehicle.construction", "vehicle.emergency.ambulance")):
        return 1
    if category.startswith("human.pedestrian"):
        return 2
    if category in ("vehicle.bicycle", "vehicle.motorcycle"):
        return 3
    return 5


def add_velocity(states, valid):
    """states (..., T, 4) [x, y, cos, sin] -> (..., T, 6) with [vx, vy] appended.
    Velocity at step t uses steps t-1 and t; it is 0 where either step is invalid (and at t = 0)."""
    vel = np.zeros(states.shape[:-1] + (2,), dtype=np.float32)
    both = valid[..., 1:] & valid[..., :-1]
    vel[..., 1:, :] = (states[..., 1:, :2] - states[..., :-1, :2]) / DT * both[..., None]
    return np.concatenate([states, vel], axis=-1)


class TrajectoryDatasetV2(Dataset):
    """
    Returns per sample (A = max_agents, L = max_lanes, P = max_points, T = 5 past, F = 12 future):

    agent_hist        (A, T, 6)  [x, y, cos, sin, vx, vy]
    agent_hist_valid  (A, T)     bool
    agent_type        (A,)       index into AGENT_TYPES
    agent_size        (A, 2)     width, length
    agent_mask        (A,)       bool, real agent slot
    agent_future      (A, F, 2)  xy (ego future is not used; its slot stays invalid)
    agent_future_valid(A, F)     bool
    lane_points       (L, P, 4)  [x, y, cos, sin]
    lane_point_valid  (L, P)     bool
    lane_is_connector (L,)       bool
    lane_mask         (L,)       bool, real lane slot
    lane_adj          (L, L)     bool, lane_adj[i, j] = segment j succeeds segment i
    target_future     (F, 2)     = agent_future[0]
    target_future_valid (F,)
    ego_future        (F, 4)     [x, y, cos, sin], the ego plan (input of the ego-plan token, not a label)
    ego_future_valid  (F,)       bool
    stand_in_slot     ()         only with stand_in_prob > 0: slot the stand-in ego came from, -1 = real ego

    stand_in_prob > 0 (v3c, training only): with this probability per sample, a non-ego vehicle that has a follower
    becomes the "ego" (see _stand_in). The draw uses torch's RNG (seeded per DataLoader worker), so runs are
    reproducible; with stand_in_prob = 0 nothing is drawn and the items are exactly as before.
    """

    def __init__(self, path, max_agents=64, max_lanes=192, max_points=21, stand_in_prob=0.0, predict_agents=16):
        with open(path, "rb") as f:
            self.samples = pickle.load(f)
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.max_points = max_points
        self.stand_in_prob = stand_in_prob
        self.predict_agents = predict_agents

    def __len__(self):
        return len(self.samples)

    def _agents(self, s):
        A = self.max_agents
        T, F = s["agent_past"].shape[1], s["agent_future"].shape[1]

        hist = add_velocity(s["agent_past"], s["agent_past_valid"])          # (N, T, 6)
        ego_hist = add_velocity(s["ego_past"][None], s["ego_past_valid"][None])  # (1, T, 6)
        types = np.array([agent_type(c) for c in s["agent_categories"]], dtype=np.int64)

        # Neighbours (index >= 1 in the stored arrays) sorted by current distance to the target
        neighbours = 1 + np.argsort(np.linalg.norm(s["agent_past"][1:, -1, :2], axis=-1), kind="stable")
        order = np.concatenate([[0], neighbours])[: A - 1]   # target + closest neighbours
        n = len(order) + 1                                    # + ego

        out = {
            "agent_hist": np.zeros((A, T, 6), np.float32),
            "agent_hist_valid": np.zeros((A, T), bool),
            "agent_type": np.full(A, AGENT_TYPES.index("other"), np.int64),
            "agent_size": np.zeros((A, 2), np.float32),
            "agent_mask": np.zeros(A, bool),
            "agent_future": np.zeros((A, F, 2), np.float32),
            "agent_future_valid": np.zeros((A, F), bool),
        }
        slots = np.concatenate([[0], np.arange(2, n)])        # slot 1 is reserved for the ego
        out["agent_hist"][slots] = hist[order]
        out["agent_hist_valid"][slots] = s["agent_past_valid"][order]
        out["agent_type"][slots] = types[order]
        out["agent_size"][slots] = s["agent_size"][order]
        out["agent_future"][slots] = s["agent_future"][order, :, :2]
        out["agent_future_valid"][slots] = s["agent_future_valid"][order]

        # Ego: past only; its future enters later as the ego-plan token (phase D)
        out["agent_hist"][1] = ego_hist[0]
        out["agent_hist_valid"][1] = s["ego_past_valid"]
        out["agent_type"][1] = EGO_TYPE
        out["agent_size"][1] = EGO_SIZE

        out["agent_mask"][:n] = True
        if self.stand_in_prob > 0:   # full future [x, y, cos, sin] per slot, for the stand-in ego plan
            out["_future4"] = np.zeros((A, F, 4), np.float32)
            out["_future4"][slots] = s["agent_future"][order]
        return out

    def _stand_in(self, out):
        """v3c: with prob stand_in_prob, a stand-in ego replaces the real ego in this training sample.

        Candidates: slots >= 2 (not the target, not the ego), car or truck/bus, valid now, speed >= 3 m/s, all F
        future steps valid, and at least one follower (is_follower in the candidate's frame: a car or truck/bus in
        a predicted slot (0, 2..predict_agents), valid now, >= 6 valid future steps) - the follower-test rules.
        One candidate is drawn uniformly. Slots 1 and j are swapped, so the stand-in sits where the ego sits:
        type 'ego' (it keeps its own size), no future label (left out of the loss like the ego), and its real
        future [x, y, cos, sin] in the target frame becomes ego_future, the same frame and format as the real
        ego plan. The real ego goes to slot j as a car (EGO_SIZE, no future label). Returns the slot or -1."""
        if torch.rand(()).item() >= self.stand_in_prob:
            return -1
        hist, hv, types = out["agent_hist"], out["agent_hist_valid"], out["agent_type"]
        mask, fv = out["agent_mask"], out["agent_future_valid"]
        now = hist[:, -1]
        pos, heading = now[:, :2], np.arctan2(now[:, 3], now[:, 2])
        speed = np.hypot(now[:, 4], now[:, 5])
        vehicle = mask & hv[:, -1] & np.isin(types, VEHICLE_TYPES)
        pred = np.zeros(len(mask), bool)
        pred[[0] + list(range(2, self.predict_agents + 1))] = True
        follower_ok = vehicle & pred & (fv.sum(-1) >= 6)
        candidates = []
        for j in np.flatnonzero(vehicle & (speed >= 3.0) & fv.all(-1)):
            if j < 2:
                continue
            c, s_ = np.cos(heading[j]), np.sin(heading[j])
            for i in np.flatnonzero(follower_ok):
                if i == j:
                    continue
                d = pos[i] - pos[j]
                dth = np.degrees(np.angle(np.exp(1j * (heading[i] - heading[j]))))
                if is_follower(c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1], dth):
                    candidates.append(j)
                    break
        if not candidates:
            return -1
        j = int(candidates[torch.randint(len(candidates), ()).item()])
        plan, plan_valid = out["_future4"][j].copy(), fv[j].copy()
        for key in ("agent_hist", "agent_hist_valid", "agent_size"):
            out[key][[1, j]] = out[key][[j, 1]]
        types[1], types[j] = EGO_TYPE, AGENT_TYPES.index("car")
        out["agent_future"][[1, j]] = 0.0
        fv[[1, j]] = False
        out["ego_future"], out["ego_future_valid"] = plan, plan_valid
        return j

    def _lanes(self, s):
        L, P = self.max_lanes, self.max_points
        segments = s["lane_points"]

        # Keep the max_lanes segments closest to the target
        dist = np.array([np.linalg.norm(seg[:, :2], axis=-1).min() for seg in segments])
        keep = np.sort(np.argsort(dist, kind="stable")[:L])
        new_index = np.full(len(segments), -1)
        new_index[keep] = np.arange(len(keep))

        out = {
            "lane_points": np.zeros((L, P, 4), np.float32),
            "lane_point_valid": np.zeros((L, P), bool),
            "lane_is_connector": np.zeros(L, bool),
            "lane_mask": np.zeros(L, bool),
            "lane_adj": np.zeros((L, L), bool),
        }
        for i, k in enumerate(keep):
            seg = segments[k][:P]
            out["lane_points"][i, : len(seg)] = seg
            out["lane_point_valid"][i, : len(seg)] = True
        out["lane_is_connector"][: len(keep)] = s["lane_is_connector"][keep]
        out["lane_mask"][: len(keep)] = True

        # Successor edges between kept segments only
        if len(s["lane_edges"]):
            a, b = new_index[s["lane_edges"][:, 0]], new_index[s["lane_edges"][:, 1]]
            ok = (a >= 0) & (b >= 0)
            out["lane_adj"][a[ok], b[ok]] = True
        return out

    def __getitem__(self, idx):
        s = self.samples[idx]
        out = {**self._agents(s), **self._lanes(s)}
        out["target_future"] = out["agent_future"][0]
        out["target_future_valid"] = out["agent_future_valid"][0]
        out["ego_future"] = s["ego_future"].astype(np.float32)
        out["ego_future_valid"] = s["ego_future_valid"]
        if self.stand_in_prob > 0:
            out["stand_in_slot"] = np.int64(self._stand_in(out))
            del out["_future4"]
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
