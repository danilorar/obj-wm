"""Demo examples with the v2 world model (ScenarioNet-like look): the same development scene driven by the CV planner
and by the WM planner. 3D: MetaDrive chase camera (offscreen; simplepbr + shadows + terrain + sky) with the chosen
plan on the ground. Top-down (light): faint candidate plans, chosen plan in the accent colour (as our car), each
planner's own predictions of moving nearby road users (thin, fading with time and probability). Distance strip.
Examples are picked by the rule in docs/demo_selection.md (outputs/demo_2x2/examples_selection_amended.json).

  .venv_sim/bin/python scripts/demo_2x2.py grid      # simulate (cached) + assets/examples_grid.png
  .venv_sim/bin/python scripts/demo_2x2.py videos    # outputs/demo_2x2/example{1,2,3}.mp4 (full length)
  .venv_sim/bin/python scripts/demo_2x2.py clip      # example 1, ~8 s around the closest moment, with a 0.5x
                                                     # slow-motion part and a 1 s freeze-frame:
                                                     # outputs/demo_2x2/example1_clip.mp4 + assets/example1_clip.gif
Development scenes only (mini / dev100), never test150.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------- style (light, ScenarioNet-like)
W, H, FPS = 1920, 1080, 20
MARGIN, GAP = 16, 16
PANEL_W = (W - 2 * MARGIN - GAP) // 2                 # 936
HEAD_H, CAM_H, TOP_H, STRIP_H = 56, 400, 420, 150     # video: 3D on top, top-down below, distance strip
Y_CAM = HEAD_H
Y_TOP = Y_CAM + CAM_H + 12
Y_STRIP = Y_TOP + TOP_H + 12
BG = (246, 246, 244)
PANEL_BG = (255, 255, 255)
TEXT = (28, 28, 30)
MUTED = (118, 118, 122)
BORDER = (222, 222, 222)
ROAD = (238, 238, 238)
CROSS = (226, 226, 226)
LANE_LINE = (196, 196, 196)
EDGE_LINE = (150, 150, 150)
ACCENT = (0, 170, 152)                               # our car + its chosen plan (MetaDrive's teal ego)
PLANNER_RGB = {"CV": (232, 118, 20), "WM": (36, 104, 214)}
AGENT_RGB = [(150, 86, 212), (214, 64, 138), (126, 150, 24), (168, 100, 56), (204, 150, 0), (84, 84, 190),
             (200, 72, 56), (90, 150, 196)]          # one per moving nearby road user, the same in every view
OTHER_RGB = (186, 186, 190)
FONT = Path.home() / ".local/share/fonts/Inter/InterVariable.ttf"
FONT_FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CAM_RENDER = 2                                       # 3D rendered at 2x, stored (JPEG) at panel size
CACHE_TAG = "v3"


def font(size, weight="Regular"):
    from PIL import ImageFont
    if FONT.exists():
        f = ImageFont.truetype(str(FONT), size)
        f.set_variation_by_name(weight)
        return f
    return ImageFont.truetype(FONT_FALLBACK, size)


# ---------------------------------------------------------------- scene selection
def _speeds_2hz(m):
    import numpy as np
    tr = np.asarray(m["trace"])
    return np.r_[0.0, np.linalg.norm(np.diff(tr[::5], axis=0), axis=1) / 0.5]


def select_demo_scene(json_paths, no_wm_stop=False):
    """Rule in outputs/demo_2x2/selection_rule.md: largest min_dist(WM) - min_dist(CV), reactive idm, excluding
    initial overlaps and scenes with no road user nearby. no_wm_stop (added after seeing scene-0019): the WM planner
    never comes to a stop once moving (2 Hz speed < 0.5 m/s after > 0.5 m/s)."""
    import json
    import math
    import numpy as np
    rows = []
    for jp in json_paths:
        res = json.loads(Path(jp).read_text())["results"]
        for cv, wm in zip(res["CV planner"], res["WM planner"]):
            assert cv["scene"] == wm["scene"]
            if cv["initial_overlap"] or wm["initial_overlap"]:
                continue
            a, b = float(cv["min_dist_m"]), float(wm["min_dist_m"])
            if math.isinf(a) or math.isinf(b):
                continue
            mv = _speeds_2hz(wm) > 0.5
            stops = int(np.sum(mv[:-1] & ~mv[1:]))
            if no_wm_stop and stops:
                continue
            rows.append(dict(data=str(jp), scene=cv["scene"], index=cv["index"], cv_min=a, wm_min=b, score=b - a,
                             cv_at_fault=cv["at_fault"], wm_at_fault=wm["at_fault"], wm_stops=stops))
    rows.sort(key=lambda r: -r["score"])           # stable: ties keep the file / index order
    return rows


def _logged_manoeuvre(expert):
    """straight / left / right / standing from the logged ego route (headings from positions 2 m apart)."""
    import numpy as np
    tr = np.asarray(expert["trace"])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr, axis=0), axis=1))]
    if seg[-1] < 5.0:
        return "standing"
    i0 = int(np.searchsorted(seg, 2.0))
    i1 = int(np.searchsorted(seg, seg[-1] - 2.0))
    h0 = np.arctan2(*(tr[i0] - tr[0])[::-1])
    h1 = np.arctan2(*(tr[-1] - tr[i1])[::-1])
    d = np.degrees(np.arctan2(np.sin(h1 - h0), np.cos(h1 - h0)))
    return "straight" if abs(d) < 30 else ("left" if d > 0 else "right")


def _closest_class(data_dir, index, reactive=True):
    """CV planner re-run without the camera: category of the road user nearest (box to box) to our car at the CV
    planner's closest moment. Returns (class, category, min distance, t)."""
    import numpy as np
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.scenario.utils import get_number_of_scenarios
    from sim import closed_loop as cl
    zoe = dict(vehicle_model="varying_dynamics", width=cl.EGO_SIZE_ZOE[0], length=cl.EGO_SIZE_ZOE[1])
    env = ScenarioEnv(dict(use_render=False, num_scenarios=get_number_of_scenarios(str(data_dir)),
                           data_directory=str(data_dir), horizon=10_000, log_level=50, crash_vehicle_done=False,
                           crash_object_done=False, crash_human_done=False, vehicle_config=zoe,
                           agent_configs={"default_agent": dict(use_special_color=True, spawn_lane_index=None, **zoe)}))
    best = [np.inf, None, None]

    def hook(info):
        if info["near_d"] < best[0]:
            x, y, yaw = info["pose"]
            w_, l_ = info["ego_size"]
            eb = cl.box(x, y, yaw, l_, w_)
            cand = []
            for i, st in info["snapshot"].items():
                cat = st[5]
                if i == "ego" or not (cat.startswith(cl.AGENT_PREFIXES) or cat in ("VEHICLE", "CYCLIST", "PEDESTRIAN")):
                    continue
                if np.hypot(st[0] - x, st[1] - y) > 30:
                    continue
                cand.append((eb.distance(cl.box(st[0], st[1], st[2], max(st[4], 0.3), max(st[3], 0.3))), cat))
            if cand:
                d, cat = min(cand)
                best[:] = [info["near_d"], cat, info["step"] * 0.1]

    try:
        env.reset(seed=index)
        m, _ = cl.run_scene(env, cl.CVPredictor(), reactive=reactive, step_hook=hook)
    finally:
        env.close()
    cat = best[1] or ""
    vru = cat.startswith("human.") or cat in ("vehicle.bicycle", "vehicle.motorcycle", "PEDESTRIAN", "CYCLIST")
    return ("vulnerable" if vru else "vehicle"), cat, float(m["min_dist_m"]), best[2]


def select_examples(json_paths, data_dirs):
    """The 3-example rule in docs/demo_selection.md. json_paths[i] belongs to data_dirs[i] (dev100 first, then mini)."""
    import json
    import math
    import numpy as np
    rows = []
    for order, (jp, dd) in enumerate(zip(json_paths, data_dirs)):
        res = json.loads(Path(jp).read_text())["results"]
        for cv, wm, ex in zip(res["CV planner"], res["WM planner"], res["expert (log)"]):
            assert cv["scene"] == wm["scene"] == ex["scene"]
            a, b = float(cv["min_dist_m"]), float(wm["min_dist_m"])
            if cv["initial_overlap"] or wm["initial_overlap"] or math.isinf(a) or math.isinf(b):
                continue
            if cv["at_fault"] or wm["at_fault"]:
                continue
            vw, ve = _speeds_2hz(wm), _speeds_2hz(ex)
            n = min(len(vw), len(ve))
            if np.any((vw[1:n] < 0.5) & (ve[1:n] > 2.0)):
                continue
            rows.append(dict(scene=cv["scene"], index=cv["index"], data=str(dd), order=order, cv_min=a, wm_min=b,
                             gain=b - a, manoeuvre=_logged_manoeuvre(ex)))
    rows.sort(key=lambda r: (-r["gain"], r["order"], r["index"]))
    med = float(np.median([r["gain"] for r in rows]))
    ex1 = rows[0]
    ex2 = min(rows[1:], key=lambda r: (abs(r["gain"] - med), -r["gain"], r["order"], r["index"]))
    for r in (ex1, ex2):
        r["cls"], r["closest_cat"], r["cv_min_rerun"], r["t_closest"] = _closest_class(r["data"], r["index"])
    types = {(ex1["cls"], ex1["manoeuvre"]), (ex2["cls"], ex2["manoeuvre"])}
    ex3, checked = None, []
    for r in rows:
        if r is ex1 or r is ex2:
            continue
        r["cls"], r["closest_cat"], r["cv_min_rerun"], r["t_closest"] = _closest_class(r["data"], r["index"])
        checked.append(r)
        if (r["cls"], r["manoeuvre"]) not in types:
            ex3 = r
            break
    note = ""
    if ex3 is None:
        ex3 = next(r for r in rows if r is not ex1 and r is not ex2)
        note = "no eligible scene of a different type: took the largest-gain scene not yet chosen"
    return dict(n_eligible=len(rows), median_gain=med, examples=[ex1, ex2, ex3], checked_for_3=checked, note=note,
                eligible=rows)


# ---------------------------------------------------------------- 3D chase camera
class _ChaseCam:
    """MetaDrive RGB camera behind and above the ego, heading smoothed; projects world points to pixels."""

    def __init__(self, env, size, back=13.0, up=5.5, pitch=-13.0):
        import numpy as np
        self.cam = env.engine.get_sensor("rgb_camera")
        self.env, self.size, self.back, self.up, self.pitch = env, size, back, up, pitch
        hfov, vfov = self.cam.get_lens().getFov()
        self.fx = size[0] / 2 / np.tan(np.radians(hfov) / 2)
        self.fy = size[1] / 2 / np.tan(np.radians(vfov) / 2)
        self.yaw = None

    def render(self, pose):
        import numpy as np
        from panda3d.core import NodePath
        x, y, yaw = pose
        if self.yaw is None:
            self.yaw = yaw
        self.yaw += 0.3 * np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        pos = (x - self.back * c, y - self.back * s, self.up)
        hpr = (np.degrees(self.yaw) - 90.0, self.pitch, 0.0)
        img = self.cam.perceive(to_float=False, new_parent_node=self.env.engine.origin, position=pos, hpr=hpr)
        node = NodePath("demo_cam")
        node.reparentTo(self.env.engine.origin)
        node.setPos(*pos)
        node.setHpr(*hpr)
        mat = node.getMat(self.env.engine.origin)
        node.removeNode()
        inv = np.array([[mat.getCell(r, c_) for c_ in range(4)] for r in range(4)])   # row-vector convention
        return np.ascontiguousarray(img[..., ::-1]), np.linalg.inv(inv)


def project(world_to_cam, fx, fy, size, pts, z=0.12):
    """World xy(z) -> pixel uv, mask of points in front of the camera (panda3d: x right, y forward, z up)."""
    import numpy as np
    p = np.asarray(pts, float)
    p = np.c_[p[:, :2], p[:, 2] if p.shape[1] > 2 else np.full(len(p), z), np.ones(len(p))]
    q = p @ world_to_cam
    ok = q[:, 1] > 0.5
    d = np.maximum(q[:, 1], 1e-3)
    return np.stack([size[0] / 2 + fx * q[:, 0] / d, size[1] / 2 - fy * q[:, 2] / d], 1), ok


# ---------------------------------------------------------------- simulation (cached)
def simulate(data_dir, index, predictor, reactive, max_steps=None, sidewalk=True):
    """One closed-loop episode with the 3D camera on. Per sim step: raw 3D image (panel size), camera matrix, ego
    pose / speed, all object states, the plan in use (35 dense candidates, chosen index, predictions), distance."""
    import numpy as np
    from PIL import Image
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.scenario.utils import get_number_of_scenarios
    from sim import closed_loop as cl
    from sim.adapter import EGO_ID
    rw, rh = PANEL_W * CAM_RENDER, CAM_H * CAM_RENDER
    zoe = dict(vehicle_model="varying_dynamics", width=cl.EGO_SIZE_ZOE[0], length=cl.EGO_SIZE_ZOE[1])
    cfg = dict(use_render=False, num_scenarios=get_number_of_scenarios(str(data_dir)), data_directory=str(data_dir),
               horizon=10_000, log_level=50, crash_vehicle_done=False, crash_object_done=False,
               crash_human_done=False, vehicle_config=dict(image_source="rgb_camera", **zoe),
               agent_configs={"default_agent": dict(use_special_color=True, random_color=True, spawn_lane_index=None,
                                                    image_source="rgb_camera", **zoe)},      # teal ego in 3D
               image_observation=True, sensors={"rgb_camera": (RGBCamera, rw, rh)}, show_interface=False,
               show_logo=False, show_fps=False, show_sidewalk=sidewalk)
    env = ScenarioEnv(cfg)
    steps, state = [], {}
    try:
        env.reset(seed=index)

        def hook(info):
            if "cam" not in state:
                state["cam"] = _ChaseCam(env, (rw, rh))
                state["features"] = info["features"]
            img, w2c = state["cam"].render(info["pose"])
            buf = __import__("io").BytesIO()
            Image.fromarray(img).resize((PANEL_W, CAM_H), Image.LANCZOS).save(buf, "JPEG", quality=93)
            small = buf.getvalue()
            plan = info["plan"]
            steps.append(dict(step=info["step"], pose=info["pose"], speed=info["speed"], near_d=info["near_d"],
                              collided=info["collided"], img=small, w2c=w2c,
                              snapshot={k: v for k, v in info["snapshot"].items() if k != EGO_ID},
                              plan=None if plan is None else dict(step=plan["step"], dense=plan["dense"],
                                                                  k=plan["k"], paths=plan["paths"],
                                                                  probs=plan["probs"], ids=plan["ids"])))

        m, _ = cl.run_scene(env, predictor, reactive=reactive, step_hook=hook, max_steps=max_steps)
        cam = state["cam"]
        meta = dict(fx=cam.fx / CAM_RENDER, fy=cam.fy / CAM_RENDER, ego_size=list(map(float, (env.agent.WIDTH,
                                                                                          env.agent.LENGTH))))
    finally:
        env.close()
    return dict(metrics=m, steps=steps, meta=meta, features=state.get("features", {}))


# ---------------------------------------------------------------- per-frame state (interpolated)
def _lerp_pose(a, b, f):
    import numpy as np
    d = np.arctan2(np.sin(b[2] - a[2]), np.cos(b[2] - a[2]))
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * d)


def frame_state(steps, k, f):
    """State between sim steps k and k+1 (f in [0, 1)); the plan in use at step k."""
    a = steps[k]
    if f == 0 or k + 1 >= len(steps):
        return dict(a, t=(a["step"]) * 0.1, frac=0.0)
    b = steps[k + 1]
    snap = {}
    for i, s in a["snapshot"].items():
        if i in b["snapshot"]:
            x, y, yaw = _lerp_pose(s[:3], b["snapshot"][i][:3], f)
            snap[i] = (x, y, yaw) + tuple(s[3:])
        else:
            snap[i] = s
    return dict(a, pose=_lerp_pose(a["pose"], b["pose"], f), snapshot=snap, speed=a["speed"] + f * (b["speed"] - a["speed"]),
                near_d=a["near_d"] + f * (b["near_d"] - a["near_d"]), t=(a["step"] + f) * 0.1, frac=f)


def nearby(st, radius=30.0, n=6):
    import numpy as np
    from sim.closed_loop import AGENT_PREFIXES
    plan = st["plan"]
    if plan is None:
        return []
    x0, y0 = st["pose"][:2]
    out = []
    for a, i in enumerate(plan["ids"]):
        s = st["snapshot"].get(i)
        if s is None or not s[5].startswith(AGENT_PREFIXES):
            continue
        d = float(np.hypot(s[0] - x0, s[1] - y0)) - max(s[4], s[3]) / 2          # roughly to the box edge
        pr = np.asarray(plan["probs"][a], float)
        move = float(np.sum(pr * np.hypot(*(plan["paths"][a][:, -1] - np.asarray(s[:2])).T)))
        if d <= radius and move > 1.0:                                           # parked / standing: muted
            out.append((d, i))
    return [i for _, i in sorted(out)[:n]]


def layers(st, colors):
    """World-xy geometry for one frame: candidates, chosen plan, predictions (pts, times, prob, colour id)."""
    import numpy as np
    plan, pose = st["plan"], st["pose"]
    cands, chosen, preds = [], None, []
    if plan is None:
        return cands, chosen, preds, []
    el = (st["step"] - plan["step"]) * 0.1 + st["frac"] * 0.1         # seconds since the plan was made
    j = el / 0.1
    for p_, d_ in enumerate(plan["dense"]):
        t = np.arange(len(d_)) * 0.1
        keep = t > el
        pts = np.vstack([pose[:2], d_[keep, :2]]) if keep.any() else np.array([pose[:2], pose[:2]])
        far = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1]) > 2.1     # start at the front bumper
        pts = pts[far] if far.sum() >= 2 else pts[-1:].repeat(2, 0)
        if p_ == plan["k"]:
            chosen = pts
        else:
            cands.append(pts)
    near = nearby(st)
    for i in near:
        colors.setdefault(i, AGENT_RGB[len(colors) % len(AGENT_RGB)])
        a = plan["ids"].index(i)
        pr = np.asarray(plan["probs"][a], float)
        tt = np.arange(1, 13) * 0.5 - el                                  # time ahead of now
        keep = tt > 1e-6
        for m in np.argsort(pr)[::-1]:
            if pr[m] < 0.05:
                continue
            pts = np.vstack([st["snapshot"][i][:2], plan["paths"][a, m][keep]])
            preds.append(dict(id=i, p=float(pr[m]), pts=pts, t=np.r_[0.0, tt[keep]], top=m == np.argmax(pr)))
    return cands, chosen, preds, near




# ---------------------------------------------------------------- top-down panel (light)
def rounded_box(x, y, yaw, length, width, r=None):
    import numpy as np
    r = min(r if r is not None else 0.45, width / 2.2, length / 2.2)
    hl, hw = length / 2 - r, width / 2 - r
    pts = []
    for cx, cy, a0 in ((hl, hw, 0), (-hl, hw, 90), (-hl, -hw, 180), (hl, -hw, 270)):
        for a in np.radians(np.linspace(a0, a0 + 90, 6)):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    p = np.asarray(pts)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.c_[x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]]


def _rgb(c, a=1.0):
    return (c[0] / 255, c[1] / 255, c[2] / 255, a)


class TopDown:
    """Light top-down view following the ego, north up. Map drawn once; dynamic artists replaced per frame."""

    def __init__(self, features, size, half_w=36.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
        import numpy as np
        self.plt, self.size, self.half_w = plt, size, half_w
        self.half_h = half_w * size[1] / size[0]
        self.lw = size[0] / PANEL_W                                       # line widths scale with panel size
        self.fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100)
        self.fig.patch.set_facecolor(_rgb(PANEL_BG))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(_rgb(PANEL_BG))
        surf, cross, lines, edges = [], [], [], []
        for f in features.values():
            t, poly, pl = f["type"], f.get("polygon"), f.get("polyline")
            if t == "LANE_SURFACE_STREET" and poly is not None and len(poly) >= 3:
                surf.append(np.asarray(poly)[:, :2])
            elif t == "CROSSWALK" and poly is not None and len(poly) >= 3:
                cross.append(np.asarray(poly)[:, :2])
            elif t.startswith("ROAD_LINE") and (pl if pl is not None else poly) is not None:
                lines.append(np.asarray(pl if pl is not None else poly)[:, :2])
            elif t.startswith("ROAD_EDGE") and (pl if pl is not None else poly) is not None:
                edges.append(np.asarray(pl if pl is not None else poly)[:, :2])
        lw = self.lw
        self.ax.add_collection(PolyCollection(surf, facecolors=_rgb(ROAD), edgecolors="none", zorder=1))
        self.ax.add_collection(PolyCollection(cross, facecolors=_rgb(CROSS), edgecolors="none", zorder=1.5))
        self.ax.add_collection(LineCollection(lines, colors=[_rgb(LANE_LINE)], linewidths=0.6 * lw, zorder=2))
        self.ax.add_collection(LineCollection(edges, colors=[_rgb(EDGE_LINE)], linewidths=0.9 * lw, zorder=2))
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.dyn = []

    def frame(self, st, colors, lay, trail, ego_size):
        import numpy as np
        from matplotlib.collections import LineCollection
        for h in self.dyn:
            h.remove()
        self.dyn = []
        ax, lw = self.ax, self.lw
        x0, y0, yaw0 = st["pose"]
        cands, chosen, preds, near = lay
        add = lambda h: self.dyn.extend(h if isinstance(h, list) else [h])
        if cands:                                                          # candidate plans: very faint
            add(ax.add_collection(LineCollection(cands, colors=[(0, 0, 0, 0.07)], linewidths=0.8 * lw, zorder=3)))
        for pr in preds:                                                   # predictions: thin, fading
            pts, t = pr["pts"], pr["t"]
            tt = np.linspace(0, t[-1], max(2, int(t[-1] / 0.1) + 1))
            xy = np.c_[np.interp(tt, t, pts[:, 0]), np.interp(tt, t, pts[:, 1])]
            seg = np.stack([xy[:-1], xy[1:]], 1)
            a = (0.35 + 0.65 * pr["p"]) * (1 - 0.8 * tt[1:] / 6.0)
            add(ax.add_collection(LineCollection(seg, colors=[_rgb(colors[pr["id"]], float(ai)) for ai in a],
                                                 linewidths=(1.6 if pr["top"] else 1.1) * lw, zorder=4,
                                                 capstyle="round")))
        for i, s in st["snapshot"].items():                                # road users: small rounded boxes
            x, y, yaw, w_, l_, cat = s
            if abs(x - x0) > self.half_w + 10 or abs(y - y0) > self.half_h + 10:
                continue
            ped = cat.startswith("human.") or cat == "PEDESTRIAN"
            L, Wd = (max(l_, 0.7), max(w_, 0.7)) if ped else (max(l_, 1.0), max(w_, 0.8))
            col = colors[i] if i in near else OTHER_RGB
            add(ax.fill(*rounded_box(x, y, yaw, L, Wd, r=Wd / 2.2 if ped else 0.4).T, color=_rgb(col), lw=0,
                        zorder=5))
            if not ped:                                                    # heading tick
                c, s_ = np.cos(yaw), np.sin(yaw)
                add(ax.plot([x + c * L * 0.18, x + c * L * 0.46], [y + s_ * L * 0.18, y + s_ * L * 0.46],
                            color=(1, 1, 1, 0.9), lw=1.2 * lw, zorder=6, solid_capstyle="round"))
        if len(trail) > 1:
            tr = np.asarray(trail)
            add(ax.plot(tr[:, 0], tr[:, 1], color=_rgb(ACCENT, 0.3), lw=1.0 * lw, zorder=3))
        if chosen is not None:
            add(ax.plot(chosen[:, 0], chosen[:, 1], color=_rgb(ACCENT), lw=2.2 * lw, zorder=7, solid_capstyle="round"))
        we, le = ego_size
        add(ax.fill(*rounded_box(x0, y0, yaw0, le, we).T, color=_rgb(ACCENT), lw=0, zorder=8))
        c, s_ = np.cos(yaw0), np.sin(yaw0)
        add(ax.plot([x0 + c * le * 0.18, x0 + c * le * 0.46], [y0 + s_ * le * 0.18, y0 + s_ * le * 0.46],
                    color=(1, 1, 1), lw=1.4 * lw, zorder=9, solid_capstyle="round"))
        ax.set_xlim(x0 - self.half_w, x0 + self.half_w)
        ax.set_ylim(y0 - self.half_h, y0 + self.half_h)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        self.plt.close(self.fig)


# ---------------------------------------------------------------- 3D panel
def cam_panel(st, meta, lay):
    """The rendered 3D frame (no grading) + the chosen plan as a thin line on the ground (drawn at 2x)."""
    import io
    from PIL import Image, ImageDraw
    base = Image.open(io.BytesIO(st["img"])).convert("RGBA")
    chosen = lay[1]
    if chosen is not None and len(chosen) >= 2:
        s = 2
        ov = Image.new("RGBA", (base.size[0] * s, base.size[1] * s), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        uv, ok = project(st["w2c"], meta["fx"] * s, meta["fy"] * s, ov.size, chosen, z=0.08)
        for a in range(len(uv) - 1):
            if ok[a] and ok[a + 1]:
                dr.line([tuple(uv[a]), tuple(uv[a + 1])], fill=ACCENT + (235,), width=5)
        base = Image.alpha_composite(base, ov.resize(base.size, Image.LANCZOS))
    return base.convert("RGB")


# ---------------------------------------------------------------- distance strip (light)
def strip_panel(dists, t_now, t_end, d_max, width):
    import numpy as np
    from PIL import Image, ImageDraw
    s = 2
    w, h = width * s, STRIP_H * s
    im = Image.new("RGBA", (w, h), PANEL_BG + (255,))
    dr = ImageDraw.Draw(im)
    x0, x1, y0, y1 = 70 * s, w - 30 * s, 44 * s, h - 30 * s
    X = lambda t: x0 + (x1 - x0) * t / t_end
    Y = lambda d: y1 - (y1 - y0) * min(d, d_max) / d_max
    f_s, f_m = font(12 * s), font(14 * s, "Medium")
    for d in range(0, int(d_max) + 1, 5):
        dr.line([(x0, Y(d)), (x1, Y(d))], fill=(236, 236, 236), width=s)
        dr.text((x0 - 10 * s, Y(d)), f"{d} m", font=f_s, fill=MUTED, anchor="rm")
    for t in np.arange(0, t_end + 1e-6, 2.5):
        dr.text((X(t), y1 + 8 * s), f"{t:g} s", font=f_s, fill=MUTED, anchor="mt")
    for xx in np.arange(x0, x1, 12 * s):
        dr.line([(xx, Y(1.0)), (min(xx + 6 * s, x1), Y(1.0))], fill=(170, 170, 170), width=s)
    dr.text((x1 + 4 * s, Y(1.0)), "1 m", font=f_s, fill=MUTED, anchor="lm")
    for name in ("WM", "CV"):                                       # CV on top, a bit thicker
        d = np.asarray(dists[name], float)
        t = np.arange(1, len(d) + 1) * 0.1
        pts = [(X(a), Y(b)) for a, b in zip(t, d)]
        past = [p for p, a in zip(pts, t) if a <= t_now + 1e-9]
        dr.line(pts, fill=PLANNER_RGB[name] + (60,), width=2 * s, joint="curve")
        if len(past) > 1:
            dr.line(past, fill=PLANNER_RGB[name] + (255,), width=(4 if name == "CV" else 3) * s, joint="curve")
        if past:
            u, v = past[-1]
            dr.ellipse([u - 5 * s, v - 5 * s, u + 5 * s, v + 5 * s], fill=PLANNER_RGB[name] + (255,))
    dr.line([(X(t_now), y0 - 6 * s), (X(t_now), y1)], fill=(0, 0, 0, 110), width=s)
    dr.text((x0, 14 * s), "Distance from our car to the nearest road user", font=f_m, fill=TEXT, anchor="lt")
    xr = x1
    for name, lab in (("WM", "WM planner"), ("CV", "CV planner")):
        tw = dr.textlength(lab, font=f_s)
        dr.text((xr, 16 * s), lab, font=f_s, fill=TEXT, anchor="rt")
        dr.line([(xr - tw - 30 * s, 23 * s), (xr - tw - 8 * s, 23 * s)], fill=PLANNER_RGB[name], width=3 * s)
        xr -= tw + 54 * s
    return im.resize((width, STRIP_H), Image.LANCZOS).convert("RGB")


# ---------------------------------------------------------------- helpers
def framed_paste(canvas, img, xy, r=8):
    from PIL import Image, ImageDraw
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], r, fill=255)
    canvas.paste(img, xy, mask)
    ImageDraw.Draw(canvas).rounded_rectangle([xy[0], xy[1], xy[0] + img.size[0] - 1, xy[1] + img.size[1] - 1], r,
                                             outline=BORDER, width=1)


def pill(dr, xy, text, f, fg=TEXT, bg=(255, 255, 255, 215), pad=(8, 4), anchor="l"):
    x, y = xy
    tw = dr.textlength(text, font=f)
    a, b = f.getbbox("Ag")[1], f.getbbox("Ag")[3]
    if anchor == "r":
        x -= tw + 2 * pad[0]
    dr.rounded_rectangle([x, y, x + tw + 2 * pad[0], y + (b - a) + 2 * pad[1]], 6, fill=bg)
    dr.text((x + pad[0], y + pad[1] - a), text, font=f, fill=fg)


def assign_colors(runs):
    colors = {}
    n = min(len(r["steps"]) for r in runs.values())
    for k in range(n):
        for name in ("CV", "WM"):
            layers(dict(runs[name]["steps"][k], frac=0.0), colors)
    return colors


def legend(dr, x, y, f, items=("car", "cand", "pred")):
    labs = {"car": "our car + chosen plan", "cand": "35 candidate plans",
            "pred": "predicted futures (opacity = probability, fades with time)"}
    for kind in items:
        if kind == "car":
            dr.line([(x, y), (x + 22, y)], fill=ACCENT, width=3)
        elif kind == "cand":
            dr.line([(x, y), (x + 22, y)], fill=(0, 0, 0, 70), width=1)
        else:
            for q in range(3):
                dr.line([(x + 8 * q, y), (x + 8 * q + 6, y)], fill=AGENT_RGB[q] + (240 - 70 * q,), width=2)
        dr.text((x + 30, y), labs[kind], font=f, fill=MUTED, anchor="lm")
        x += 30 + dr.textlength(labs[kind], font=f) + 22
    return x


# ---------------------------------------------------------------- video frame
SUBTITLE = {"CV": "assumes everyone keeps speed and heading", "WM": "predictions from the learned world model"}


def compose(runs, tops, colors, k, f, t_end, d_max, title):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(canvas, "RGBA")
    f_title, f_sub, f_small, f_time = font(20, "SemiBold"), font(15), font(13), font(20, "Medium")
    t_now = 0.0
    for c, name in enumerate(("CV", "WM")):
        run = runs[name]
        kk = min(k, len(run["steps"]) - 1)
        st = frame_state(run["steps"], kk, f if kk == k else 0.0)
        t_now = st["t"]
        lay = layers(st, colors)
        x0 = MARGIN + c * (PANEL_W + GAP)
        dr.ellipse([x0 + 2, 22, x0 + 12, 32], fill=PLANNER_RGB[name])
        dr.text((x0 + 20, 27), f"{name} planner", font=f_title, fill=TEXT, anchor="lm")
        dr.text((x0 + 32 + dr.textlength(f"{name} planner", font=f_title), 28), SUBTITLE[name], font=f_sub,
                fill=MUTED, anchor="lm")
        framed_paste(canvas, cam_panel(run["steps"][kk], run["meta"], layers(dict(run["steps"][kk], frac=0.0), {})),
                     (x0, Y_CAM))
        trail = [s_["pose"][:2] for s_ in run["steps"][:kk + 1]] + [st["pose"][:2]]
        framed_paste(canvas, Image.fromarray(tops[name].frame(st, colors, lay, trail, run["meta"]["ego_size"])),
                     (x0, Y_TOP))
        closest = min(s_["near_d"] for s_ in run["steps"][:kk + 1])
        pill(dr, (x0 + 12, Y_TOP + 12), f"speed {st['speed']:.1f} m/s   ·   closest so far {closest:.1f} m", f_small)
        if st["collided"]:
            pill(dr, (x0 + PANEL_W - 12, Y_TOP + 12), "contact", f_small, fg=(255, 255, 255), bg=(200, 50, 50, 230),
                 anchor="r")
        px = 10 * PANEL_W / (2 * tops[name].half_w)                  # 10 m scale bar, top-right
        bx, by = x0 + PANEL_W - 24 - px, Y_TOP + 23
        dr.rounded_rectangle([bx - 52, by - 12, bx + px + 8, by + 10], 6, fill=(255, 255, 255, 215))
        dr.line([(bx, by), (bx + px, by)], fill=MUTED, width=2)
        dr.text((bx - 8, by), "10 m", font=f_small, fill=MUTED, anchor="rm")
    dr.text((W - MARGIN, 27), f"t = {t_now:4.1f} s", font=f_time, fill=TEXT, anchor="rm")
    dr.text((W - MARGIN - dr.textlength(f"t = {t_now:4.1f} s", font=f_time) - 16, 28), title, font=f_small,
            fill=MUTED, anchor="rm")
    lx, ly = MARGIN + 14, Y_TOP + TOP_H - 20
    tot = legend(ImageDraw.Draw(Image.new("RGBA", (10, 10))), 0, 0, f_small)
    dr.rounded_rectangle([lx - 8, ly - 12, lx + tot - 10, ly + 10], 6, fill=(255, 255, 255, 215))
    legend(dr, lx, ly, f_small)
    dists = {n: [s_["near_d"] for s_ in r["steps"]] for n, r in runs.items()}
    framed_paste(canvas, strip_panel(dists, t_now, t_end, d_max, W - 2 * MARGIN), (MARGIN, Y_STRIP))
    return canvas


def write_gif(frames, gif, gif_mb=10.0, n_scene_colors=104):
    """GIF with one global palette: exact theme colours + colours fitted to sample frames. Downscaled to <= gif_mb."""
    import numpy as np
    from PIL import Image
    fixed = [BG, PANEL_BG, TEXT, MUTED, BORDER, ROAD, CROSS, LANE_LINE, EDGE_LINE, ACCENT, OTHER_RGB,
             *PLANNER_RGB.values(), *AGENT_RGB]
    for size in ((960, 540), (896, 504), (832, 468), (768, 432), (704, 396), (640, 360)):
        small = [Image.fromarray(f_).resize(size, Image.LANCZOS) for f_ in frames]
        sample = np.concatenate([np.asarray(small[i]) for i in np.linspace(0, len(small) - 1, 8).astype(int)], 0)
        fitted = Image.fromarray(sample).quantize(colors=n_scene_colors, method=Image.MEDIANCUT).getpalette()
        pal = [c for rgb in fixed for c in rgb] + fitted[:3 * n_scene_colors]
        pal_img = Image.new("P", (1, 1))
        pal_img.putpalette(pal + [0] * (768 - len(pal)))
        g = [im.quantize(palette=pal_img, dither=Image.Dither.NONE) for im in small]
        g[0].save(gif, save_all=True, append_images=g[1:], duration=200, loop=0)
        if gif.stat().st_size / 1e6 <= gif_mb:
            break
    return size


def d_max_of(runs):
    import numpy as np
    return float(np.clip(max(np.max(np.minimum([s["near_d"] for s in r["steps"]], 50)) for r in runs.values()), 4, 15))


def render_video(runs, out, title, gif=False):
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image
    colors = assign_colors(runs)
    n = min(len(r["steps"]) for r in runs.values())
    tops = {name: TopDown(r["features"], (PANEL_W, TOP_H)) for name, r in runs.items()}
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        wr = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "20", "-preset", "slow", "-movflags", "+faststart"],
                                         macro_block_size=8)                # 1080 = 8 x 135: no padding
        wr.send(None)
        gif_frames, i = [], 0
        for k in range(n):
            for f in (0.0, 0.5):
                if k == n - 1 and f > 0:
                    continue
                img = np.asarray(compose(runs, tops, colors, k, f, n * 0.1, d_max_of(runs), title))
                wr.send(np.ascontiguousarray(img))
                if gif and i % 4 == 0:
                    gif_frames.append(img)
                i += 1
        for _ in range(2 * FPS):
            wr.send(np.ascontiguousarray(img))
        wr.close()
        info = dict(mp4=str(out), mp4_mb=out.stat().st_size / 1e6, frames=i)
        if gif:
            g = out.with_suffix(".gif")
            info["gif_size"] = list(write_gif(gif_frames + [gif_frames[-1]] * 10, g))
            info["gif_mb"] = g.stat().st_size / 1e6
        return info
    finally:
        for t in tops.values():
            t.close()


def render_clip(runs, out_mp4, out_gif, title, before_s=3.9, after_s=2.1, slow_s=0.5, freeze_s=1.0):
    """Short clip around the CV planner's closest moment: real time, then 0.5x slow motion within +-slow_s of the
    closest step, a freeze-frame of freeze_s with both closest distances in large type, then real time again."""
    import numpy as np
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    colors = assign_colors(runs)
    n = min(len(r["steps"]) for r in runs.values())
    kc = int(np.argmin([s["near_d"] for s in runs["CV"]["steps"]]))
    k0, k1, ks = max(0, kc - int(round(before_s * 10))), min(n - 1, kc + int(round(after_s * 10))), int(round(slow_s * 10))
    mins = {name: r["metrics"]["min_dist_m"] for name, r in runs.items()}
    tops = {name: TopDown(r["features"], (PANEL_W, TOP_H)) for name, r in runs.items()}
    big, mid = font(64, "SemiBold"), font(26, "Medium")

    def overlay(img, slow, freeze):
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        if slow:
            pill(dr, (W // 2, 922), "0.5x slow motion", mid, anchor="m")
        if freeze:
            text = f"CV {mins['CV']:.2f} m   vs   WM {mins['WM']:.2f} m"
            box = dr.textbbox((0, 0), text, font=big)
            bw, bh = box[2] - box[0] + 80, box[3] - box[1] + 110
            x, y = (W - bw) // 2, (H - bh) // 2
            dr.rounded_rectangle((x, y, x + bw, y + bh), 22, fill=(255, 255, 255, 235), outline=BORDER, width=2)
            dr.text((W // 2, y + 34), "closest distance to another road user", font=mid, fill=MUTED, anchor="mt")
            dr.text((W // 2, y + bh - 30), text, font=big, fill=TEXT, anchor="mb")
        return np.ascontiguousarray(np.asarray(im))

    try:
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        wr = imageio_ffmpeg.write_frames(str(out_mp4), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         output_params=["-crf", "22", "-preset", "slow", "-movflags", "+faststart"],
                                         macro_block_size=8)
        wr.send(None)
        gif_frames, i = [], 0
        for k in range(k0, k1 + 1):
            slow = abs(k - kc) < ks
            for f in ((0.0, 0.25, 0.5, 0.75) if slow else (0.0, 0.5)):
                if k == k1 and f > 0:
                    continue
                base = np.asarray(compose(runs, tops, colors, k, f, n * 0.1, d_max_of(runs), title))
                reps = [(False, 1)] + ([(True, int(freeze_s * FPS))] if (k == kc and f == 0.0) else [])
                for freeze, count in reps:
                    img = overlay(base, slow, freeze)
                    for _ in range(count):
                        wr.send(img)
                        if i % 4 == 0:
                            gif_frames.append(img)
                        i += 1
        wr.close()
        size = write_gif(gif_frames, out_gif)
        return dict(mp4=str(out_mp4), mp4_mb=out_mp4.stat().st_size / 1e6, frames=i, seconds=i / FPS,
                    closest_step=kc, window_steps=[k0, k1], gif=str(out_gif), gif_size=list(size),
                    gif_mb=out_gif.stat().st_size / 1e6)
    finally:
        for t in tops.values():
            t.close()


# ---------------------------------------------------------------- grid image
GRID_LABELS = {1: "clear win", 2: "typical", 3: "different situation"}


def render_grid(examples, all_runs, out):
    """One row per example: CV 3D | CV top-down | WM 3D | WM top-down, at the CV planner's closest moment."""
    import numpy as np
    from PIL import Image, ImageDraw
    cw = (W - 2 * MARGIN - 3 * 12) // 4                                # 465
    ch = 284
    head, row_t, foot = 66, 36, 40
    canvas = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(canvas, "RGBA")
    f_h, f_t, f_s, f_c = font(22, "SemiBold"), font(16, "SemiBold"), font(13), font(15, "Medium")
    dr.text((MARGIN, 30), "Three development scenes, CV planner vs WM planner, at the CV planner's closest moment",
            font=f_h, fill=TEXT, anchor="lm")
    legend(dr, W - MARGIN - 760, 30, f_s)
    cols = [("CV", "3D"), ("CV", "top"), ("WM", "3D"), ("WM", "top")]
    y = head
    for e, ex in enumerate(examples, 1):
        runs = all_runs[ex["scene"]]
        colors = assign_colors(runs)
        k = int(np.argmin([s["near_d"] for s in runs["CV"]["steps"]]))
        t = runs["CV"]["steps"][k]["step"] * 0.1
        m = {n: r["metrics"]["min_dist_m"] for n, r in runs.items()}
        dr.text((MARGIN, y + row_t / 2), f"{e} · {GRID_LABELS[e]}", font=f_t, fill=TEXT, anchor="lm")
        x_ = MARGIN + dr.textlength(f"{e} · {GRID_LABELS[e]}", font=f_t) + 16
        dr.text((x_, y + row_t / 2 + 1), f"{ex['scene']} (dev100)   ·   closest distance: CV {m['CV']:.2f} m, "
                f"WM {m['WM']:.2f} m   ·   t = {t:.1f} s", font=f_s, fill=MUTED, anchor="lm")
        y += row_t
        for c, (name, kind) in enumerate(cols):
            x0 = MARGIN + c * (cw + 12)
            run = runs[name]
            kk = min(k, len(run["steps"]) - 1)
            st = dict(run["steps"][kk], frac=0.0, t=run["steps"][kk]["step"] * 0.1)
            lay = layers(st, colors)
            if kind == "3D":
                full = cam_panel(run["steps"][kk], run["meta"], lay)           # 936 x 400 -> centre crop to cw:ch
                cw2 = int(full.size[1] * cw / ch)
                left = (full.size[0] - cw2) // 2
                img = full.crop((left, 0, left + cw2, full.size[1])).resize((cw, ch), Image.LANCZOS)
            else:
                td = TopDown(run["features"], (cw * 2, ch * 2), half_w=24.0)
                trail = [s_["pose"][:2] for s_ in run["steps"][:kk + 1]]
                img = Image.fromarray(td.frame(st, colors, lay, trail, run["meta"]["ego_size"])).resize((cw, ch),
                                                                                                    Image.LANCZOS)
                td.close()
            framed_paste(canvas, img, (x0, y))
            pill(dr, (x0 + 8, y + 8), f"     {name} planner · {'3D' if kind == '3D' else 'top-down'}", f_s)
            dr.ellipse([x0 + 17, y + 15, x0 + 25, y + 23], fill=PLANNER_RGB[name])
        y += ch + 14
    dr.text((MARGIN, H - foot / 2), "Picked by a fixed rule (post-hoc amended, see docs/demo_selection.md) among 66 eligible "
            "development scenes; WM keeps more distance in 25, CV in 8, equal in 33. Not test scenes.",
            font=f_s, fill=MUTED, anchor="lm")
    canvas.save(out)


# ---------------------------------------------------------------- main
def load_runs(data, index, reactive, checkpoint, resim=False, seconds=None):
    cache = PROJECT_ROOT / f"outputs/demo_2x2/cache/{CACHE_TAG}_{Path(data).name}_{index}_{reactive}" \
                           f"_{seconds or 'full'}.pkl"
    if cache.exists() and not resim:
        return pickle.loads(cache.read_bytes())
    from sim import closed_loop as cl
    runs = {}
    for name, pred in (("CV", cl.CVPredictor()), ("WM", cl.WorldModelPredictor(checkpoint))):
        runs[name] = simulate(Path(data), index, pred, reactive == "idm",
                              None if seconds is None else int(round(seconds * 10)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(runs))
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=("grid", "videos", "clip"))
    p.add_argument("--selection", type=Path, default=PROJECT_ROOT / "outputs/demo_2x2/examples_selection_amended.json")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints/v2_d_rerun/v2_seed42_best.pth"))
    p.add_argument("--reactive", choices=("off", "idm"), default="idm")
    p.add_argument("--resim", action="store_true", help="ignore the simulation cache")
    args = p.parse_args()
    import json
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    sel = json.loads(args.selection.read_text())
    examples = sel["examples"]
    if any("test150" in ex["data"] for ex in examples):
        raise SystemExit("the demo is for development scenes only")
    all_runs, report = {}, []
    for e, ex in enumerate(examples, 1):
        runs = load_runs(ex["data"], ex["index"], args.reactive, args.checkpoint, args.resim)
        all_runs[ex["scene"]] = runs
        ok = all(abs(runs[n]["metrics"]["min_dist_m"] - ex[k]) < 1e-6 for n, k in (("CV", "cv_min"), ("WM", "wm_min")))
        line = (f"example {e} {ex['scene']}: CV {runs['CV']['metrics']['min_dist_m']:.3f} m (sweep {ex['cv_min']:.3f}), "
                f"WM {runs['WM']['metrics']['min_dist_m']:.3f} m (sweep {ex['wm_min']:.3f}) -> "
                f"{'reproduced' if ok else 'MISMATCH'}")
        print(line, flush=True)
        report.append(dict(example=e, scene=ex["scene"], reproduced=ok,
                           **{f"{k}_{n}": r["metrics"][k] for n, r in runs.items()
                              for k in ("min_dist_m", "at_fault", "progress_ratio", "share_dist_below_1m")}))
    aux = PROJECT_ROOT / "outputs/demo_2x2/final"
    aux.mkdir(parents=True, exist_ok=True)
    if args.what == "clip":
        ex = examples[0]
        info = render_clip(all_runs[ex["scene"]], aux.parent / "example1_clip.mp4",
                           PROJECT_ROOT / "assets/example1_clip.gif", f"example 1 · {GRID_LABELS[1]} · {ex['scene']}")
        report[0].update(info)
        print(info, flush=True)
    elif args.what == "grid":
        render_grid(examples, all_runs, PROJECT_ROOT / "assets/examples_grid.png")
        print("saved assets/examples_grid.png")
    else:
        for e, ex in enumerate(examples, 1):
            info = render_video(all_runs[ex["scene"]], aux.parent / f"example{e}.mp4",
                                f"example {e} · {GRID_LABELS[e]} · {ex['scene']}", gif=e == 1)
            report[e - 1].update(info)
            print(info, flush=True)
    (aux / f"examples_{args.what}_info.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
