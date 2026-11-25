"""CELL 11B.8 — replay crosswind and measure exact on_pad predicates.

Replays the same paired development scenarios as CELL 11B.4 (seed 271828,
40 crosswind episodes/model). It does not tune or modify any weights.

For each terminal episode it recomputes the exact predicates from Lander._step:
  inside: all eight leg vertices lie within pad_center +/- half_width
  feet_ok: abs(minimum y of each leg's four vertices) < 0.03
  angle_ok: abs(state[:, 4]) <= 0.25 rad
  on_pad: inside & feet_ok & angle_ok

The replay must match CELL 11B.4 terminal outcomes. This is diagnostic
reproduction, NOT independent validation and NOT a model promotion test.

Outputs:
  cell11b8_exact_landing_predicates.csv
  cell11b8_predicate_summary.csv
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES
from torch_lander.env import SETTLE_SECONDS


MODELS = {
    "v3b": "weights_v3b.pt",
    "v3b_wind_p00192": "weights_v3b_wind_p00192.pt",
}
FAMILY = "crosswind"
SEED = 271828
N_EPISODES = 40
HORIZON = 800


def _numpy(value):
    return value.detach().cpu().numpy()


def _status(landed, hull, timeout, terminated, settle):
    if landed:
        return "LANDED"
    if hull:
        return "HULL_CONTACT"
    if timeout:
        return "TIMEOUT"
    if terminated and settle >= SETTLE_SECONDS - 1e-10:
        return "INVALID_LANDING"
    if terminated:
        return "OUT_OF_BOUNDS"
    return "OTHER"


def _evaluate(name, model_path, models_dir):
    model = load(model_path).eval()
    family_id = FAMILIES.index(FAMILY)
    episodes = sample(
        N_EPISODES, SEED, family=family_id, profile="medium"
    )
    env = Lander(episodes, score_mode="contest", horizon=HORIZON)
    rewards = torch.zeros(N_EPISODES, dtype=torch.float64)
    started = time.perf_counter()

    torch.set_num_threads(1)
    with torch.no_grad():
        while not bool(env.done.all()):
            actions = model(env.observe()).argmax(dim=1)
            _, step_reward, _, _, _ = env.step(actions)
            rewards += step_reward.detach().cpu().double()

    # Recompute the exact source predicates on the terminal configuration.
    vertices = env.rigid.vertices_world(env.q)[:, 6:]
    vertex_x = vertices[:, :, 0]
    vertex_y = vertices[:, :, 1]
    pad = env.params[:, 6:7]
    half_width = env.params[:, 7:8]
    lower = pad - half_width
    upper = pad + half_width

    inside_each_vertex = (vertex_x >= lower) & (vertex_x <= upper)
    inside = inside_each_vertex.all(dim=1)

    feet = vertex_y.reshape(-1, 2, 4).amin(dim=2)
    feet_each_ok = feet.abs() < 0.03
    feet_ok = feet_each_ok.all(dim=1)

    state = env.state
    angle = state[:, 4]
    angle_ok = angle.abs() <= 0.25
    on_pad = inside & feet_ok & angle_ok

    landed = env.landed
    hull = env.hull_contact
    timeout = env.truncated
    terminated = env.terminated
    settle = env.settle_time
    settled = settle >= SETTLE_SECONDS - 1e-10

    arrays = {
        "landed": _numpy(landed).astype(bool),
        "hull": _numpy(hull).astype(bool),
        "timeout": _numpy(timeout).astype(bool),
        "terminated": _numpy(terminated).astype(bool),
        "settle": _numpy(settle).astype(float),
        "settled": _numpy(settled).astype(bool),
        "inside": _numpy(inside).astype(bool),
        "feet_ok": _numpy(feet_ok).astype(bool),
        "angle_ok": _numpy(angle_ok).astype(bool),
        "on_pad": _numpy(on_pad).astype(bool),
        "angle": _numpy(angle).astype(float),
        "feet": _numpy(feet).astype(float),
        "pad": _numpy(pad[:, 0]).astype(float),
        "half_width": _numpy(half_width[:, 0]).astype(float),
        "lower_margin": _numpy((vertex_x - lower).amin(dim=1)).astype(float),
        "upper_margin": _numpy((upper - vertex_x).amin(dim=1)).astype(float),
        "min_vertex_x": _numpy(vertex_x.amin(dim=1)).astype(float),
        "max_vertex_x": _numpy(vertex_x.amax(dim=1)).astype(float),
        "min_foot_y": _numpy(feet.amin(dim=1)).astype(float),
        "max_abs_foot_y": _numpy(feet.abs().amax(dim=1)).astype(float),
        "state": _numpy(state).astype(float),
        "reward": _numpy(rewards).astype(float),
    }

    rows = []
    for i in range(N_EPISODES):
        status = _status(
            arrays["landed"][i], arrays["hull"][i],
            arrays["timeout"][i], arrays["terminated"][i],
            arrays["settle"][i],
        )
        failed = []
        if not arrays["inside"][i]:
            failed.append("inside")
        if not arrays["feet_ok"][i]:
            failed.append("feet")
        if not arrays["angle_ok"][i]:
            failed.append("angle")

        rows.append({
            "model": name,
            "family": FAMILY,
            "episode_id": i,
            "status": status,
            "return": arrays["reward"][i],
            "settled": int(arrays["settled"][i]),
            "settle_time": arrays["settle"][i],
            "landed": int(arrays["landed"][i]),
            "inside": int(arrays["inside"][i]),
            "feet_ok": int(arrays["feet_ok"][i]),
            "angle_ok": int(arrays["angle_ok"][i]),
            "on_pad_recomputed": int(arrays["on_pad"][i]),
            "failed_predicates": "+".join(failed) if failed else "none",
            "abs_theta_deg": abs(np.degrees(arrays["angle"][i])),
            "pad_center": arrays["pad"][i],
            "pad_half_width": arrays["half_width"][i],
            "lower_geometry_margin": arrays["lower_margin"][i],
            "upper_geometry_margin": arrays["upper_margin"][i],
            "min_vertex_x": arrays["min_vertex_x"][i],
            "max_vertex_x": arrays["max_vertex_x"][i],
            "left_foot_min_y": arrays["feet"][i, 0],
            "right_foot_min_y": arrays["feet"][i, 1],
            "max_abs_foot_y": arrays["max_abs_foot_y"][i],
            "final_x": arrays["state"][i, 0],
            "final_y": arrays["state"][i, 1],
            "final_vx": arrays["state"][i, 2],
            "final_vy": arrays["state"][i, 3],
            "final_theta": arrays["state"][i, 4],
            "final_omega": arrays["state"][i, 5],
        })

    frame = pd.DataFrame(rows)

    # The recomputed rule must agree with the environment's landed flag
    # whenever a stable settle event occurred.
    stable_rows = frame["settled"].astype(bool)
    if not np.array_equal(
        frame.loc[stable_rows, "landed"].to_numpy(dtype=bool),
        frame.loc[stable_rows, "on_pad_recomputed"].to_numpy(dtype=bool),
    ):
        raise AssertionError(
            f"{name}: recomputed on_pad predicate does not match env.landed"
        )

    print(
        f"{name} | R={frame['return'].sum():.2f} | "
        f"landed={frame['landed'].sum()}/{N_EPISODES} | "
        f"invalid={(frame['status'] == 'INVALID_LANDING').sum()} | "
        f"hull={(frame['status'] == 'HULL_CONTACT').sum()} | "
        f"time={time.perf_counter()-started:.1f}s"
    )
    return frame


def run_cell11b8(models_dir, reports_dir):
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if FAMILY not in FAMILIES:
        raise ValueError(f"{FAMILY!r} not in FAMILIES={FAMILIES}")

    frames = []
    for name, filename in MODELS.items():
        path = models_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        frames.append(_evaluate(name, path, models_dir))

    flights = pd.concat(frames, ignore_index=True)
    expected = len(MODELS) * N_EPISODES
    if len(flights) != expected:
        raise AssertionError(f"Expected {expected} rows, got {len(flights)}")
    if flights.duplicated(["model", "family", "episode_id"]).any():
        raise AssertionError("Duplicate model/family/episode rows")

    invalid = flights.loc[
        flights["status"] == "INVALID_LANDING"
    ].copy()
    summary = (
        invalid.groupby("model")
        .agg(
            invalid_episodes=("episode_id", "count"),
            fail_inside=("inside", lambda x: int((x == 0).sum())),
            fail_feet=("feet_ok", lambda x: int((x == 0).sum())),
            fail_angle=("angle_ok", lambda x: int((x == 0).sum())),
            fail_multiple=("failed_predicates",
                           lambda x: int((x.str.count(r"\+") > 0).sum())),
            mean_abs_theta_deg=("abs_theta_deg", "mean"),
            median_abs_theta_deg=("abs_theta_deg", "median"),
            mean_abs_foot_y=("max_abs_foot_y", "mean"),
        )
        .reset_index()
    )
    # Add offset from pad centre (kept separate from leg-vertex geometry).
    invalid["abs_center_offset"] = (
        invalid["final_x"] - invalid["pad_center"]
    ).abs()
    offset = invalid.groupby("model")["abs_center_offset"].mean()
    summary["mean_abs_center_offset"] = summary["model"].map(offset)

    flights_path = reports_dir / "cell11b8_exact_landing_predicates.csv"
    summary_path = reports_dir / "cell11b8_predicate_summary.csv"
    flights.to_csv(flights_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nCELL 11B.8 — EXACT FAILURE PREDICATES (INVALID LANDINGS)")
    print(summary.round(4).to_string(index=False))
    print("\nFAILURE SIGNATURES")
    print(
        invalid.groupby(["model", "failed_predicates"])
        .size().rename("episodes").reset_index()
        .sort_values(["model", "episodes"], ascending=[True, False])
        .to_string(index=False)
    )
    print("\nINVALID-LANDING EPISODES")
    print(
        invalid[[
            "model", "episode_id", "failed_predicates",
            "abs_theta_deg", "lower_geometry_margin",
            "upper_geometry_margin", "left_foot_min_y",
            "right_foot_min_y",
        ]].round(4).to_string(index=False)
    )
    print("\nSaved:", flights_path)
    print("Saved:", summary_path)
    print("QC: PASS")
    print("Reproduction seed only; not independent validation.")
    print("No weights modified; no tuning or promotion.")

    return flights, summary, flights_path, summary_path
