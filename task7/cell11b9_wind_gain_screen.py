"""CELL 11B.9 — stronger signed-wind gain screen with exact landing predicates.

Targeted follow-up to CELL 11B.8. The dominant failure signature was
inside+feet (angle was secondary), so this screen tests whether a somewhat
stronger signed-wind feed-forward term improves final pad geometry.

Models are evaluated on the SAME 48 fresh crosswind scenarios:
  v3b (original horizontal baseline)
  v3b_wind_p00192 (current wind candidate / comparison reference)
  four larger positive wind gains.

This is development screening only. No candidate is promoted and original
v2c/v3b/p00192 checkpoints are never overwritten.

Outputs:
  cell11b9_wind_gain_summary.csv
  cell11b9_wind_gain_episodes.csv
  cell11b9_wind_gain_paired.csv
  cell11b9_wind_gain_manifest.csv
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, export, FAMILIES
from torch_lander.env import SETTLE_SECONDS


SEED = 51739
N_EPISODES = 48
HORIZON = 800
FAMILY = "crosswind"
BASELINE = "v3b_wind_p00192"
ALPHAS = (0.0240, 0.0288, 0.0384, 0.0480)
WIND_SHARPNESS = 2.0
N_BOOTSTRAP = 5000


def _find_vertical_gate(model):
    w0 = model[0].weight.detach()
    mask = (
        (w0[:, 8].abs() > 50.0)
        & (w0[:, 9].abs() > 50.0)
        & ((w0[:, 8] * w0[:, 9]) < 0)
    )
    ids = torch.nonzero(mask, as_tuple=True)[0]
    if len(ids) != 1:
        raise ValueError(
            f"Expected exactly one v3b gravity/thrust gate, found {len(ids)}"
        )
    return int(ids.item())


def _make_candidate(base_model, alpha):
    model = deepcopy(base_model).cpu().eval()
    w0, b0 = model[0].weight, model[0].bias
    w2, w3 = model[2].weight, model[4].weight
    vertical_gate = _find_vertical_gate(model)

    turn_signature = torch.tensor(
        [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
        dtype=w3.dtype,
    )
    if not torch.allclose(
        w3[:, 1].detach().cpu(), turn_signature,
        atol=1e-3, rtol=0.0
    ):
        raise ValueError("Verified horizontal turn output channel has changed")

    influence = (w3.abs() @ w2.abs()).sum(dim=0)
    available = [
        j for j in range(2, w2.shape[1]) if j != vertical_gate
    ]
    wind_gate = min(available, key=lambda j: float(influence[j]))

    with torch.no_grad():
        w0[wind_gate].zero_()
        b0[wind_gate].zero_()
        w0[wind_gate, 12] = float(WIND_SHARPNESS)
        w2[:, wind_gate].zero_()
        w2[1, wind_gate] = float(alpha)

    return model, vertical_gate, wind_gate, float(influence[wind_gate])


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


def _evaluate(name, model_path, episodes):
    model = load(model_path).eval()
    env = Lander(episodes, score_mode="contest", horizon=HORIZON)
    rewards = torch.zeros(N_EPISODES, dtype=torch.float64)
    min_abs_offset_below8 = np.full(N_EPISODES, np.inf, dtype=np.float64)

    started = time.perf_counter()
    torch.set_num_threads(1)
    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            actions = model(obs).argmax(dim=1)
            active = (~env.done).detach().cpu().numpy().astype(bool)
            state_before = env.state.detach().cpu().numpy().astype(float)
            params_before = env.params.detach().cpu().numpy().astype(float)

            _, step_reward, _, _, _ = env.step(actions)
            rewards += step_reward.detach().cpu().double()

            for i in np.flatnonzero(active):
                if state_before[i, 1] <= 8.0:
                    dx = state_before[i, 0] - params_before[i, 6]
                    min_abs_offset_below8[i] = min(
                        min_abs_offset_below8[i], abs(float(dx))
                    )

    # Recompute exact geometry predicates from the terminal rigid-body state.
    vertices = env.rigid.vertices_world(env.q)[:, 6:]
    vertex_x = vertices[:, :, 0]
    vertex_y = vertices[:, :, 1]
    pad = env.params[:, 6:7]
    half_width = env.params[:, 7:8]
    inside = (
        (vertex_x >= pad - half_width)
        & (vertex_x <= pad + half_width)
    ).all(dim=1)

    feet = vertex_y.reshape(-1, 2, 4).amin(dim=2)
    feet_ok = (feet.abs() < 0.03).all(dim=1)
    angle = env.state[:, 4]
    angle_ok = angle.abs() <= 0.25
    on_pad = inside & feet_ok & angle_ok

    landed = env.landed.detach().cpu().numpy().astype(bool)
    hull = env.hull_contact.detach().cpu().numpy().astype(bool)
    timeout = env.truncated.detach().cpu().numpy().astype(bool)
    terminated = env.terminated.detach().cpu().numpy().astype(bool)
    settle = env.settle_time.detach().cpu().numpy().astype(float)
    state = env.state.detach().cpu().numpy().astype(float)
    params = env.params.detach().cpu().numpy().astype(float)
    inside_np = inside.detach().cpu().numpy().astype(bool)
    feet_ok_np = feet_ok.detach().cpu().numpy().astype(bool)
    angle_ok_np = angle_ok.detach().cpu().numpy().astype(bool)
    on_pad_np = on_pad.detach().cpu().numpy().astype(bool)
    feet_np = feet.detach().cpu().numpy().astype(float)
    rewards_np = rewards.detach().cpu().numpy().astype(float)
    min_offset_np = min_abs_offset_below8

    settled = settle >= SETTLE_SECONDS - 1e-10
    if not np.array_equal(
        landed[settled], on_pad_np[settled]
    ):
        raise AssertionError(
            f"{name}: exact on_pad recomputation disagrees with env.landed"
        )

    rows = []
    for i in range(N_EPISODES):
        status = _status(
            bool(landed[i]), bool(hull[i]), bool(timeout[i]),
            bool(terminated[i]), float(settle[i])
        )
        failed = []
        if not inside_np[i]:
            failed.append("inside")
        if not feet_ok_np[i]:
            failed.append("feet")
        if not angle_ok_np[i]:
            failed.append("angle")

        rows.append({
            "model": name,
            "family": FAMILY,
            "episode_id": i,
            "status": status,
            "return": float(rewards_np[i]),
            "landed": int(landed[i]),
            "hull_contact": int(hull[i]),
            "timeout": int(timeout[i]),
            "invalid_landing": int(status == "INVALID_LANDING"),
            "settled": int(settled[i]),
            "settle_time": float(settle[i]),
            "inside": int(inside_np[i]),
            "feet_ok": int(feet_ok_np[i]),
            "angle_ok": int(angle_ok_np[i]),
            "on_pad_recomputed": int(on_pad_np[i]),
            "failed_predicates": "+".join(failed) if failed else "none",
            "wind": float(params[i, 3]),
            "pad_half_width": float(params[i, 7]),
            "final_pad_offset": float(state[i, 0] - params[i, 6]),
            "final_abs_pad_offset": abs(float(state[i, 0] - params[i, 6])),
            "min_abs_pad_offset_below8": (
                float(min_offset_np[i])
                if np.isfinite(min_offset_np[i]) else np.nan
            ),
            "final_height": float(state[i, 1]),
            "final_vx": float(state[i, 2]),
            "final_vy": float(state[i, 3]),
            "final_theta_deg": float(np.degrees(state[i, 4])),
            "final_omega": float(state[i, 5]),
            "abs_theta_deg": abs(float(np.degrees(state[i, 4]))),
            "left_foot_min_y": float(feet_np[i, 0]),
            "right_foot_min_y": float(feet_np[i, 1]),
        })

    frame = pd.DataFrame(rows)
    print(
        f"{name:22s} | R={frame['return'].sum():9.2f} | "
        f"landed={frame['landed'].sum():2d}/{N_EPISODES} | "
        f"hull={frame['hull_contact'].sum():2d} | "
        f"invalid={frame['invalid_landing'].sum():2d} | "
        f"mean_offset={frame['final_abs_pad_offset'].mean():.3f} | "
        f"time={time.perf_counter()-started:.1f}s"
    )
    return frame


def run_cell11b9(models_dir, reports_dir):
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    if FAMILY not in FAMILIES:
        raise ValueError(f"{FAMILY!r} not found in FAMILIES={FAMILIES}")
    family_id = FAMILIES.index(FAMILY)
    episodes = sample(
        N_EPISODES, SEED, family=family_id, profile="medium"
    )

    v3b_path = models_dir / "weights_v3b.pt"
    incumbent_path = models_dir / "weights_v3b_wind_p00192.pt"
    for path in (v3b_path, incumbent_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required checkpoint: {path}")

    base_model = load(v3b_path).eval()
    model_paths = {
        "v3b": v3b_path,
        BASELINE: incumbent_path,
    }
    manifest_rows = []

    for alpha in ALPHAS:
        tag = f"{alpha:.4f}".replace(".", "")
        name = f"v3b_wind_p{tag}"
        candidate, vertical_gate, wind_gate, influence = _make_candidate(
            base_model, alpha
        )
        candidate_path = models_dir / f"weights_{name}.pt"
        export(candidate, candidate_path)
        model_paths[name] = candidate_path
        manifest_rows.append({
            "model": name,
            "alpha": alpha,
            "wind_sharpness": WIND_SHARPNESS,
            "vertical_gate_neuron": vertical_gate,
            "wind_gate_neuron": wind_gate,
            "replaced_path_influence": influence,
            "base_model": "v3b",
            "seed": SEED,
        })

    frames = [
        _evaluate(name, path, episodes)
        for name, path in model_paths.items()
    ]
    flights = pd.concat(frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_rows)

    expected = len(model_paths) * N_EPISODES
    if len(flights) != expected:
        raise AssertionError(f"Expected {expected} rows, got {len(flights)}")
    if flights.duplicated(["model", "family", "episode_id"]).any():
        raise AssertionError("Duplicate model/family/episode rows")
    if not np.isfinite(
        flights[["return", "final_pad_offset", "final_theta_deg"]].to_numpy()
    ).all():
        raise AssertionError("Non-finite result detected")

    summary = (
        flights.groupby("model")
        .agg(
            episodes=("episode_id", "count"),
            total_return=("return", "sum"),
            landed=("landed", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            timeouts=("timeout", "sum"),
            mean_abs_final_offset=("final_abs_pad_offset", "mean"),
            median_abs_final_offset=("final_abs_pad_offset", "median"),
            mean_min_offset_below8=("min_abs_pad_offset_below8", "mean"),
            invalid_fail_inside=("inside", lambda x: int(
                ((x == 0) & flights.loc[x.index, "invalid_landing"].eq(1)).sum()
            )),
            invalid_fail_feet=("feet_ok", lambda x: int(
                ((x == 0) & flights.loc[x.index, "invalid_landing"].eq(1)).sum()
            )),
            invalid_fail_angle=("angle_ok", lambda x: int(
                ((x == 0) & flights.loc[x.index, "invalid_landing"].eq(1)).sum()
            )),
        )
        .reset_index()
    )

    ref = (
        flights[flights["model"] == BASELINE]
        .set_index("episode_id")["return"].sort_index()
    )
    rng = np.random.default_rng(SEED + 2026)
    paired_rows = []
    for name in model_paths:
        if name == BASELINE:
            continue
        other = (
            flights[flights["model"] == name]
            .set_index("episode_id")["return"].sort_index()
        )
        delta = (other - ref).to_numpy(dtype=np.float64)
        boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for b in range(N_BOOTSTRAP):
            boot[b] = rng.choice(delta, size=len(delta), replace=True).mean()
        low, high = np.percentile(boot, [2.5, 97.5])
        paired_rows.append({
            "model": name,
            "reference": BASELINE,
            "episodes": len(delta),
            "wins": int((delta > 1e-8).sum()),
            "losses": int((delta < -1e-8).sum()),
            "ties": int((np.abs(delta) <= 1e-8).sum()),
            "mean_delta_return": float(delta.mean()),
            "bootstrap_ci_low": float(low),
            "bootstrap_ci_high": float(high),
            "landing_delta": int(
                flights.loc[flights["model"] == name, "landed"].sum()
                - flights.loc[flights["model"] == BASELINE, "landed"].sum()
            ),
            "hull_delta": int(
                flights.loc[flights["model"] == name, "hull_contact"].sum()
                - flights.loc[flights["model"] == BASELINE, "hull_contact"].sum()
            ),
        })

    paired = pd.DataFrame(paired_rows)
    outputs = {
        "summary": reports_dir / "cell11b9_wind_gain_summary.csv",
        "flights": reports_dir / "cell11b9_wind_gain_episodes.csv",
        "paired": reports_dir / "cell11b9_wind_gain_paired.csv",
        "manifest": reports_dir / "cell11b9_wind_gain_manifest.csv",
    }
    summary.to_csv(outputs["summary"], index=False)
    flights.to_csv(outputs["flights"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    manifest.to_csv(outputs["manifest"], index=False)

    print("\nCELL 11B.9 — STRONGER WIND-GAIN SCREEN")
    print(summary.round(3).to_string(index=False))
    print("\nPAIRED VS v3b_wind_p00192")
    print(paired.round(4).to_string(index=False))
    print("\nCANDIDATE MANIFEST")
    print(manifest.round(6).to_string(index=False))
    print("\nQC: PASS")
    print(
        f"Seed={SEED} | episodes/model={N_EPISODES} | "
        f"family={FAMILY} | horizon={HORIZON}"
    )
    print("Development screen only; no promotion or final-validation claim.")
    print("Original v2c, v3b and v3b_wind_p00192 checkpoints were not overwritten.")

    return summary, flights, paired, manifest, outputs
