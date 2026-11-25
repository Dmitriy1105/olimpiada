"""CELL 11B.10 — final balanced six-family validation.

Compares the original v3b, the previous signed-wind candidate p00192, and
the CELL 11B.9 winner p00384 on the same fresh 240-episode medium batch:
40 episodes in each of the six simulator families.

This is a locked validation run: no tuning, candidate generation, or weight
changes. It also recomputes the exact landing predicates from torch_lander.env
and verifies them against env.landed for settled episodes.

Outputs:
  cell11b10_flights.csv
  cell11b10_summary.csv
  cell11b10_family_summary.csv
  cell11b10_paired.csv
  cell11b10_predicate_summary.csv
  cell11b10_manifest.csv
"""
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, FAMILIES
from torch_lander.env import SETTLE_SECONDS


SEED = 681204
N_EPISODES = 240
HORIZON = 800
N_BOOTSTRAP = 10000

MODEL_FILES = {
    "v3b": "weights_v3b.pt",
    "v3b_wind_p00192": "weights_v3b_wind_p00192.pt",
    "v3b_wind_p00384": "weights_v3b_wind_p00384.pt",
}
PAIRWISE = (
    ("v3b_wind_p00192", "v3b"),
    ("v3b_wind_p00384", "v3b"),
    ("v3b_wind_p00384", "v3b_wind_p00192"),
)


def _initial_signature(env):
    return {
        "state": env.state.detach().cpu().numpy().copy(),
        "params": env.params.detach().cpu().numpy().copy(),
        "family": env.family.detach().cpu().numpy().copy(),
    }


def _verify_same_scenarios(signature, reference, model_name):
    for key in ("state", "params", "family"):
        if not np.array_equal(signature[key], reference[key]):
            raise AssertionError(
                f"{model_name}: scenario mismatch in initial {key}; "
                "paired comparison is invalid"
            )


def _status(landed, hull, timeout, terminated, settled):
    if landed:
        return "LANDED"
    if hull:
        return "HULL_CONTACT"
    if timeout:
        return "TIMEOUT"
    if terminated and settled:
        return "INVALID_LANDING"
    if terminated:
        return "OUT_OF_BOUNDS"
    return "OTHER"


def _evaluate(model_name, model_path, reference_signature=None):
    episodes = sample(N_EPISODES, SEED, profile="medium")
    env = Lander(episodes, score_mode="contest", horizon=HORIZON)
    signature = _initial_signature(env)
    if reference_signature is not None:
        _verify_same_scenarios(signature, reference_signature, model_name)

    model = load(model_path).eval()
    rewards = torch.zeros(N_EPISODES, dtype=torch.float64)
    started = time.perf_counter()

    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            actions = model(obs).argmax(dim=1)
            _, reward, _, _, _ = env.step(actions)
            rewards += reward.detach().cpu().double()

    if not bool(env.done.all()):
        raise AssertionError(f"{model_name}: unfinished episodes")
    if not torch.isfinite(rewards).all():
        raise AssertionError(f"{model_name}: non-finite rewards")

    # Recompute the exact on_pad rule from Lander._step.
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
    angle_ok = env.state[:, 4].abs() <= 0.25
    on_pad = inside & feet_ok & angle_ok

    landed = env.landed.detach().cpu().numpy().astype(bool)
    hull = env.hull_contact.detach().cpu().numpy().astype(bool)
    timeout = env.truncated.detach().cpu().numpy().astype(bool)
    terminated = env.terminated.detach().cpu().numpy().astype(bool)
    settle = env.settle_time.detach().cpu().numpy().astype(float)
    steps = env.steps.detach().cpu().numpy().astype(int)
    state = env.state.detach().cpu().numpy().astype(float)
    params = env.params.detach().cpu().numpy().astype(float)
    family_ids = env.family.detach().cpu().numpy().astype(int)
    rewards_np = rewards.detach().cpu().numpy().astype(float)

    inside_np = inside.detach().cpu().numpy().astype(bool)
    feet_ok_np = feet_ok.detach().cpu().numpy().astype(bool)
    angle_ok_np = angle_ok.detach().cpu().numpy().astype(bool)
    on_pad_np = on_pad.detach().cpu().numpy().astype(bool)
    feet_np = feet.detach().cpu().numpy().astype(float)

    settled = settle >= SETTLE_SECONDS - 1e-10
    if not np.array_equal(landed[settled], on_pad_np[settled]):
        raise AssertionError(
            f"{model_name}: recomputed landing predicate disagrees with env.landed"
        )

    rows = []
    for i in range(N_EPISODES):
        status = _status(
            bool(landed[i]), bool(hull[i]), bool(timeout[i]),
            bool(terminated[i]), bool(settled[i])
        )
        failed = []
        if not inside_np[i]:
            failed.append("inside")
        if not feet_ok_np[i]:
            failed.append("feet")
        if not angle_ok_np[i]:
            failed.append("angle")

        pad_offset = float(state[i, 0] - params[i, 6])
        rows.append({
            "model": model_name,
            "episode_id": i,
            "family": FAMILIES[int(family_ids[i])],
            "return": float(rewards_np[i]),
            "status": status,
            "landed": int(landed[i]),
            "hull_contact": int(hull[i]),
            "timeout": int(timeout[i]),
            "invalid_landing": int(status == "INVALID_LANDING"),
            "out_of_bounds": int(status == "OUT_OF_BOUNDS"),
            "settled": int(settled[i]),
            "steps": int(steps[i]),
            "wind": float(params[i, 3]),
            "pad_half_width": float(params[i, 7]),
            "final_pad_offset": pad_offset,
            "final_abs_pad_offset": abs(pad_offset),
            "final_height": float(state[i, 1]),
            "final_vx": float(state[i, 2]),
            "final_vy": float(state[i, 3]),
            "final_theta_deg": float(np.degrees(state[i, 4])),
            "final_omega": float(state[i, 5]),
            "inside": int(inside_np[i]),
            "feet_ok": int(feet_ok_np[i]),
            "angle_ok": int(angle_ok_np[i]),
            "on_pad_recomputed": int(on_pad_np[i]),
            "failed_predicates": "+".join(failed) if failed else "none",
            "left_foot_min_y": float(feet_np[i, 0]),
            "right_foot_min_y": float(feet_np[i, 1]),
            "final_settle_time": float(settle[i]),
        })

    frame = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    print(
        f"{model_name:22s} | R={frame['return'].sum():10.2f} | "
        f"landed={frame['landed'].sum():3d}/{N_EPISODES} "
        f"({100.0*frame['landed'].mean():5.1f}%) | "
        f"hull={frame['hull_contact'].sum():3d} | "
        f"invalid={frame['invalid_landing'].sum():3d} | "
        f"time={elapsed:.1f}s"
    )
    return frame, signature


def _paired_comparisons(flights):
    rng = np.random.default_rng(SEED + 20261127)
    output = []

    for candidate, reference in PAIRWISE:
        left = flights[flights["model"] == candidate].set_index("episode_id")
        right = flights[flights["model"] == reference].set_index("episode_id")
        left = left.sort_index()
        right = right.sort_index()

        if not np.array_equal(left["family"].to_numpy(), right["family"].to_numpy()):
            raise AssertionError(f"Family mismatch: {candidate} vs {reference}")

        delta = (
            left["return"].to_numpy(dtype=np.float64)
            - right["return"].to_numpy(dtype=np.float64)
        )
        delta_landed = (
            left["landed"].to_numpy(dtype=np.int64)
            - right["landed"].to_numpy(dtype=np.int64)
        )
        delta_hull = (
            left["hull_contact"].to_numpy(dtype=np.int64)
            - right["hull_contact"].to_numpy(dtype=np.int64)
        )
        delta_invalid = (
            left["invalid_landing"].to_numpy(dtype=np.int64)
            - right["invalid_landing"].to_numpy(dtype=np.int64)
        )

        family_values = [
            delta[left["family"].to_numpy() == family]
            for family in FAMILIES
        ]
        if any(len(values) != N_EPISODES // len(FAMILIES) for values in family_values):
            raise AssertionError("Paired bootstrap requires balanced family counts")

        boot_family = np.empty(N_BOOTSTRAP, dtype=np.float64)
        boot_total = np.empty(N_BOOTSTRAP, dtype=np.float64)
        for b in range(N_BOOTSTRAP):
            family_means = [
                rng.choice(values, size=len(values), replace=True).mean()
                for values in family_values
            ]
            boot_family[b] = float(np.mean(family_means))
            boot_total[b] = float(
                rng.choice(delta, size=len(delta), replace=True).sum()
            )

        family_ci = np.percentile(boot_family, [2.5, 97.5])
        total_ci = np.percentile(boot_total, [2.5, 97.5])

        output.append({
            "candidate": candidate,
            "reference": reference,
            "episodes": len(delta),
            "wins": int((delta > 1e-8).sum()),
            "losses": int((delta < -1e-8).sum()),
            "ties": int((np.abs(delta) <= 1e-8).sum()),
            "mean_delta_return": float(delta.mean()),
            "family_weighted_ci_low": float(family_ci[0]),
            "family_weighted_ci_high": float(family_ci[1]),
            "total_delta_return": float(delta.sum()),
            "total_delta_ci_low": float(total_ci[0]),
            "total_delta_ci_high": float(total_ci[1]),
            "landing_delta": int(delta_landed.sum()),
            "hull_delta": int(delta_hull.sum()),
            "invalid_delta": int(delta_invalid.sum()),
        })

    return pd.DataFrame(output)


def run_cell11b10(models_dir, reports_dir):
    if N_EPISODES % len(FAMILIES) != 0:
        raise ValueError("N_EPISODES must divide evenly across all families")
    if len(FAMILIES) != 6:
        raise AssertionError(f"Expected six scenario families, got {FAMILIES}")

    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    paths = {
        name: models_dir / filename
        for name, filename in MODEL_FILES.items()
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint: {path}. Run CELL 11B.9 first."
            )

    manifest_rows = []
    for name, path in paths.items():
        manifest_rows.append({
            "model": name,
            "checkpoint": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seed": SEED,
            "episodes": N_EPISODES,
            "horizon": HORIZON,
        })

    frames = []
    reference_signature = None
    for name, path in paths.items():
        frame, signature = _evaluate(name, path, reference_signature)
        if reference_signature is None:
            reference_signature = signature
        frames.append(frame)

    flights = pd.concat(frames, ignore_index=True)
    if len(flights) != len(paths) * N_EPISODES:
        raise AssertionError("Unexpected number of episode rows")
    if flights.duplicated(["model", "episode_id"]).any():
        raise AssertionError("Duplicate model/episode rows")
    if not np.isfinite(
        flights[["return", "final_pad_offset", "final_theta_deg"]].to_numpy()
    ).all():
        raise AssertionError("Non-finite result detected")

    expected_per_family = N_EPISODES // len(FAMILIES)
    for name in paths:
        subset = flights[flights["model"] == name]
        counts = subset.groupby("family")["episode_id"].count()
        if len(counts) != len(FAMILIES) or not counts.eq(expected_per_family).all():
            raise AssertionError(
                f"{name}: unbalanced family counts {counts.to_dict()}"
            )

    summary = (
        flights.groupby("model")
        .agg(
            episodes=("episode_id", "count"),
            total_return=("return", "sum"),
            mean_return=("return", "mean"),
            landed=("landed", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            timeouts=("timeout", "sum"),
            out_of_bounds=("out_of_bounds", "sum"),
            mean_abs_final_offset=("final_abs_pad_offset", "mean"),
            median_abs_final_offset=("final_abs_pad_offset", "median"),
            mean_steps=("steps", "mean"),
        )
    )
    summary["success_pct"] = 100.0 * summary["landed"] / summary["episodes"]

    family_summary = (
        flights.groupby(["family", "model"])
        .agg(
            episodes=("episode_id", "count"),
            return_sum=("return", "sum"),
            mean_return=("return", "mean"),
            landed=("landed", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            timeouts=("timeout", "sum"),
            mean_abs_final_offset=("final_abs_pad_offset", "mean"),
        )
        .reset_index()
    )

    invalid = flights[flights["invalid_landing"] == 1]
    predicate_summary = (
        invalid.groupby(["family", "model"])
        .agg(
            invalid_episodes=("episode_id", "count"),
            fail_inside=("inside", lambda x: int((x == 0).sum())),
            fail_feet=("feet_ok", lambda x: int((x == 0).sum())),
            fail_angle=("angle_ok", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )

    paired = _paired_comparisons(flights)
    manifest = pd.DataFrame(manifest_rows)

    outputs = {
        "flights": reports_dir / "cell11b10_flights.csv",
        "summary": reports_dir / "cell11b10_summary.csv",
        "family": reports_dir / "cell11b10_family_summary.csv",
        "paired": reports_dir / "cell11b10_paired.csv",
        "predicates": reports_dir / "cell11b10_predicate_summary.csv",
        "manifest": reports_dir / "cell11b10_manifest.csv",
    }
    flights.to_csv(outputs["flights"], index=False)
    summary.to_csv(outputs["summary"])
    family_summary.to_csv(outputs["family"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    predicate_summary.to_csv(outputs["predicates"], index=False)
    manifest.to_csv(outputs["manifest"], index=False)

    print("\nEPISODES BY FAMILY")
    print(
        flights[flights["model"] == "v3b"]
        .groupby("family")["episode_id"].count().to_string()
    )
    print("\nCELL 11B.10 — OVERALL SUMMARY")
    print(summary.round(3).to_string())
    print("\nFAMILY PERFORMANCE")
    print(family_summary.round(3).to_string(index=False))
    print("\nPAIRED COMPARISONS")
    print(paired.round(4).to_string(index=False))
    print("\nINVALID-LANDING PREDICATES")
    print(predicate_summary.to_string(index=False))
    print("\nCHECKPOINT MANIFEST")
    print(manifest.to_string(index=False))
    print("\nQC: PASS")
    print(
        f"Seed={SEED} | episodes/model={N_EPISODES} | "
        f"episodes/family={expected_per_family} | horizon={HORIZON}"
    )
    print("All models evaluated on exactly matched initial scenarios.")
    print("Exact on_pad predicates matched env.landed for all settled episodes.")
    print("No tuning, candidate generation, promotion, or weight overwrite.")
    print("CELL 11B.10 is a validation gate, not an automatic promotion.")
    return flights, summary, family_summary, paired, predicate_summary, manifest, outputs
