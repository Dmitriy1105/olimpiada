"""CELL 11B.12 — short local refinement around the validated p00384 wind gain.

Tests two nearby gains against the immutable p00384 champion:
  alpha=0.03456
  alpha=0.04224

Matched development scenarios:
  crosswind: 48 medium episodes
  approach:  24 medium episodes (regression guard)

This is a short screening experiment, not final validation. It never changes
v3b, p00384, or any existing candidate checkpoint. Existing output files are
not overwritten. No automatic promotion is performed.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, load, export, FAMILIES
from torch_lander.env import SETTLE_SECONDS


ALPHAS = (0.03456, 0.04224)
WIND_SHARPNESS = 2.0
HORIZON = 800
FAMILY_CONFIG = {
    "crosswind": {"seed": 842917, "episodes": 48},
    "approach": {"seed": 842918, "episodes": 24},
}
N_BOOTSTRAP = 5000

EXPECTED_V3B_SHA256 = (
    "31ae96a0347a69f04d09a83954b2d56a048c4b4ba74d15ac4e8d9d4addc1fe4c"
)
EXPECTED_P00384_SHA256 = (
    "ef5a6b8d9b2ba95712a3b0dc02551dd8ad4aec8cdd43d1cb4bcd865d17195c23"
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _scenario_signature(env):
    return {
        "state": env.state.detach().cpu().numpy().copy(),
        "params": env.params.detach().cpu().numpy().copy(),
        "family": env.family.detach().cpu().numpy().copy(),
    }


def _verify_signature(current, reference, label):
    for key in ("state", "params", "family"):
        if not np.array_equal(current[key], reference[key]):
            raise AssertionError(
                f"{label}: matched scenario mismatch in {key}"
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


def _evaluate(model_name, model_path, family, config, reference_signature=None):
    n = config["episodes"]
    seed = config["seed"]
    family_id = FAMILIES.index(family)

    episodes = sample(n, seed, family=family_id, profile="medium")
    env = Lander(episodes, score_mode="contest", horizon=HORIZON)
    signature = _scenario_signature(env)
    if reference_signature is not None:
        _verify_signature(signature, reference_signature, model_name)

    model = load(model_path).eval()
    rewards = torch.zeros(n, dtype=torch.float64)
    started = time.perf_counter()

    with torch.no_grad():
        while not bool(env.done.all()):
            obs = env.observe()
            actions = model(obs).argmax(dim=1)
            _, reward, _, _, _ = env.step(actions)
            rewards += reward.detach().cpu().double()

    if not bool(env.done.all()):
        raise AssertionError(f"{model_name}/{family}: unfinished episodes")

    # Recompute exact landing predicates and verify against env.landed.
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
    rewards_np = rewards.detach().cpu().numpy().astype(float)
    inside_np = inside.detach().cpu().numpy().astype(bool)
    feet_ok_np = feet_ok.detach().cpu().numpy().astype(bool)
    angle_ok_np = angle_ok.detach().cpu().numpy().astype(bool)

    settled = settle >= SETTLE_SECONDS - 1e-10
    on_pad_np = on_pad.detach().cpu().numpy().astype(bool)
    if not np.array_equal(landed[settled], on_pad_np[settled]):
        raise AssertionError(
            f"{model_name}/{family}: exact landing predicate mismatch"
        )

    rows = []
    for i in range(n):
        status = _status(
            bool(landed[i]), bool(hull[i]), bool(timeout[i]),
            bool(terminated[i]), bool(settled[i])
        )
        rows.append({
            "model": model_name,
            "family": family,
            "episode_id": i,
            "return": float(rewards_np[i]),
            "status": status,
            "landed": int(landed[i]),
            "hull_contact": int(hull[i]),
            "invalid_landing": int(status == "INVALID_LANDING"),
            "timeout": int(timeout[i]),
            "steps": int(steps[i]),
            "wind": float(params[i, 3]),
            "final_pad_offset": float(state[i, 0] - params[i, 6]),
            "final_abs_pad_offset": abs(float(state[i, 0] - params[i, 6])),
            "inside": int(inside_np[i]),
            "feet_ok": int(feet_ok_np[i]),
            "angle_ok": int(angle_ok_np[i]),
        })

    frame = pd.DataFrame(rows)
    elapsed = time.perf_counter() - started
    print(
        f"{model_name:18s} | {family:10s} | "
        f"R={frame['return'].sum():9.2f} | "
        f"landed={frame['landed'].sum():2d}/{n} | "
        f"hull={frame['hull_contact'].sum():2d} | "
        f"invalid={frame['invalid_landing'].sum():2d} | "
        f"time={elapsed:.1f}s"
    )
    return frame, signature


def _paired_table(flights):
    rows = []
    champion = "p00384"
    candidates = ("p003456", "p004224")
    rng = np.random.default_rng(842999)

    for family in FAMILY_CONFIG:
        ref = (
            flights[(flights["model"] == champion) & (flights["family"] == family)]
            .set_index("episode_id")["return"].sort_index()
        )
        for candidate in candidates:
            other = (
                flights[
                    (flights["model"] == candidate)
                    & (flights["family"] == family)
                ]
                .set_index("episode_id")["return"].sort_index()
            )
            if not np.array_equal(ref.index.to_numpy(), other.index.to_numpy()):
                raise AssertionError("Paired episode IDs do not match")

            delta = (other - ref).to_numpy(dtype=np.float64)
            boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
            for b in range(N_BOOTSTRAP):
                boot[b] = rng.choice(
                    delta, size=len(delta), replace=True
                ).mean()
            low, high = np.percentile(boot, [2.5, 97.5])

            cand_rows = flights[
                (flights["model"] == candidate)
                & (flights["family"] == family)
            ]
            ref_rows = flights[
                (flights["model"] == champion)
                & (flights["family"] == family)
            ]
            rows.append({
                "candidate": candidate,
                "reference": champion,
                "family": family,
                "episodes": len(delta),
                "wins": int((delta > 1e-8).sum()),
                "losses": int((delta < -1e-8).sum()),
                "ties": int((np.abs(delta) <= 1e-8).sum()),
                "mean_delta_return": float(delta.mean()),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "landing_delta": int(
                    cand_rows["landed"].sum() - ref_rows["landed"].sum()
                ),
                "hull_delta": int(
                    cand_rows["hull_contact"].sum() - ref_rows["hull_contact"].sum()
                ),
                "invalid_delta": int(
                    cand_rows["invalid_landing"].sum()
                    - ref_rows["invalid_landing"].sum()
                ),
            })
    return pd.DataFrame(rows)


def _write_csv_new(frame, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite previous experiment output: {path}"
        )
    frame.to_csv(path, index=False)


def run_cell11b12(models_dir, reports_dir):
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    v3b_path = models_dir / "weights_v3b.pt"
    champion_path = models_dir / "weights_v3b_wind_p00384.pt"
    for path in (v3b_path, champion_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required checkpoint missing: {path}")

    v3b_hash = _sha256(v3b_path)
    champion_hash = _sha256(champion_path)
    if v3b_hash != EXPECTED_V3B_SHA256:
        raise RuntimeError(
            f"v3b SHA-256 mismatch: expected {EXPECTED_V3B_SHA256}, got {v3b_hash}"
        )
    if champion_hash != EXPECTED_P00384_SHA256:
        raise RuntimeError(
            "p00384 SHA-256 mismatch; refusing to use a different champion"
        )

    base_model = load(v3b_path).eval()
    model_paths = {"p00384": champion_path}
    manifest_rows = [{
        "model": "p00384",
        "alpha": 0.0384,
        "checkpoint": champion_path.name,
        "sha256": champion_hash,
        "role": "immutable_champion",
    }]

    for alpha in ALPHAS:
        tag = f"{alpha:.5f}".replace(".", "")
        name = f"p{tag}"
        candidate_path = models_dir / f"weights_v3b_wind_{name}.pt"
        if candidate_path.exists():
            raise FileExistsError(
                f"Candidate path already exists; refusing overwrite: {candidate_path}"
            )

        candidate, vertical_gate, wind_gate, influence = _make_candidate(
            base_model, alpha
        )
        export(candidate, candidate_path)
        actual_hash = _sha256(candidate_path)
        model_paths[name] = candidate_path
        manifest_rows.append({
            "model": name,
            "alpha": float(alpha),
            "checkpoint": candidate_path.name,
            "sha256": actual_hash,
            "role": "development_candidate",
            "vertical_gate_neuron": vertical_gate,
            "wind_gate_neuron": wind_gate,
            "replaced_path_influence": influence,
            "base_model": "v3b",
        })

    episode_frames = []
    reference_signatures = {}
    for model_name, model_path in model_paths.items():
        for family, config in FAMILY_CONFIG.items():
            key = family
            frame, signature = _evaluate(
                model_name, model_path, family, config,
                reference_signatures.get(key)
            )
            if key not in reference_signatures:
                reference_signatures[key] = signature
            episode_frames.append(frame)

    flights = pd.concat(episode_frames, ignore_index=True)
    expected_rows = len(model_paths) * sum(
        config["episodes"] for config in FAMILY_CONFIG.values()
    )
    if len(flights) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} episode rows, got {len(flights)}"
        )
    if flights.duplicated(["model", "family", "episode_id"]).any():
        raise AssertionError("Duplicate model/family/episode rows")
    if not np.isfinite(flights["return"].to_numpy()).all():
        raise AssertionError("Non-finite rewards detected")

    summary = (
        flights.groupby(["family", "model"])
        .agg(
            episodes=("episode_id", "count"),
            total_return=("return", "sum"),
            mean_return=("return", "mean"),
            landed=("landed", "sum"),
            hull_contacts=("hull_contact", "sum"),
            invalid_landings=("invalid_landing", "sum"),
            timeouts=("timeout", "sum"),
            mean_abs_final_offset=("final_abs_pad_offset", "mean"),
        )
        .reset_index()
    )
    paired = _paired_table(flights)
    manifest = pd.DataFrame(manifest_rows)

    outputs = {
        "summary": reports_dir / "cell11b12_summary.csv",
        "flights": reports_dir / "cell11b12_flights.csv",
        "paired": reports_dir / "cell11b12_paired.csv",
        "manifest": reports_dir / "cell11b12_manifest.csv",
    }
    _write_csv_new(summary, outputs["summary"])
    _write_csv_new(flights, outputs["flights"])
    _write_csv_new(paired, outputs["paired"])
    _write_csv_new(manifest, outputs["manifest"])

    print("\nCELL 11B.12 — MICRO REFINEMENT SCREEN")
    print(summary.round(3).to_string(index=False))
    print("\nPAIRED COMPARISON AGAINST IMMUTABLE p00384")
    print(paired.round(4).to_string(index=False))
    print("\nCANDIDATE MANIFEST")
    print(manifest.to_string(index=False))
    print("\nQC: PASS")
    print("Matched scenarios: crosswind=48, approach=24 per model.")
    print("Exact landing predicates verified against env.landed.")
    print("p00384 and v3b source checkpoints were SHA-256 verified.")
    print("No source checkpoint or existing candidate was overwritten.")
    print("Development screen only; no automatic promotion.")
    return summary, paired, flights, manifest, outputs
