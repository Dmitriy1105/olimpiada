"""CELL 11B.6 — inspect the exact landing acceptance rule before tuning.

This is a fast source/data audit. It does not run a batch simulation, alter
model weights, or assume undocumented landing thresholds.

Inputs:
  - installed participant package torch_lander
  - reports/cell11b4_flights.csv (from CELL 11B.4)

Outputs:
  - cell11b6_landing_rule_source.txt
  - cell11b6_invalid_landing_profile.csv

Run in the participant package environment, after CELL 11B.4.
"""
from pathlib import Path
import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import torch

from torch_lander import Lander, sample, FAMILIES


TOKENS = (
    "landed", "on_pad", "settle_time", "pad_half_width", "pad_center",
    "foot", "leg", "hull_contact", "terminated", "settled",
    "theta", "outside", "crash",
)


def _tensor_shape(value):
    if torch.is_tensor(value):
        return tuple(value.shape)
    return None


def _source_methods(cls):
    rows = []
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        try:
            src = inspect.getsource(member)
        except (OSError, TypeError):
            continue
        low = src.lower()
        hits = [token for token in TOKENS if token in low]
        if hits:
            rows.append((name, hits, textwrap.dedent(src)))
    return rows


def run_cell11b6(reports_dir, output_dir=None, probe_seed=271828):
    reports_dir = Path(reports_dir)
    output_dir = Path(output_dir) if output_dir else reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    flights_path = reports_dir / "cell11b4_flights.csv"
    if not flights_path.exists():
        raise FileNotFoundError(
            f"Missing {flights_path}; run CELL 11B.4 first."
        )
    flights = pd.read_csv(flights_path)

    required = {
        "model", "family", "episode_id", "status", "return",
        "final_abs_pad_offset", "final_vx", "final_vy",
        "final_theta", "final_omega", "final_y", "final_settle_time",
    }
    missing = required - set(flights.columns)
    if missing:
        raise ValueError(f"CELL 11B.4 CSV missing columns: {sorted(missing)}")

    source_file = inspect.getsourcefile(Lander)
    if not source_file:
        raise RuntimeError("Could not locate installed Lander source file.")
    source_path = Path(source_file)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    module_source = source_path.read_text(encoding="utf-8")

    methods = _source_methods(Lander)
    if not methods:
        raise RuntimeError(
            "Could not inspect Lander methods. Check that torch_lander "
            "was installed from the participant source package."
        )

    # Keep complete methods that contain landing/settlement logic. This is
    # source-derived evidence, not a guessed reconstruction of the rules.
    relevant = []
    for name, hits, src in methods:
        if any(token in hits for token in (
            "landed", "on_pad", "settle_time", "pad_half_width",
            "foot", "leg", "hull_contact", "settled",
        )):
            relevant.append(
                f"\n===== Lander.{name} | matched: {', '.join(hits)} =====\n{src}"
            )

    report_lines = [
        "CELL 11B.6 — EXACT LANDING RULE SOURCE AUDIT",
        f"Lander source: {source_path}",
        f"Lander constructor: {inspect.signature(Lander)}",
        "",
        "===== MATCHING CLASS METHODS =====",
        *relevant,
        "",
        "===== MODULE SOURCE MATCH WINDOWS =====",
    ]

    lines = module_source.splitlines()
    matched = [
        i for i, line in enumerate(lines)
        if any(token in line.lower() for token in TOKENS)
    ]
    windows = []
    for i in matched:
        lo, hi = max(0, i - 3), min(len(lines), i + 5)
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], hi))
        else:
            windows.append((lo, hi))
    for lo, hi in windows:
        report_lines.append(
            f"\n--- {source_path.name}:{lo + 1}-{hi} ---"
        )
        report_lines.extend(
            f"{j + 1:04d}: {lines[j]}" for j in range(lo, hi)
        )

    # Probe public environment fields and tensor shapes without advancing
    # the simulator. No episode outcome or model weights are changed.
    if "crosswind" not in FAMILIES:
        raise ValueError(f"crosswind family missing from {FAMILIES}")
    episodes = sample(
        1, probe_seed, family=FAMILIES.index("crosswind"), profile="medium"
    )
    env = Lander(episodes, score_mode="contest", horizon=800)

    report_lines.extend([
        "",
        "===== LIVE ENVIRONMENT SHAPES (NO STEP EXECUTED) =====",
        f"env.state: {_tensor_shape(getattr(env, 'state', None))}",
        f"env.params: {_tensor_shape(getattr(env, 'params', None))}",
        f"env.observe(): {_tensor_shape(env.observe())}",
    ])
    for name in (
        "landed", "hull_contact", "settle_time", "terminated",
        "truncated", "done",
    ):
        value = getattr(env, name, None)
        report_lines.append(
            f"env.{name}: shape={_tensor_shape(value)} "
            f"type={type(value).__name__}"
        )

    report_path = output_dir / "cell11b6_landing_rule_source.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Summarize only observed INVALID_LANDING terminal states. Do not infer
    # which individual predicate failed until the source rule is reviewed.
    invalid = flights.loc[
        (flights["status"] == "INVALID_LANDING")
        & (flights["family"] == "crosswind")
    ].copy()
    invalid["abs_theta_deg"] = np.degrees(invalid["final_theta"].abs())
    invalid["abs_vx"] = invalid["final_vx"].abs()
    invalid["abs_vy"] = invalid["final_vy"].abs()
    invalid["abs_omega"] = invalid["final_omega"].abs()

    profile = (
        invalid.groupby("model")
        .agg(
            episodes=("episode_id", "count"),
            mean_abs_pad_offset=("final_abs_pad_offset", "mean"),
            median_abs_pad_offset=("final_abs_pad_offset", "median"),
            p90_abs_pad_offset=("final_abs_pad_offset",
                                lambda x: x.quantile(0.90)),
            mean_abs_theta_deg=("abs_theta_deg", "mean"),
            median_abs_theta_deg=("abs_theta_deg", "median"),
            p90_abs_theta_deg=("abs_theta_deg",
                               lambda x: x.quantile(0.90)),
            mean_abs_vx=("abs_vx", "mean"),
            mean_abs_vy=("abs_vy", "mean"),
            mean_abs_omega=("abs_omega", "mean"),
            mean_final_y=("final_y", "mean"),
            mean_settle_time=("final_settle_time", "mean"),
        )
        .reset_index()
    )
    profile_path = output_dir / "cell11b6_invalid_landing_profile.csv"
    profile.to_csv(profile_path, index=False)

    print("\nCELL 11B.6 — SOURCE-VERIFIED LANDING RULE")
    print(f"Source file: {source_path}")
    print(f"Source report: {report_path}")
    print("\nRelevant methods:")
    for name, hits, _ in methods:
        if any(token in hits for token in (
            "landed", "on_pad", "settle_time", "pad_half_width",
            "foot", "leg", "hull_contact", "settled",
        )):
            print(f"  Lander.{name}: {', '.join(hits)}")

    print("\nCROSSWIND INVALID-LANDING PROFILE")
    print(profile.round(4).to_string(index=False))
    print(f"\nSaved: {profile_path}")
    print("QC: PASS")
    print("No simulator steps executed; no model weights changed.")
    print("No landing threshold was guessed or modified.")

    return report_path, profile, profile_path
