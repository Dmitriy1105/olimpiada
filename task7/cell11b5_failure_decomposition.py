"""CELL 11B.5 — paired failure decomposition from CELL 11B.4 outputs.

This is a post-processing cell. It does NOT run the simulator, tune weights,
or overwrite checkpoints. It requires the five CSVs produced by CELL 11B.4.

Outputs:
  cell11b5_paired_transitions.csv
  cell11b5_transition_summary.csv
  cell11b5_terminal_metrics.csv
  cell11b5_closest_approach.csv
  cell11b5_action_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE = "v3b"
CANDIDATE = "v3b_wind_p00192"
FAMILIES = ("crosswind", "low_thrust")
REQUIRED_FLIGHT_COLUMNS = {
    "model", "family", "episode_id", "return", "status", "landed",
    "hull_contact", "invalid_landing", "final_abs_pad_offset",
    "final_vx", "final_vy", "final_theta", "final_omega",
    "final_y", "final_settle_time",
}
REQUIRED_TRACE_COLUMNS = {
    "model", "family", "episode_id", "step", "post_pad_offset",
    "post_y", "post_vx", "post_vy", "post_theta", "post_omega",
    "action", "throttle", "turn", "terminal_after_step",
}


def run_cell11b5(reports_dir):
    reports_dir = Path(reports_dir)
    inputs = {
        "flights": reports_dir / "cell11b4_flights.csv",
        "trace": reports_dir / "cell11b4_trajectory_trace.csv",
        "actions": reports_dir / "cell11b4_action_distribution.csv",
    }
    for label, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing CELL 11B.4 {label} file: {path}"
            )

    flights = pd.read_csv(inputs["flights"])
    trace = pd.read_csv(inputs["trace"])
    actions = pd.read_csv(inputs["actions"])

    missing = REQUIRED_FLIGHT_COLUMNS - set(flights.columns)
    if missing:
        raise ValueError(f"Flights CSV is missing columns: {sorted(missing)}")
    missing = REQUIRED_TRACE_COLUMNS - set(trace.columns)
    if missing:
        raise ValueError(f"Trace CSV is missing columns: {sorted(missing)}")

    expected_models = {BASELINE, CANDIDATE}
    assert set(flights["model"]) == expected_models
    assert set(flights["family"]) == set(FAMILIES)
    assert not flights.duplicated(
        ["model", "family", "episode_id"]
    ).any()

    counts = flights.groupby(["model", "family"]).size()
    if len(counts) != 4 or counts.nunique() != 1:
        raise AssertionError(
            "Expected equal episode counts for both models and families: "
            f"{counts.to_dict()}"
        )

    # Pair the exact same scenario IDs and compare terminal outcomes.
    status = flights.pivot(
        index=["family", "episode_id"],
        columns="model",
        values="status",
    ).reset_index()
    reward = flights.pivot(
        index=["family", "episode_id"],
        columns="model",
        values="return",
    ).reset_index()
    for model in expected_models:
        if model not in status.columns or model not in reward.columns:
            raise ValueError(f"Missing paired model column: {model}")

    paired = status.rename(columns={
        BASELINE: "status_v3b",
        CANDIDATE: "status_candidate",
    })
    paired["return_v3b"] = reward[BASELINE].to_numpy(dtype=float)
    paired["return_candidate"] = reward[CANDIDATE].to_numpy(dtype=float)
    paired["delta_return"] = (
        paired["return_candidate"] - paired["return_v3b"]
    )
    paired["transition"] = (
        paired["status_v3b"].astype(str)
        + " -> "
        + paired["status_candidate"].astype(str)
    )

    transition_summary = (
        paired.groupby(["family", "transition"])
        .agg(
            episodes=("episode_id", "count"),
            mean_delta_return=("delta_return", "mean"),
            median_delta_return=("delta_return", "median"),
            sum_delta_return=("delta_return", "sum"),
        )
        .reset_index()
        .sort_values(["family", "episodes"], ascending=[True, False])
    )

    # Terminal metrics are separated by outcome. Absolute angles are also
    # expressed in degrees for easier interpretation.
    terminal = flights.copy()
    terminal["abs_final_vx"] = terminal["final_vx"].abs()
    terminal["abs_final_vy"] = terminal["final_vy"].abs()
    terminal["abs_final_theta_deg"] = np.degrees(
        terminal["final_theta"].abs()
    )
    terminal["abs_final_omega"] = terminal["final_omega"].abs()

    terminal_metrics = (
        terminal.groupby(["family", "model", "status"])
        .agg(
            episodes=("episode_id", "count"),
            mean_return=("return", "mean"),
            mean_abs_pad_offset=("final_abs_pad_offset", "mean"),
            median_abs_pad_offset=("final_abs_pad_offset", "median"),
            mean_abs_vx=("abs_final_vx", "mean"),
            mean_abs_vy=("abs_final_vy", "mean"),
            mean_abs_theta_deg=("abs_final_theta_deg", "mean"),
            mean_abs_omega=("abs_final_omega", "mean"),
            mean_final_y=("final_y", "mean"),
            mean_settle_time=("final_settle_time", "mean"),
        )
        .reset_index()
    )

    # Locate the closest recorded post-step state to the pad centre during
    # the final <=8 height window. This is a sampled trajectory minimum,
    # not a continuous-time minimum.
    near = trace.loc[
        (trace["post_y"] <= 8.0) | (trace["pre_y"] <= 8.0)
        if "pre_y" in trace.columns
        else (trace["post_y"] <= 8.0)
    ].copy()
    if near.empty:
        raise ValueError(
            "No trajectory samples at or below height 8 were found"
        )

    near["abs_post_pad_offset"] = near["post_pad_offset"].abs()
    idx = near.groupby(
        ["model", "family", "episode_id"]
    )["abs_post_pad_offset"].idxmin()
    closest = near.loc[idx].copy()
    closest["abs_post_vx"] = closest["post_vx"].abs()
    closest["abs_post_vy"] = closest["post_vy"].abs()
    closest["abs_post_theta_deg"] = np.degrees(
        closest["post_theta"].abs()
    )
    closest = closest[[
        "model", "family", "episode_id", "step", "next_step",
        "wind", "pad_half_width", "abs_post_pad_offset", "post_y",
        "abs_post_vx", "abs_post_vy", "abs_post_theta_deg",
        "post_omega", "action", "throttle", "turn",
        "terminal_after_step",
    ]].rename(columns={
        "step": "closest_sample_step",
        "next_step": "closest_sample_next_step",
        "post_y": "height_at_closest_sample",
        "post_omega": "omega_at_closest_sample",
        "action": "action_at_closest_sample",
        "throttle": "throttle_at_closest_sample",
        "turn": "turn_at_closest_sample",
    })

    # Summarize action selection at family level, keeping all nine actions.
    action_summary = (
        actions.groupby(["family", "model", "throttle"])
        .agg(
            action_count=("count", "sum"),
            action_percent=("percent", "sum"),
        )
        .reset_index()
        .sort_values(["family", "model", "throttle"])
    )

    out = {
        "cell11b5_paired_transitions.csv": paired,
        "cell11b5_transition_summary.csv": transition_summary,
        "cell11b5_terminal_metrics.csv": terminal_metrics,
        "cell11b5_closest_approach.csv": closest,
        "cell11b5_action_summary.csv": action_summary,
    }
    for filename, frame in out.items():
        frame.to_csv(reports_dir / filename, index=False)

    print("\nCELL 11B.5 — PAIRED TERMINAL TRANSITIONS")
    print(transition_summary.round(4).to_string(index=False))
    print("\nTERMINAL METRICS BY STATUS")
    print(terminal_metrics.round(4).to_string(index=False))
    print("\nCLOSEST RECORDED APPROACH — SUMMARY")
    print(
        closest.groupby(["family", "model"])
        .agg(
            episodes=("episode_id", "count"),
            mean_abs_offset=("abs_post_pad_offset", "mean"),
            median_abs_offset=("abs_post_pad_offset", "median"),
            mean_abs_vx=("abs_post_vx", "mean"),
            mean_abs_vy=("abs_post_vy", "mean"),
            mean_abs_theta_deg=("abs_post_theta_deg", "mean"),
        )
        .round(4).to_string()
    )
    print("\nTHROTTLE DISTRIBUTION")
    print(action_summary.round(3).to_string(index=False))

    assert len(paired) == int(counts.iloc[0]) * len(FAMILIES)
    assert np.isfinite(paired["delta_return"]).all()
    print("\nQC: PASS")
    print("Post-processing only: no simulator run, no tuning, no weight changes.")
    print(f"Input episodes/model: {int(counts.iloc[0]) * len(FAMILIES)}")
    print(f"Paired episodes: {len(paired)}")

    return paired, transition_summary, terminal_metrics, closest, action_summary
