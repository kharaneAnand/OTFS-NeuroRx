"""Sequential controller audit with lazy candidate inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.supervisor.reliability_bank import FrozenReceiverRunner, RECEIVER_NAMES, resolve_existing_path
from models.supervisor.reliability_controller import (
    ControllerThresholds,
    ReliabilityController,
)
from models.supervisor.simple_environment_detector import SimpleEnvironmentReference
from src.config.loader import load_config


def load_test_metrics(project_root: Path, config) -> dict[str, dict[int, dict[str, object]]]:
    """Load held-out labels only for post-decision auditing, never routing."""

    oamp_path = project_root / config.oamp_dl.output.directory / config.oamp_dl.output.per_sample_file
    oamp_frame = pd.read_csv(oamp_path)
    labels: dict[str, dict[int, dict[str, object]]] = {
        receiver: {} for receiver in RECEIVER_NAMES
    }
    for _, row in oamp_frame.iterrows():
        sample_index = int(row["sample_index"])
        labels["mmse"][sample_index] = {
            name: float(row[f"mmse_{name}"]) for name in ("ber", "ser", "nmse")
        }
        labels["oamp_dl"][sample_index] = {
            name: float(row[f"oamp_dl_{name}"]) for name in ("ber", "ser", "nmse")
        }

    gnn_path = project_root / config.gnn.output.mmse_directory / config.gnn.output.per_sample_file
    gnn_frame = pd.read_csv(gnn_path)
    gnn_frame = gnn_frame.loc[gnn_frame["initialization"] == "mmse"]
    for _, row in gnn_frame.iterrows():
        labels["original_gnn"][int(row["sample_index"])] = {
            name: float(row[name]) for name in ("ber", "ser", "nmse")
        }

    pi_path = resolve_existing_path(project_root, "experiments/pi_egnn/per_sample_results.csv")
    pi_frame = pd.read_csv(pi_path)
    for _, row in pi_frame.iterrows():
        labels["pi_egnn"][int(row["sample_index"])] = {
            name: float(row[f"pi_egnn_{name}"]) for name in ("ber", "ser", "nmse")
        }

    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--previous-log",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "reliability_controller" / "decision_log.csv",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    all_metadata = pd.read_csv(metadata_path)
    test_metadata = all_metadata.loc[
        all_metadata["split"].astype(str).str.lower() == "test"
    ].reset_index(drop=True)
    expected_test_count = int(config.dataset.expected_samples * config.split.test_ratio)
    if len(test_metadata) != expected_test_count:
        raise ValueError("Test split does not match the configured fixed split.")

    env_reference_path = (
        PROJECT_ROOT / "experiments" / "environment_detector_simple" / "reference.json"
    )
    with env_reference_path.open(encoding="utf-8") as file:
        environment_reference = SimpleEnvironmentReference.from_dict(json.load(file))
    reliability_reference_path = (
        PROJECT_ROOT / "experiments" / "reliability_detector" / "references.json"
    )
    with reliability_reference_path.open(encoding="utf-8") as file:
        reliability_payload = json.load(file)
    controller_config_path = PROJECT_ROOT / "configs" / "reliability_controller_v1.json"
    with controller_config_path.open(encoding="utf-8") as file:
        controller_config = json.load(file)

    runner = FrozenReceiverRunner(
        test_metadata,
        raw_dir,
        config,
        PROJECT_ROOT,
        environment_reference,
        device,
    )
    threshold_values = controller_config["thresholds"]
    thresholds = ControllerThresholds(
        low_z_threshold=float(threshold_values["low_confidence_z"]),
        switch_z_gap=float(threshold_values["switch_confidence_z_gap"]),
        consecutive_low_frames=int(threshold_values["consecutive_low_frames"]),
        minimum_frames_between_switches=int(
            threshold_values["minimum_frames_between_switches"]
        ),
    )
    absolute_gate = controller_config["absolute_performance_gate"]
    controller = ReliabilityController(
        reliability_payload["receivers"],
        thresholds=thresholds,
        initial_model=str(controller_config["active_model_initial"]),
        validated_ber={
            str(name): float(value)
            for name, value in absolute_gate["source_results"].items()
        },
        performance_margin=float(
            absolute_gate["noninferiority_margin_absolute_ber"]
        ),
        best_reference_receiver=str(absolute_gate["best_reference_receiver"]),
    )

    decision_rows = []
    live_frame_scores: dict[int, dict[str, float]] = {}
    for sample_index, metadata_row in test_metadata.iterrows():
        active_before = controller.active_model
        active_estimate, active_margin, environment_distance = runner.estimate(
            sample_index, active_before
        )
        del active_estimate

        def compute_candidates(index: int = sample_index) -> dict[str, float]:
            live_frame_scores[index] = runner.candidate_margins(index)
            return live_frame_scores[index]

        decision = controller.step(
            active_margin=active_margin,
            environment_distance=environment_distance,
            candidate_provider=compute_candidates,
        )
        candidate_margins = live_frame_scores.get(sample_index, {})
        decision_rows.append(
            {
                "frame_index": decision.frame_index,
                "sample_index": sample_index,
                "file": str(metadata_row["file"]),
                "snr_db": int(metadata_row["snr_db"]),
                "velocity_kmh": int(metadata_row["velocity_kmh"]),
                "active_model_before": decision.active_before,
                "active_confidence_margin": decision.active_margin,
                "active_confidence_z": decision.active_confidence_z,
                "environment_distance": decision.environment_distance,
                "low_z_threshold": decision.low_z_threshold,
                "switch_z_gap": decision.switch_z_gap,
                "low_streak": decision.low_streak,
                "decision": decision.decision,
                "active_model_after": decision.active_after,
                "candidate_confidence_margins": json.dumps(
                    candidate_margins, sort_keys=True
                ),
                "candidate_confidence_z_scores": json.dumps(
                    decision.candidate_confidence_z_scores, sort_keys=True
                ),
                "candidate_validated_ber": json.dumps(
                    decision.candidate_validated_ber, sort_keys=True
                ),
                "candidate_ber_ceiling": decision.candidate_ber_ceiling,
                "candidate_gate_results": json.dumps(
                    decision.candidate_gate_results, sort_keys=True
                ),
                "recommended_adaptation_model": decision.recommended_adaptation_model,
                "reason": decision.reason,
            }
        )

    decisions = pd.DataFrame(decision_rows)
    labels = load_test_metrics(PROJECT_ROOT, config)
    label_rows = []
    for _, row in test_metadata.iterrows():
        index = int(row.name)
        for receiver in RECEIVER_NAMES:
            metrics = labels[receiver].get(index)
            if metrics is None:
                raise KeyError(f"Missing offline {receiver} metrics for sample index {index}.")
            label_rows.append(
                {
                    "sample_index": index,
                    "snr_db": int(row["snr_db"]),
                    "velocity_kmh": int(row["velocity_kmh"]),
                    "receiver": receiver,
                    **metrics,
                }
            )
    label_frame = pd.DataFrame(label_rows)
    if len(label_frame) != len(test_metadata) * len(RECEIVER_NAMES):
        raise ValueError("Offline per-sample metric files do not cover all test samples/receivers.")

    best_by_frame = label_frame.loc[
        label_frame.groupby("sample_index")["ber"].idxmin()
    ][["sample_index", "receiver", "ber"]].rename(
        columns={"receiver": "best_ber_receiver", "ber": "best_ber"}
    )
    selected = decisions[["sample_index", "active_model_after"]].merge(
        label_frame,
        left_on=["sample_index", "active_model_after"],
        right_on=["sample_index", "receiver"],
        how="left",
        validate="one_to_one",
    ).merge(best_by_frame, on="sample_index", how="left", validate="one_to_one")
    selected["selected_is_best_ber"] = (
        selected["active_model_after"] == selected["best_ber_receiver"]
    )

    decisions["active_ber"] = selected["ber"].to_numpy()
    decisions["active_ser"] = selected["ser"].to_numpy()
    decisions["active_nmse"] = selected["nmse"].to_numpy()
    decisions["best_ber_receiver"] = selected["best_ber_receiver"].to_numpy()
    decisions["selected_is_best_ber"] = selected["selected_is_best_ber"].to_numpy()

    always_oamp = label_frame.loc[label_frame["receiver"] == "oamp_dl"]
    result = {
        "test_samples": len(test_metadata),
        "thresholds": {
            "low_confidence": f"active z-score < {thresholds.low_z_threshold:.3f} using the active model's own train reference mean/std",
            "clearly_better": f"candidate z-score >= active z-score + {thresholds.switch_z_gap:.3f}, with each model standardized by its own train reference mean/std",
            "consecutive_low_frames": thresholds.consecutive_low_frames,
            "minimum_frames_between_switches": thresholds.minimum_frames_between_switches,
            "threshold_source": "training confidence references only; no test tuning",
        },
        "absolute_performance_gate": {
            "metric": absolute_gate["metric"],
            "noninferiority_margin_absolute_ber": controller.performance_margin,
            "best_reference_receiver": controller.best_reference_receiver,
            "candidate_must_be_within_active_ber_plus_margin": True,
            "candidate_must_be_within_best_ber_plus_margin": True,
            "source_results": controller.validated_ber,
            "source_note": absolute_gate["source_note"],
        },
        "live_decision_inputs": ["active receiver confidence margin", "candidate margins only after two low frames", "active model state"],
        "environment_distance_logged_but_not_used": True,
        "retraining_performed": False,
        "receiver_inference_counts": runner.receiver_inference_counts,
        "candidate_inference_calls": sum(runner.receiver_inference_counts.values()) - len(test_metadata),
        "previous_raw_margin_policy_comparison": None,
        "controller_match_rate_to_lowest_frame_ber": float(selected["selected_is_best_ber"].mean()),
        "controller_result": {
            metric: float(selected[metric].mean()) for metric in ("ber", "ser", "nmse")
        },
        "always_oamp_dl_result": {
            metric: float(always_oamp[metric].mean()) for metric in ("ber", "ser", "nmse")
        },
        "decision_counts": decisions["decision"].value_counts().to_dict(),
        "switch_destinations": decisions.loc[
            decisions["decision"] == "SWITCH", "active_model_after"
        ].value_counts().to_dict(),
        "adapt_recommendations": decisions.loc[
            decisions["decision"] == "ADAPT", "recommended_adaptation_model"
        ].value_counts().to_dict(),
    }

    if args.previous_log.is_file():
        previous = pd.read_csv(args.previous_log)
        compare_columns = ["sample_index", "decision", "active_model_before", "active_model_after"]
        if set(compare_columns).issubset(previous.columns):
            previous = previous[compare_columns].sort_values("sample_index").reset_index(drop=True)
            current = decisions[compare_columns].sort_values("sample_index").reset_index(drop=True)
            matched = previous.merge(
                current,
                on="sample_index",
                suffixes=("_previous_raw", "_current_z"),
                how="outer",
                indicator=True,
            )
            outcome_columns = ("decision", "active_model_before", "active_model_after")
            identical_mask = matched["_merge"].eq("both")
            for column in outcome_columns:
                identical_mask &= matched[f"{column}_previous_raw"].eq(
                    matched[f"{column}_current_z"]
                )
            changed = matched.loc[~identical_mask]
            result["previous_raw_margin_policy_comparison"] = {
                "identical_all_frames": bool(changed.empty),
                "changed_frame_count": int(len(changed)),
                "changed_sample_indices": [
                    int(value)
                    for value in changed["sample_index"].dropna().tolist()
                ],
                "comparison_note": "Expected to differ because candidate comparison now uses per-receiver z-scores instead of raw margins.",
            }

    output_directory = PROJECT_ROOT / "experiments" / "reliability_controller"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "evaluation_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=2)
    decisions.to_csv(output_directory / "decision_log.csv", index=False)
    selected.to_csv(output_directory / "selected_frame_results.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
