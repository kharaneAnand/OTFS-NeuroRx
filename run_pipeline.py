"""Run one OTFS frame or a dataset split through the live receiver controller."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.supervisor.reliability_bank import FrozenReceiverRunner, RECEIVER_NAMES
from models.supervisor.reliability_controller import (
    ControllerThresholds,
    ReliabilityController,
)
from models.supervisor.reliability_detector import (
    ReceiverReliabilityReference,
    reliability_feature_vector,
)
from models.supervisor.simple_environment_detector import SimpleEnvironmentReference
from src.config.loader import Config, load_config

REFERENCE_DIR = PROJECT_ROOT / "experiments"
ENVIRONMENT_REFERENCE = REFERENCE_DIR / "environment_detector_simple" / "reference.json"
RELIABILITY_REFERENCE = REFERENCE_DIR / "reliability_detector" / "references.json"
CONTROLLER_CONFIG = PROJECT_ROOT / "configs" / "reliability_controller_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OTFS receiver inference, reliability scoring, and controller decisions."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--input", type=Path, help="Path to one existing NPZ frame.")
    modes.add_argument("--generate", action="store_true", help="Generate one fresh frame.")
    modes.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset root containing raw/, metadata/, and processed/ directories.",
    )
    modes.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Generate an isolated full dataset and run its selected split.",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_v1.yaml")
    parser.add_argument("--snr", type=float, help="Frame SNR in dB (required for --input/--generate).")
    parser.add_argument("--velocity", type=float, help="Frame velocity in km/h (required for --input/--generate).")
    parser.add_argument("--seed", type=int, help="Required RNG seed for generated data; must differ from the configured V1 seed.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, help="New run directory. Existing directories are never overwritten.")
    parser.add_argument("--state-in", type=Path, help="Restore controller state from a prior single/batch run.")
    parser.add_argument("--state-out", type=Path, help="Write controller state for the next invocation.")
    parser.add_argument("--calibration-root", type=Path, help="Dataset root whose validation split may be used for OAMP-DL ADAPT.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def make_run_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = PROJECT_ROOT / "experiments" / "pipeline_runs" / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
    else:
        requested = resolve_path(requested)
    if requested.exists():
        raise FileExistsError(f"Run output directory already exists; refusing to overwrite: {requested}")
    requested.mkdir(parents=True)
    return requested


def clone_config(config: Config, *, dataset_root: Path, seed: int) -> Config:
    values = config.to_dict()
    values["dataset"]["root"] = str(dataset_root.resolve())
    values["reproducibility"]["seed"] = int(seed)
    values["split"]["random_seed"] = int(seed)
    return Config(values)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_references() -> tuple[SimpleEnvironmentReference, dict[str, Any], dict[str, Any]]:
    environment = SimpleEnvironmentReference.from_dict(load_json(ENVIRONMENT_REFERENCE))
    reliability = load_json(RELIABILITY_REFERENCE)
    controller_config = load_json(CONTROLLER_CONFIG)
    return environment, reliability, controller_config


def build_controller(
    reliability_payload: dict[str, Any],
    controller_config: dict[str, Any],
) -> ReliabilityController:
    thresholds = controller_config["thresholds"]
    gate = controller_config["absolute_performance_gate"]
    return ReliabilityController(
        reliability_payload["receivers"],
        thresholds=ControllerThresholds(
            low_z_threshold=float(thresholds["low_confidence_z"]),
            switch_z_gap=float(thresholds["switch_confidence_z_gap"]),
            consecutive_low_frames=int(thresholds["consecutive_low_frames"]),
            minimum_frames_between_switches=int(thresholds["minimum_frames_between_switches"]),
        ),
        initial_model=str(controller_config["active_model_initial"]),
        validated_ber={name: float(value) for name, value in gate["source_results"].items()},
        performance_margin=float(gate["noninferiority_margin_absolute_ber"]),
        best_reference_receiver=str(gate["best_reference_receiver"]),
    )


def restore_controller_state(controller: ReliabilityController, state_path: Path | None) -> None:
    if state_path is None:
        return
    state = load_json(resolve_path(state_path))
    if int(state.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported controller state schema.")
    active_model = str(state["active_model"])
    if active_model not in RECEIVER_NAMES:
        raise ValueError(f"Unknown active model in state: {active_model}")
    controller.active_model = active_model
    controller.low_streak = int(state["low_streak"])
    controller.frames_since_switch = int(state["frames_since_switch"])
    controller.frame_index = int(state["frame_index"])
    if min(controller.low_streak, controller.frames_since_switch, controller.frame_index) < 0:
        raise ValueError("Controller state counters must be non-negative.")


def save_controller_state(controller: ReliabilityController, state_path: Path | None) -> None:
    if state_path is None:
        return
    path = resolve_path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "active_model": controller.active_model,
        "low_streak": controller.low_streak,
        "frames_since_switch": controller.frames_since_switch,
        "frame_index": controller.frame_index,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def controller_decision_dict(decision: Any) -> dict[str, Any]:
    return {
        "action": decision.decision,
        "reason": decision.reason,
        "active_before": decision.active_before,
        "active_after": decision.active_after,
        "active_confidence_margin": decision.active_margin,
        "active_confidence_z": decision.active_confidence_z,
        "low_z_threshold": decision.low_z_threshold,
        "switch_z_gap": decision.switch_z_gap,
        "low_streak": decision.low_streak,
        "candidate_confidence_margins": decision.candidate_margins,
        "candidate_confidence_z_scores": decision.candidate_confidence_z_scores,
        "candidate_validated_ber": decision.candidate_validated_ber,
        "candidate_ber_ceiling": decision.candidate_ber_ceiling,
        "candidate_gate_results": decision.candidate_gate_results,
        "adaptation_recommendation": decision.recommended_adaptation_model,
    }


def estimate_metrics(estimate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    estimated_bits = np.stack((estimate.real >= 0, estimate.imag >= 0), axis=-1)
    target_bits = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    bit_errors = estimated_bits != target_bits
    symbol_errors = np.any(bit_errors, axis=-1)
    denominator = float(np.sum(np.abs(target) ** 2))
    return {
        "ber": float(np.count_nonzero(bit_errors) / bit_errors.size),
        "ser": float(np.count_nonzero(symbol_errors) / target.size),
        "nmse": float(np.sum(np.abs(estimate - target) ** 2) / max(denominator, np.finfo(float).eps)),
    }


def metric_result(estimate: np.ndarray, target: np.ndarray | None) -> dict[str, Any]:
    if target is None:
        return {
            "status": "unavailable_no_ground_truth",
            "ber": None,
            "ser": None,
            "nmse": None,
        }
    return {
        "status": "reference_only_requires_ground_truth",
        **estimate_metrics(estimate, target),
    }


def hard_decisions(estimate: np.ndarray) -> list[list[int]]:
    """Map each QPSK estimate to bits by sign: real >= 0 -> 1; imag >= 0 -> 1."""
    bits = np.stack((estimate.real >= 0, estimate.imag >= 0), axis=-1)
    return bits.astype(np.int8).tolist()


def frame_record(
    *,
    sample: dict[str, Any],
    receiver_used: str,
    active_before: str,
    active_after: str,
    environment_distance: float,
    reliability_distance: float,
    decision: dict[str, Any],
    estimate: np.ndarray,
    target: np.ndarray | None,
    inference_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "sample": sample,
        "receiver": {
            "used_for_this_frame": receiver_used,
            "active_before": active_before,
            "active_after": active_after,
        },
        "environment": {
            "distance": environment_distance,
            "used_for_controller_action": False,
        },
        "reliability": {
            "distance": reliability_distance,
            "confidence_source": "receiver_qpsk_margin standardized by that receiver's training reference",
        },
        "decision": decision,
        "estimate": {
            "symbol_count": int(estimate.size),
            "complex_symbols_re_im": [
                [float(value.real), float(value.imag)] for value in estimate
            ],
            "hard_decision_rule": {
                "bit_0": "1 if real part >= 0, otherwise 0",
                "bit_1": "1 if imaginary part >= 0, otherwise 0",
            },
            "qpsk_bit_pairs": hard_decisions(estimate),
        },
        "metrics": metric_result(estimate, target),
        "inference_counts_after_decision": dict(inference_counts),
    }


def read_sample_labels(raw_dir: Path, filename: str) -> np.ndarray | None:
    with np.load(raw_dir / filename, allow_pickle=False) as sample:
        if "tx_dd" not in sample.files:
            return None
        return np.asarray(sample["tx_dd"])


def make_frame_metadata(filename: str, snr: float, velocity: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"file": filename, "snr_db": float(snr), "velocity_kmh": float(velocity), "split": "test"}]
    )


def generate_one_frame(config: Config, run_dir: Path, snr: float, velocity: float, seed: int) -> tuple[Path, Path, pd.DataFrame, Config]:
    from experiments.dataset_generation.generate_otfs_dataset import generate_dataset

    root = run_dir / "generated_frame"
    values = config.to_dict()
    values["dataset"]["root"] = str(root.resolve())
    values["dataset"]["frames_per_condition"] = 1
    values["dataset"]["expected_samples"] = 1
    values["channel"]["snr_db"] = [snr]
    values["channel"]["velocity_kmh"] = [velocity]
    values["reproducibility"]["seed"] = seed
    values["split"]["random_seed"] = seed
    one_config = Config(values)
    generate_dataset(one_config)
    raw_dir = root / one_config.dataset.raw_dir
    metadata = pd.read_csv(root / one_config.dataset.metadata_dir / "dataset_v1_metadata.csv")
    metadata["split"] = "test"
    return raw_dir, root, metadata, one_config


def prepare_generated_dataset(config: Config, run_dir: Path, seed: int) -> tuple[Path, Path, pd.DataFrame, Config]:
    from experiments.dataset_generation.generate_otfs_dataset import generate_dataset
    from experiments.dataset_preprocessing.preprocess_otfs_dataset import create_split, validate_split

    root = run_dir / "dataset"
    values = config.to_dict()
    values["dataset"]["root"] = str(root.resolve())
    values["reproducibility"]["seed"] = seed
    values["split"]["random_seed"] = seed
    dataset_config = Config(values)
    generate_dataset(dataset_config)
    metadata_path = root / dataset_config.dataset.metadata_dir / "dataset_v1_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    split_metadata = create_split(metadata, dataset_config)
    validate_split(split_metadata, dataset_config)
    processed_dir = root / dataset_config.dataset.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_path = processed_dir / dataset_config.dataset.split_metadata_file
    split_metadata.to_csv(split_path, index=False)
    return root / dataset_config.dataset.raw_dir, root, split_metadata, dataset_config


def dataset_paths(root: Path, config: Config) -> tuple[Path, Path, pd.DataFrame]:
    root = resolve_path(root)
    raw_dir = root / config.dataset.raw_dir
    split_path = root / config.dataset.processed_dir / config.dataset.split_metadata_file
    if not raw_dir.is_dir() or not split_path.is_file():
        raise FileNotFoundError(
            f"Dataset root must contain {config.dataset.raw_dir}/ and "
            f"{config.dataset.processed_dir}/{config.dataset.split_metadata_file}: {root}"
        )
    return raw_dir, root, pd.read_csv(split_path)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_runner(
    metadata: pd.DataFrame,
    raw_dir: Path,
    config: Config,
    environment: SimpleEnvironmentReference,
    device: torch.device,
    run_dir: Path,
    cache_metadata: pd.DataFrame,
) -> FrozenReceiverRunner:
    return FrozenReceiverRunner(
        metadata,
        raw_dir,
        config,
        PROJECT_ROOT,
        environment,
        device,
        cache_metadata=cache_metadata,
        cache_path=run_dir / "pi_egnn_feature_cache.npz",
    )


def process_frame(
    sample_index: int,
    metadata_row: pd.Series,
    runner: FrozenReceiverRunner,
    controller: ReliabilityController,
    reliability_payload: dict[str, Any],
    raw_dir: Path,
    *,
    score_all_receivers: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    active_before = controller.active_model
    active_estimate, active_margin, environment_distance = runner.estimate(sample_index, active_before)
    reliability_reference = ReceiverReliabilityReference.from_dict(
        reliability_payload["receivers"][active_before]
    )
    active_reliability = reliability_reference.score(
        reliability_feature_vector(active_estimate, environment_distance)
    )
    live_candidate_margins: dict[int, dict[str, float]] = {}

    def candidate_provider() -> dict[str, float]:
        live_candidate_margins[sample_index] = runner.candidate_margins(sample_index)
        return live_candidate_margins[sample_index]

    decision = controller.step(
        active_margin=active_margin,
        environment_distance=environment_distance,
        candidate_provider=candidate_provider,
    )
    receiver_estimates: dict[str, np.ndarray] = {active_before: active_estimate}
    metrics_by_receiver: dict[str, dict[str, float]] = {}
    target = read_sample_labels(raw_dir, str(metadata_row["file"]))

    if score_all_receivers:
        for receiver in RECEIVER_NAMES:
            receiver_estimates[receiver] = runner.estimate(sample_index, receiver)[0]
        if target is not None:
            metrics_by_receiver = {
                name: estimate_metrics(estimate, target)
                for name, estimate in receiver_estimates.items()
            }

    sample_info = {
        "sample_index": int(sample_index),
        "file": str(metadata_row["file"]),
        "snr_db": float(metadata_row["snr_db"]),
        "velocity_kmh": float(metadata_row["velocity_kmh"]),
        "has_ground_truth": target is not None,
    }
    record = frame_record(
        sample=sample_info,
        receiver_used=active_before,
        active_before=active_before,
        active_after=decision.active_after,
        environment_distance=float(environment_distance),
        reliability_distance=float(active_reliability),
        decision=controller_decision_dict(decision),
        estimate=active_estimate,
        target=target,
        inference_counts=runner.receiver_inference_counts,
    )
    return record, receiver_estimates, metrics_by_receiver


def write_frame_outputs(run_dir: Path, record: dict[str, Any], estimate: np.ndarray) -> None:
    with (run_dir / "frame_result.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)
    np.savez_compressed(run_dir / "estimate.npz", estimate=estimate)


def controller_log_row(record: dict[str, Any]) -> dict[str, Any]:
    sample = record["sample"]
    decision = record["decision"]
    confidence = decision["active_confidence_margin"]
    return {
        "sample_index": sample["sample_index"],
        "file": sample["file"],
        "snr_db": sample["snr_db"],
        "velocity_kmh": sample["velocity_kmh"],
        "active_model_before": decision["active_before"],
        "active_model_after": decision["active_after"],
        "active_confidence_margin": confidence,
        "active_confidence_z": decision["active_confidence_z"],
        "environment_distance": record["environment"]["distance"],
        "decision": decision["action"],
        "reason": decision["reason"],
        "recommended_adaptation_model": decision["adaptation_recommendation"],
        "candidate_confidence_z_scores": json.dumps(decision["candidate_confidence_z_scores"], sort_keys=True),
        "candidate_gate_results": json.dumps(decision["candidate_gate_results"], sort_keys=True),
        "ber": record["metrics"]["ber"],
        "ser": record["metrics"]["ser"],
        "nmse": record["metrics"]["nmse"],
    }


def write_batch_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    all_estimates: list[dict[str, np.ndarray]],
    all_metrics: list[dict[str, dict[str, float]]],
) -> dict[str, Any]:
    with (run_dir / "frames.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")
    pd.DataFrame([controller_log_row(record) for record in records]).to_csv(
        run_dir / "decision_log.csv", index=False
    )
    prediction_array = np.stack(
        [np.stack([estimates[name] for name in RECEIVER_NAMES]) for estimates in all_estimates]
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        receiver_names=np.asarray(RECEIVER_NAMES),
        estimates=prediction_array,
        sample_files=np.asarray([record["sample"]["file"] for record in records]),
    )

    decision_counts = Counter(record["decision"]["action"] for record in records)
    summary: dict[str, Any] = {
        "frames": len(records),
        "decision_counts": {name: int(decision_counts.get(name, 0)) for name in ("KEEP", "SWITCH", "ADAPT")},
        "switch_destinations": dict(Counter(
            record["decision"]["active_after"]
            for record in records
            if record["decision"]["action"] == "SWITCH"
        )),
        "adaptation_recommendations": dict(Counter(
            record["decision"]["adaptation_recommendation"]
            for record in records
            if record["decision"]["action"] == "ADAPT"
        )),
        "environment_distance_used_for_controller_action": False,
        "test_labels_used_for_controller_actions": False,
        "fresh_receiver_scoring_for_audit": True,
        "ground_truth_frames": sum(record["sample"]["has_ground_truth"] for record in records),
    }
    scored_records = [index for index, metrics in enumerate(all_metrics) if metrics]
    if scored_records:
        adaptive = [all_metrics[index][records[index]["receiver"]["used_for_this_frame"]] for index in scored_records]
        oamp = [all_metrics[index]["oamp_dl"] for index in scored_records]
        summary["adaptive_pipeline"] = {
            metric: float(np.mean([row[metric] for row in adaptive]))
            for metric in ("ber", "ser", "nmse")
        }
        summary["always_oamp_dl"] = {
            metric: float(np.mean([row[metric] for row in oamp]))
            for metric in ("ber", "ser", "nmse")
        }
        best_matches = []
        for index in scored_records:
            best_receiver = min(all_metrics[index], key=lambda name: all_metrics[index][name]["ber"])
            best_matches.append(records[index]["receiver"]["used_for_this_frame"] == best_receiver)
        summary["match_rate_to_per_frame_true_best_ber_receiver"] = float(np.mean(best_matches))
        summary["scored_frames"] = len(scored_records)
        summary["metrics_status"] = "reference_only_requires_ground_truth"
    else:
        summary["adaptive_pipeline"] = None
        summary["always_oamp_dl"] = None
        summary["match_rate_to_per_frame_true_best_ber_receiver"] = None
        summary["scored_frames"] = 0
        summary["metrics_status"] = "unavailable_no_ground_truth"

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    return summary


def load_calibration_dataset(root: Path, config: Config) -> tuple[Path, pd.DataFrame]:
    raw_dir = root / config.dataset.raw_dir
    split_path = root / config.dataset.processed_dir / config.dataset.split_metadata_file
    if not split_path.is_file():
        raise FileNotFoundError(f"Calibration split metadata not found: {split_path}")
    return raw_dir, pd.read_csv(split_path)


def adapt_oamp_dl_for_events(
    config: Config,
    calibration_raw_dir: Path,
    calibration_metadata: pd.DataFrame,
    records: list[dict[str, Any]],
    run_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run the existing fixed-step OAMP-DL adaptation protocol for ADAPT events."""
    from models.oamp.oamp_dl import OAMPDLDetector
    from training.oamp.dataset import OAMPDataset
    from training.supervisor.adapt_oamp_dl_fewshot import (
        ANCHOR_COEFFICIENT,
        FIXED_OPTIMIZER_STEPS,
        GRADIENT_CLIP_NORM,
        LEARNING_RATE,
        SEED,
        WEIGHT_DECAY,
        sample_metrics,
        set_seed,
    )

    grouped_events: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        decision = record["decision"]
        if decision["action"] == "ADAPT" and decision["adaptation_recommendation"] == "oamp_dl":
            sample = record["sample"]
            grouped_events[(int(sample["snr_db"]), int(sample["velocity_kmh"]))].append(record)
    results = []
    for (snr, velocity), events in grouped_events.items():
        condition = calibration_metadata.loc[
            (calibration_metadata["snr_db"].astype(int) == snr)
            & (calibration_metadata["velocity_kmh"].astype(int) == velocity)
        ]
        calibration = condition.loc[
            condition["split"].astype(str).str.lower() == "validation"
        ].reset_index(drop=True)
        test = condition.loc[
            condition["split"].astype(str).str.lower() == "test"
        ].reset_index(drop=True)
        if len(calibration) != 15:
            results.append({
                "condition": {"snr_db": snr, "velocity_kmh": velocity},
                "status": "not_run",
                "reason": f"requires exactly 15 validation calibration frames; found {len(calibration)}",
            })
            continue
        if len(test) < 2:
            results.append({
                "condition": {"snr_db": snr, "velocity_kmh": velocity},
                "status": "not_run",
                "reason": f"requires at least 2 held-out test frames; found {len(test)}",
            })
            continue
        event_files = {record["sample"]["file"] for record in events}
        calibration_files = set(calibration["file"].astype(str))
        test_files = set(test["file"].astype(str))
        if calibration_files & test_files or calibration_files & event_files:
            raise ValueError("ADAPT calibration frames overlap test/event frames.")

        set_seed(SEED)
        observation_count = int(config.representation.expected_channel_shape.rows)
        symbol_count = int(config.representation.expected_channel_shape.cols)
        model = OAMPDLDetector(observation_count, symbol_count, int(config.oamp_dl.iterations)).to(device)
        base_checkpoint_path = PROJECT_ROOT / config.oamp_dl.output.directory / config.oamp_dl.output.checkpoint_file
        base_checkpoint = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(base_checkpoint["model_state_dict"])
        trainable_names = {"step_logits", "damping_logits", "variance_logits"}
        parameters = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in trainable_names)
            if parameter.requires_grad:
                parameters.append(parameter)
        scalar_count = sum(parameter.numel() for parameter in parameters)
        if scalar_count != 3 * int(config.oamp_dl.iterations):
            raise ValueError(f"Unexpected OAMP-DL adaptation parameter count: {scalar_count}.")
        original_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if name in trainable_names
        }
        calibration_data = OAMPDataset(calibration, calibration_raw_dir, config)
        calibration_samples = [calibration_data[index] for index in range(len(calibration_data))]
        packed_calibration = torch.stack([item[0] for item in calibration_samples]).to(device)
        target_calibration = torch.stack([item[1] for item in calibration_samples]).to(device)
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        history = []
        for step in range(1, FIXED_OPTIMIZER_STEPS + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(packed_calibration)
            data_loss = torch.nn.functional.mse_loss(prediction.real, target_calibration.real)
            data_loss = data_loss + torch.nn.functional.mse_loss(prediction.imag, target_calibration.imag)
            drift = torch.cat([
                (parameter - original_parameters[name]).reshape(-1)
                for name, parameter in model.named_parameters()
                if name in trainable_names
            ]).square().mean()
            anchor_loss = ANCHOR_COEFFICIENT * drift
            loss = data_loss + anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite OAMP-DL adaptation loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
            optimizer.step()
            history.append({
                "step": step,
                "data_loss": float(data_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            })
        model.eval()

        condition_output = run_dir / "adapted" / f"snr{snr}_v{velocity}"
        condition_output.mkdir(parents=True, exist_ok=True)
        adapted_checkpoint_path = condition_output / "oamp_dl_adapted.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(base_checkpoint_path.relative_to(PROJECT_ROOT)),
            "adaptation": {
                "condition": {"snr_db": snr, "velocity_kmh": velocity},
                "calibration_split": "validation",
                "calibration_samples": calibration["file"].astype(str).tolist(),
                "controller_adapt_events": [event["sample"] for event in events],
                "trainable_parameters": sorted(trainable_names),
                "trainable_scalar_count": scalar_count,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "anchor_coefficient": ANCHOR_COEFFICIENT,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "fixed_optimizer_steps": FIXED_OPTIMIZER_STEPS,
                "seed": SEED,
            },
        }, adapted_checkpoint_path)
        pd.DataFrame(history).to_csv(condition_output / "adaptation_history.csv", index=False)

        test_dataset = OAMPDataset(test, calibration_raw_dir, config)
        test_samples = [test_dataset[index] for index in range(len(test_dataset))]
        packed_test = torch.stack([item[0] for item in test_samples]).to(device)
        target_test = torch.stack([item[1] for item in test_samples]).to(device)
        base_model = OAMPDLDetector(observation_count, symbol_count, int(config.oamp_dl.iterations)).to(device)
        base_model.load_state_dict(base_checkpoint["model_state_dict"])
        base_model.eval()
        with torch.no_grad():
            original_predictions = base_model(packed_test).cpu().numpy()
            adapted_predictions = model(packed_test).cpu().numpy()
        target_array = target_test.cpu().numpy()
        original_metrics = np.asarray([sample_metrics(p, t) for p, t in zip(original_predictions, target_array)])
        adapted_metrics = np.asarray([sample_metrics(p, t) for p, t in zip(adapted_predictions, target_array)])
        paired_delta = adapted_metrics - original_metrics
        metrics_names = ("ber", "ser", "nmse")
        paired = {}
        for index, name in enumerate(metrics_names):
            mean_delta = float(paired_delta[:, index].mean())
            half_width = float(1.96 * paired_delta[:, index].std(ddof=1) / np.sqrt(len(paired_delta)))
            paired[name] = {
                "mean": mean_delta,
                "ci95_low": mean_delta - half_width,
                "ci95_high": mean_delta + half_width,
            }
        report = {
            "status": "adapted",
            "condition": {"snr_db": snr, "velocity_kmh": velocity},
            "adapt_events": len(events),
            "calibration_samples": len(calibration),
            "test_samples": len(test),
            "trainable_scalar_count": scalar_count,
            "fixed_optimizer_steps": FIXED_OPTIMIZER_STEPS,
            "adapted_checkpoint": display_path(adapted_checkpoint_path),
            "original_checkpoint_preserved": base_checkpoint_path.is_file(),
            "original_mean_metrics": {
                name: float(original_metrics.mean(axis=0)[index])
                for index, name in enumerate(metrics_names)
            },
            "adapted_mean_metrics": {
                name: float(adapted_metrics.mean(axis=0)[index])
                for index, name in enumerate(metrics_names)
            },
            "paired_adapted_minus_original_ci95": paired,
            "test_labels_used_for_training": False,
        }
        with (condition_output / "evaluation_results.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
        results.append(report)
    return results


def validate_args(args: argparse.Namespace, config: Config) -> None:
    if args.generate or args.generate_dataset:
        if args.seed is None:
            raise ValueError("--seed is required with generated-data modes.")
        if args.seed == int(config.reproducibility.seed):
            raise ValueError("Generated-data seed must differ from the configured V1 seed.")
    if args.generate or args.input:
        if args.snr is None or args.velocity is None:
            raise ValueError("--snr and --velocity are required with --input and --generate.")
    if args.generate_dataset and args.output_dir is None:
        raise ValueError("--output-dir is required with --generate-dataset to keep data isolated.")
    if args.input and args.dataset_root:
        raise ValueError("Choose one data source mode.")
    if args.state_in and args.state_out and resolve_path(args.state_in) == resolve_path(args.state_out):
        raise ValueError("--state-in and --state-out must use different files.")


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    validate_args(args, config)
    run_dir = make_run_directory(args.output_dir)
    device = select_device(args.device)
    environment_reference, reliability_payload, controller_config = load_references()

    if args.generate:
        raw_dir, dataset_root, source_metadata, run_config = generate_one_frame(
            config, run_dir, args.snr, args.velocity, args.seed
        )
        source_row = source_metadata.iloc[0].copy()
        source_row["split"] = "test"
        metadata = pd.DataFrame([source_row])
        cache_metadata = metadata.copy()
        mode = "generated_single_frame"
    elif args.generate_dataset:
        raw_dir, dataset_root, full_metadata, run_config = prepare_generated_dataset(
            config, run_dir, args.seed
        )
        cache_metadata = full_metadata
        metadata = full_metadata.loc[
            full_metadata["split"].astype(str).str.lower() == args.split
        ].reset_index(drop=True)
        if metadata.empty:
            raise ValueError(f"Generated dataset has no {args.split} split.")
        mode = "generated_dataset_split"
    elif args.dataset_root:
        raw_dir, dataset_root, full_metadata = dataset_paths(args.dataset_root, config)
        cache_metadata = full_metadata
        metadata = full_metadata.loc[
            full_metadata["split"].astype(str).str.lower() == args.split
        ].reset_index(drop=True)
        if metadata.empty:
            raise ValueError(f"Dataset has no {args.split} split.")
        run_config = config
        mode = "existing_dataset_split"
    else:
        input_path = resolve_path(args.input)
        if not input_path.is_file() or input_path.suffix.lower() != ".npz":
            raise FileNotFoundError(f"--input must point to an existing .npz frame: {input_path}")
        raw_dir = input_path.parent
        dataset_root = raw_dir.parent
        metadata = make_frame_metadata(input_path.name, args.snr, args.velocity)
        cache_metadata = metadata.copy()
        run_config = config
        mode = "existing_single_frame"

    controller = build_controller(reliability_payload, controller_config)
    restore_controller_state(controller, args.state_in)
    runner = build_runner(
        metadata,
        raw_dir,
        run_config,
        environment_reference,
        device,
        run_dir,
        cache_metadata,
    )

    records: list[dict[str, Any]] = []
    all_estimates: list[dict[str, np.ndarray]] = []
    all_metrics: list[dict[str, dict[str, float]]] = []
    for sample_index, (_, metadata_row) in enumerate(metadata.iterrows()):
        record, estimates, metrics = process_frame(
            sample_index,
            metadata_row,
            runner,
            controller,
            reliability_payload,
            raw_dir,
            score_all_receivers=len(metadata) > 1,
        )
        record["run_mode"] = mode
        records.append(record)
        all_estimates.append(estimates)
        all_metrics.append(metrics)

    save_controller_state(controller, args.state_out)
    adaptation_results = []
    adapt_events = [
        record for record in records
        if record["decision"]["action"] == "ADAPT"
    ]
    if adapt_events:
        oamp_events = [
            record for record in adapt_events
            if record["decision"]["adaptation_recommendation"] == "oamp_dl"
        ]
        unsupported = [
            record["decision"]["adaptation_recommendation"]
            for record in adapt_events
            if record["decision"]["adaptation_recommendation"] != "oamp_dl"
        ]
        if args.calibration_root:
            calibration_root = resolve_path(args.calibration_root)
            calibration_raw_dir, calibration_metadata = load_calibration_dataset(calibration_root, run_config)
        elif args.generate_dataset or args.dataset_root:
            calibration_raw_dir, calibration_metadata = raw_dir, cache_metadata
        else:
            calibration_raw_dir, calibration_metadata = None, pd.DataFrame()
        if oamp_events and calibration_raw_dir is not None:
            try:
                adaptation_results = adapt_oamp_dl_for_events(
                    run_config,
                    calibration_raw_dir,
                    calibration_metadata,
                    records,
                    run_dir,
                    device,
                )
            except Exception as error:
                adaptation_results.append({
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                })
        elif oamp_events:
            adaptation_results.append({
                "status": "not_run",
                "reason": "OAMP-DL ADAPT was recommended, but no validation calibration dataset was supplied.",
            })
        if unsupported:
            adaptation_results.append({
                "status": "unsupported_receiver_adaptation",
                "recommended_models": unsupported,
                "reason": "The existing few-shot adapter updates OAMP-DL scalar controls only.",
            })

    if len(metadata) == 1:
        record = records[0]
        record["adaptation_runs"] = adaptation_results
        write_frame_outputs(run_dir, record, all_estimates[0][record["receiver"]["used_for_this_frame"]])
        print(json.dumps(record, indent=2, allow_nan=False))
        return

    summary = write_batch_outputs(run_dir, records, all_estimates, all_metrics)
    summary["run_mode"] = mode
    summary["dataset_root"] = str(dataset_root)
    summary["split"] = args.split
    summary["seed"] = int(run_config.reproducibility.seed)
    summary["adaptation_runs"] = adaptation_results
    summary["receiver_inference_counts"] = runner.receiver_inference_counts
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
