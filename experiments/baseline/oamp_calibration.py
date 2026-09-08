"""Diagnose and calibrate plain OAMP against the MMSE baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config
from receivers.mmse import build_data_observation, mmse_detect
from receivers.oamp import oamp_detect


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "oamp"
CONDITION_RESULTS_PATH = OUTPUT_DIRECTORY / "plain_oamp_condition_results.csv"
RESCALED_RESULTS_PATH = OUTPUT_DIRECTORY / "plain_oamp_rescaled_results.csv"
DIAGNOSTIC_PATH = OUTPUT_DIRECTORY / "oamp_calibration_diagnostic.json"
OAMP_ITERATIONS = 20


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    prediction_bits = np.stack(
        (prediction.real >= 0, prediction.imag >= 0),
        axis=-1,
    )
    target_bits = np.stack(
        (target.real >= 0, target.imag >= 0),
        axis=-1,
    )
    bit_error_mask = prediction_bits != target_bits
    symbol_error_mask = np.any(bit_error_mask, axis=-1)
    signal_power = np.sum(np.abs(target) ** 2)
    error_power = np.sum(np.abs(prediction - target) ** 2)
    return (
        float(np.count_nonzero(bit_error_mask) / bit_error_mask.size),
        float(np.count_nonzero(symbol_error_mask) / target.size),
        float(error_power / signal_power),
    )


def load_estimates(
    metadata: pd.DataFrame,
    raw_dir: Path,
    config,
) -> list[dict[str, object]]:
    records = []

    for _, row in metadata.iterrows():
        with np.load(
            raw_dir / str(row["file"]),
            allow_pickle=False,
        ) as sample:
            rx_dd = sample["rx_dd"]
            h_hat = sample["h_hat"]
            tx_dd = sample["tx_dd"]

        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        pilot_power = noise_power * 10.0 ** (
            float(config.pilot.pilot_snr_db) / 10.0
        )
        y_data = build_data_observation(
            rx_dd,
            config,
            pilot_power,
        )
        mmse_prediction = mmse_detect(
            y_data,
            h_hat,
            noise_power,
        )
        oamp_prediction = oamp_detect(
            y_data,
            h_hat,
            noise_power,
            iterations=OAMP_ITERATIONS,
        )

        records.append(
            {
                "sample_id": int(row["sample_id"]),
                "snr_db": int(row["snr_db"]),
                "velocity_kmh": int(row["velocity_kmh"]),
                "target": tx_dd,
                "mmse": mmse_prediction,
                "oamp": oamp_prediction,
            }
        )

    return records


def summarize_metrics(
    records: list[dict[str, object]],
    prediction_key: str,
) -> dict[str, float]:
    values = [
        compute_metrics(record[prediction_key], record["target"])
        for record in records
    ]
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    return {
        "ber": float(mean[0]),
        "ser": float(mean[1]),
        "nmse": float(mean[2]),
    }


def condition_rows(
    records: list[dict[str, object]],
    prediction_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    grouped = {}

    for record in records:
        key = (record["snr_db"], record["velocity_kmh"])
        grouped.setdefault(key, []).append(record)

    for (snr_db, velocity_kmh), group in sorted(grouped.items()):
        row: dict[str, object] = {
            "snr_db": snr_db,
            "velocity_kmh": velocity_kmh,
            "samples": len(group),
        }
        summaries = {
            key: summarize_metrics(group, key)
            for key in prediction_keys
        }

        for key, summary in summaries.items():
            row[f"{key}_ber"] = summary["ber"]
            row[f"{key}_ser"] = summary["ser"]
            row[f"{key}_nmse"] = summary["nmse"]

        if "oamp" in summaries and "mmse" in summaries:
            row["oamp_beats_mmse_all_metrics"] = all(
                summaries["oamp"][metric]
                < summaries["mmse"][metric]
                for metric in ("ber", "ser", "nmse")
            )

        if "oamp_rescaled" in summaries and "mmse" in summaries:
            row["oamp_rescaled_beats_mmse_all_metrics"] = all(
                summaries["oamp_rescaled"][metric]
                < summaries["mmse"][metric]
                for metric in ("ber", "ser", "nmse")
            )

        rows.append(row)

    return rows


def fit_validation_scale(
    records: list[dict[str, object]],
) -> float:
    numerator = 0.0
    denominator = 0.0

    for record in records:
        oamp = record["oamp"]
        target = record["target"]
        numerator += float(np.real(np.vdot(oamp, target)))
        denominator += float(np.vdot(oamp, oamp).real)

    if denominator <= np.finfo(float).eps:
        raise ValueError("Cannot fit rescaling factor from zero OAMP output.")

    return numerator / denominator


def calibration_diagnostics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    magnitudes = {}
    true_values = np.concatenate([record["target"] for record in records])
    oamp_values = np.concatenate([record["oamp"] for record in records])
    mmse_values = np.concatenate([record["mmse"] for record in records])

    for name, values in (
        ("true", true_values),
        ("mmse", mmse_values),
        ("oamp", oamp_values),
    ):
        magnitudes[name] = {
            "real_abs_mean": float(np.mean(np.abs(values.real))),
            "imag_abs_mean": float(np.mean(np.abs(values.imag))),
            "complex_abs_mean": float(np.mean(np.abs(values))),
        }

    worse_sample_margins = []
    for record in records:
        target = record["target"]
        oamp = record["oamp"]
        mmse = record["mmse"]
        oamp_bits = np.stack((oamp.real >= 0, oamp.imag >= 0), axis=-1)
        mmse_bits = np.stack((mmse.real >= 0, mmse.imag >= 0), axis=-1)
        target_bits = np.stack(
            (target.real >= 0, target.imag >= 0),
            axis=-1,
        )
        oamp_errors = np.any(oamp_bits != target_bits, axis=-1)
        mmse_errors = np.any(mmse_bits != target_bits, axis=-1)
        oamp_worse = oamp_errors & ~mmse_errors
        margins = np.minimum(np.abs(oamp.real), np.abs(oamp.imag))
        worse_sample_margins.extend(margins[oamp_worse].tolist())

    return {
        "magnitude_means": magnitudes,
        "oamp_worse_than_mmse_flipped_symbol_count": len(worse_sample_margins),
        "oamp_worse_than_mmse_flipped_margin_mean": (
            float(np.mean(worse_sample_margins))
            if worse_sample_margins
            else None
        ),
        "oamp_worse_than_mmse_flipped_margin_median": (
            float(np.median(worse_sample_margins))
            if worse_sample_margins
            else None
        ),
        "oamp_worse_than_mmse_flipped_margin_near_zero_fraction": (
            float(np.mean(np.asarray(worse_sample_margins) < 0.1))
            if worse_sample_margins
            else None
        ),
    }


def main() -> None:
    config = load_config(CONFIG_PATH)
    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    split_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )
    metadata = pd.read_csv(split_path)
    validation_metadata = metadata.loc[
        metadata["split"].astype(str).str.lower() == "validation"
    ].reset_index(drop=True)
    test_metadata = metadata.loc[
        metadata["split"].astype(str).str.lower() == "test"
    ].reset_index(drop=True)

    expected_validation = int(
        config.dataset.expected_samples * config.split.validation_ratio
    )
    expected_test = int(
        config.dataset.expected_samples * config.split.test_ratio
    )
    if len(validation_metadata) != expected_validation:
        raise ValueError(
            f"Expected {expected_validation} validation samples; "
            f"found {len(validation_metadata)}."
        )
    if len(test_metadata) != expected_test:
        raise ValueError(
            f"Expected {expected_test} test samples; "
            f"found {len(test_metadata)}."
        )

    validation_records = load_estimates(
        validation_metadata,
        raw_dir,
        config,
    )
    test_records = load_estimates(
        test_metadata,
        raw_dir,
        config,
    )
    scale = fit_validation_scale(validation_records)

    for record in test_records:
        record["oamp_rescaled"] = scale * record["oamp"]

    raw_rows = condition_rows(test_records, ("mmse", "oamp"))
    rescaled_rows = condition_rows(
        test_records,
        ("mmse", "oamp", "oamp_rescaled"),
    )
    diagnostics = calibration_diagnostics(test_records)
    diagnostics.update(
        {
            "validation_samples": len(validation_records),
            "test_samples": len(test_records),
            "validation_scale": scale,
            "oamp_iterations": OAMP_ITERATIONS,
            "rescaling_fit_source": "validation split only",
            "oamp_beats_mmse_all_test_metrics": all(
                summarize_metrics(test_records, "oamp")[metric]
                < summarize_metrics(test_records, "mmse")[metric]
                for metric in ("ber", "ser", "nmse")
            ),
            "oamp_rescaled_beats_mmse_all_test_metrics": all(
                summarize_metrics(test_records, "oamp_rescaled")[metric]
                < summarize_metrics(test_records, "mmse")[metric]
                for metric in ("ber", "ser", "nmse")
            ),
            "aggregate_raw": {
                "mmse": summarize_metrics(test_records, "mmse"),
                "oamp": summarize_metrics(test_records, "oamp"),
            },
            "aggregate_rescaled": {
                "mmse": summarize_metrics(test_records, "mmse"),
                "oamp_rescaled": summarize_metrics(
                    test_records,
                    "oamp_rescaled",
                ),
            },
        }
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(
        CONDITION_RESULTS_PATH,
        index=False,
    )
    pd.DataFrame(rescaled_rows).to_csv(
        RESCALED_RESULTS_PATH,
        index=False,
    )
    with DIAGNOSTIC_PATH.open("w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)

    print(json.dumps(diagnostics, indent=2))
    print(f"Raw condition results saved to: {CONDITION_RESULTS_PATH}")
    print(f"Rescaled condition results saved to: {RESCALED_RESULTS_PATH}")
    print(f"Calibration diagnostic saved to: {DIAGNOSTIC_PATH}")


if __name__ == "__main__":
    main()