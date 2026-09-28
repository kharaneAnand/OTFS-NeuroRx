"""Train PI-EGNN or run its validation-only OAMP prior sanity check."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.pi_egnn import PIEGNN, oamp_prior_trace
from src.config.loader import load_config
from src.receivers.mmse import build_data_observation
from training.dnn.trainer import DNNTrainer
from training.gnn.pi_egnn_cache import (
    load_or_build_feature_cache,
    load_all_split_metadata,
    read_locked_prior_iterations,
)
from training.gnn.pi_egnn_dataset import PIEGNNGraphDataset, load_split_metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    predicted_bits = np.stack((prediction.real >= 0, prediction.imag >= 0), axis=-1)
    target_bits = np.stack((target.real >= 0, target.imag >= 0), axis=-1)
    bit_errors = predicted_bits != target_bits
    symbol_errors = np.any(bit_errors, axis=-1)
    return (
        float(np.count_nonzero(bit_errors) / bit_errors.size),
        float(np.count_nonzero(symbol_errors) / target.size),
        float(np.sum(np.abs(prediction - target) ** 2) / np.sum(np.abs(target) ** 2)),
    )


def summarize(values: list[tuple[float, float, float]]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0)
    standard_deviation = array.std(axis=0, ddof=1)
    half_width = 1.96 * standard_deviation / np.sqrt(len(values))
    result = {}
    for index, name in enumerate(("ber", "ser", "nmse")):
        result[name] = float(mean[index])
        result[f"{name}_ci95"] = float(half_width[index])
    return result


def run_prior_sanity(config, raw_dir: Path, metadata: pd.DataFrame) -> dict[str, object]:
    values = {count: [] for count in (3, 5, 10)}
    elapsed = {count: [] for count in (3, 5, 10)}
    for _, row in metadata.iterrows():
        with np.load(raw_dir / str(row["file"]), allow_pickle=False) as sample:
            rx_dd, h_hat, target = sample["rx_dd"], sample["h_hat"], sample["tx_dd"]
        noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
        trace = oamp_prior_trace(
            build_data_observation(rx_dd, config), h_hat, noise_power, iterations=10
        )
        for count in values:
            values[count].append(compute_metrics(trace[count - 1].estimate, target))
            elapsed[count].append(trace[count - 1].elapsed_seconds)
    result = {
        "split": "validation",
        "samples": len(metadata),
        "input_contract": "rx_dd, H_hat, configured per-sample noise power; target used only for metrics",
        "iterations": {
            str(count): {
                **summarize(values[count]),
                "mean_elapsed_seconds_per_sample": float(np.mean(elapsed[count])),
            }
            for count in values
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prior-sanity-only",
        action="store_true",
        help="Compare 3/5/10 OAMP iterations on validation data, without training.",
    )
    parser.add_argument(
        "--prepare-cache-only",
        action="store_true",
        help="Build/verify the all-split analytical feature cache without training.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.reproducibility.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    prior_iterations = read_locked_prior_iterations(PROJECT_ROOT)
    all_metadata = load_all_split_metadata(metadata_path)
    cache_path = PROJECT_ROOT / "experiments" / "pi_egnn" / "analytical_feature_cache.npz"

    if args.prepare_cache_only:
        cache, built = load_or_build_feature_cache(
            all_metadata,
            raw_dir,
            config,
            cache_path,
            prior_iterations,
            verify_samples=5,
        )
        print(f"Analytical feature cache {'built' if built else 'reused'}: {cache_path}")
        print(f"Cached samples: {len(cache['sample_files'])}; locked prior iterations: {prior_iterations}")
        return

    if args.prior_sanity_only:
        validation = load_split_metadata(metadata_path, "validation")
        result = run_prior_sanity(config, raw_dir, validation)
        output_directory = PROJECT_ROOT / "experiments" / "pi_egnn"
        output_directory.mkdir(parents=True, exist_ok=True)
        with (output_directory / "prior_iteration_sanity.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(json.dumps(result, indent=2))
        return

    cache, built = load_or_build_feature_cache(
        all_metadata, raw_dir, config, cache_path, prior_iterations
    )
    print(f"Analytical feature cache {'built' if built else 'reused'}: {cache_path}")

    train_dataset = PIEGNNGraphDataset(
        load_split_metadata(metadata_path, "train"), raw_dir, config, cache
    )
    validation_dataset = PIEGNNGraphDataset(
        load_split_metadata(metadata_path, "validation"), raw_dir, config, cache
    )
    test_metadata = load_split_metadata(metadata_path, "test")
    batch_size = int(config.gnn.training.batch_size)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = PIEGNN(
        observation_count=int(config.representation.expected_channel_shape.rows),
        symbol_count=int(config.representation.expected_channel_shape.cols),
        hidden_features=int(config.gnn.hidden_features),
        layers=int(config.gnn.message_passing_layers),
    )
    output_directory = PROJECT_ROOT / "experiments" / "pi_egnn"
    trainer = DNNTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=int(config.gnn.training.epochs),
        learning_rate=float(config.gnn.training.learning_rate),
        weight_decay=float(config.gnn.training.weight_decay),
        gradient_clip_norm=float(config.gnn.training.gradient_clip_norm),
        early_stopping_patience=int(config.gnn.training.early_stopping_patience),
        early_stopping_min_delta=float(config.gnn.training.early_stopping_min_delta),
        scheduler_factor=float(config.gnn.training.scheduler_factor),
        scheduler_patience=int(config.gnn.training.scheduler_patience),
        checkpoint_path=output_directory / "best_model.pt",
        history_path=output_directory / "training_history.csv",
        device=device,
    )
    trainer.fit()
    print(f"PI-EGNN trained; fixed test split has {len(test_metadata)} samples.")
    print(f"Checkpoint: {output_directory / 'best_model.pt'}")


if __name__ == "__main__":
    main()