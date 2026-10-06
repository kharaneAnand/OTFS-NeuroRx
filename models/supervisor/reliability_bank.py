"""Fresh inference adapters for reliability-reference generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.gnn.otfs_gnn import OTFSGNN
from models.gnn.pi_egnn import PIEGNN
from models.supervisor.reliability_detector import reliability_feature_vector
from models.supervisor.simple_environment_detector import (
    SimpleEnvironmentReference,
    extract_simple_environment_features,
)
from models.oamp.oamp_dl import OAMPDLDetector
from src.receivers.mmse import build_data_observation, mmse_detect
from training.gnn.dataset import OTFSGraphDataset
from training.gnn.pi_egnn_cache import (
    load_all_split_metadata,
    load_or_build_feature_cache,
    read_locked_prior_iterations,
)
from training.gnn.pi_egnn_dataset import PIEGNNGraphDataset

RECEIVER_NAMES = ("mmse", "oamp_dl", "original_gnn", "pi_egnn")


def resolve_existing_path(project_root: Path, relative: str) -> Path:
    candidates = (
        project_root / relative,
        project_root / "gpu_results" / Path(relative).relative_to("experiments"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find artifact in: {candidates}")


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


def load_environment_reference(project_root: Path) -> SimpleEnvironmentReference:
    path = project_root / "experiments" / "environment_detector_simple" / "reference.json"
    with path.open(encoding="utf-8") as file:
        return SimpleEnvironmentReference.from_dict(json.load(file))


def _load_sample(raw_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(raw_dir / filename, allow_pickle=False) as sample:
        return (
            np.asarray(sample["rx_dd"]),
            np.asarray(sample["h_hat"]),
            np.asarray(sample["tx_dd"]),
        )


def _packed_oamp(y_data: np.ndarray, h_hat: np.ndarray, noise_power: float) -> torch.Tensor:
    packed = np.concatenate(
        (
            y_data.reshape(-1),
            h_hat.reshape(-1),
            np.asarray([noise_power], dtype=np.complex64),
        )
    ).astype(np.complex64)
    return torch.from_numpy(packed).unsqueeze(0)


def build_receiver_models(config: Any, project_root: Path, device: torch.device) -> dict[str, torch.nn.Module]:
    observation_count = int(config.representation.expected_channel_shape.rows)
    symbol_count = int(config.representation.expected_channel_shape.cols)
    models: dict[str, torch.nn.Module] = {}

    oamp = OAMPDLDetector(observation_count, symbol_count, int(config.oamp_dl.iterations)).to(device)
    load_model_checkpoint(
        oamp,
        resolve_existing_path(
            project_root,
            str(Path(config.oamp_dl.output.directory) / config.oamp_dl.output.checkpoint_file),
        ),
        device,
    )
    models["oamp_dl"] = oamp

    gnn = OTFSGNN(
        observation_count,
        symbol_count,
        int(config.gnn.hidden_features),
        int(config.gnn.message_passing_layers),
    ).to(device)
    load_model_checkpoint(
        gnn,
        project_root / config.gnn.output.mmse_directory / config.gnn.output.checkpoint_file,
        device,
    )
    models["original_gnn"] = gnn

    pi = PIEGNN(
        observation_count,
        symbol_count,
        int(config.gnn.hidden_features),
        int(config.gnn.message_passing_layers),
    ).to(device)
    pi_checkpoint = resolve_existing_path(project_root, "experiments/pi_egnn/best_model.pt")
    load_model_checkpoint(pi, pi_checkpoint, device)
    models["pi_egnn"] = pi
    return models


def collect_receiver_rows(
    metadata: pd.DataFrame,
    raw_dir: Path,
    config: Any,
    project_root: Path,
    environment_reference: SimpleEnvironmentReference,
    device: torch.device,
) -> list[dict[str, object]]:
    """Run all frozen receivers fresh and return target-free features plus offline labels."""

    models = build_receiver_models(config, project_root, device)
    prior_iterations = read_locked_prior_iterations(project_root)
    all_metadata = load_all_split_metadata(
        project_root / config.dataset.root / config.dataset.processed_dir / config.dataset.split_metadata_file
    )
    pi_cache_path = project_root / "experiments" / "environment_detector_simple" / "pi_egnn_feature_cache.npz"
    pi_cache, _ = load_or_build_feature_cache(
        all_metadata,
        raw_dir,
        config,
        pi_cache_path,
        prior_iterations,
    )
    gnn_datasets = {
        "original_gnn": OTFSGraphDataset(metadata, raw_dir, config, "mmse"),
        "pi_egnn": PIEGNNGraphDataset(metadata, raw_dir, config, pi_cache),
    }

    rows: list[dict[str, object]] = []
    for index, (_, metadata_row) in enumerate(metadata.iterrows()):
        filename = str(metadata_row["file"])
        rx_dd, h_hat, tx_dd = _load_sample(raw_dir, filename)
        noise_power = 10.0 ** (-float(metadata_row["snr_db"]) / 10.0)
        y_data = build_data_observation(rx_dd, config)
        environment_features = extract_simple_environment_features(
            rx_dd, h_hat, noise_power, config
        )
        environment_distance = environment_reference.score(environment_features)
        predictions: dict[str, np.ndarray] = {
            "mmse": mmse_detect(y_data, h_hat, noise_power),
        }
        with torch.no_grad():
            predictions["oamp_dl"] = models["oamp_dl"](
                _packed_oamp(y_data, h_hat, noise_power).to(device)
            )[0].cpu().numpy()
            for receiver_name, dataset in gnn_datasets.items():
                packed, _ = dataset[index]
                predictions[receiver_name] = models[receiver_name](
                    packed.unsqueeze(0).to(device)
                )[0].cpu().numpy()

        for receiver_name in RECEIVER_NAMES:
            estimate = predictions[receiver_name]
            confidence = float(
                reliability_feature_vector(estimate, environment_distance)[0]
            )
            bit_errors = np.stack((estimate.real >= 0, estimate.imag >= 0), axis=-1) != np.stack(
                (tx_dd.real >= 0, tx_dd.imag >= 0), axis=-1
            )
            symbol_errors = np.any(bit_errors, axis=-1)
            rows.append(
                {
                    "sample_index": index,
                    "split": str(metadata_row["split"]),
                    "file": filename,
                    "snr_db": int(metadata_row["snr_db"]),
                    "velocity_kmh": int(metadata_row["velocity_kmh"]),
                    "receiver": receiver_name,
                    "receiver_qpsk_margin": confidence,
                    "environment_distance": environment_distance,
                    "ber": float(np.count_nonzero(bit_errors) / bit_errors.size),
                    "ser": float(np.count_nonzero(symbol_errors) / tx_dd.size),
                    "nmse": float(
                        np.sum(np.abs(estimate - tx_dd) ** 2)
                        / np.sum(np.abs(tx_dd) ** 2)
                    ),
                }
            )
    return rows
