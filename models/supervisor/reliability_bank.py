"""Fresh inference adapters for reliability-reference generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from models.gnn.otfs_gnn import OTFSGNN
from models.gnn.pi_egnn import PIEGNN, qpsk_max_posterior
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


def _pack_original_gnn(
    rx_dd: np.ndarray,
    h_hat: np.ndarray,
    y_data: np.ndarray,
    mmse_estimate: np.ndarray,
    noise_power: float,
    config: Any,
) -> torch.Tensor:
    threshold = float(config.gnn.edge_threshold_fraction_of_max) * max(
        float(np.max(np.abs(h_hat))), np.finfo(float).eps
    )
    edge_mask = (np.abs(h_hat) >= threshold).astype(np.float32)
    edge_features = np.stack((h_hat.real, h_hat.imag), axis=-1).astype(np.float32)
    observation_features = np.concatenate(
        (
            np.stack((y_data.real, y_data.imag), axis=-1),
            edge_mask.mean(axis=1, keepdims=True),
            np.full((y_data.shape[0], 1), noise_power, dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    symbol_features = np.concatenate(
        (
            np.stack((mmse_estimate.real, mmse_estimate.imag), axis=-1),
            np.linalg.norm(h_hat, axis=0)[:, None].astype(np.float32),
            edge_mask.mean(axis=0)[:, None],
        ),
        axis=-1,
    ).astype(np.float32)
    packed = np.concatenate(
        (
            observation_features.reshape(-1),
            symbol_features.reshape(-1),
            edge_features.reshape(-1),
            edge_mask.reshape(-1),
        )
    ).astype(np.float32)
    return torch.from_numpy(packed).unsqueeze(0)


def _pack_pi_egnn(
    h_hat: np.ndarray,
    y_data: np.ndarray,
    mmse_estimate: np.ndarray,
    noise_power: float,
    cache: dict[str, np.ndarray],
    cache_index: int,
    config: Any,
) -> torch.Tensor:
    linear_estimate = cache["linear_estimate"][cache_index]
    effective_variance = float(cache["effective_variance"][cache_index])
    oamp_estimate = cache["prior_estimate"][cache_index]
    threshold = float(config.gnn.edge_threshold_fraction_of_max) * max(
        float(np.max(np.abs(h_hat))), np.finfo(float).eps
    )
    edge_mask = (np.abs(h_hat) >= threshold).astype(np.float32)
    edge_features = np.stack((h_hat.real, h_hat.imag), axis=-1).astype(np.float32)
    symbol_features = np.concatenate(
        (
            np.stack((mmse_estimate.real, mmse_estimate.imag), axis=-1),
            np.linalg.norm(h_hat, axis=0)[:, None],
            edge_mask.mean(axis=0)[:, None],
            np.stack((oamp_estimate.real, oamp_estimate.imag), axis=-1),
            np.abs(oamp_estimate)[:, None],
            np.full((h_hat.shape[1], 1), np.log1p(effective_variance)),
            np.full(
                (h_hat.shape[1], 1),
                noise_power / max(noise_power + effective_variance, np.finfo(float).eps),
            ),
            qpsk_max_posterior(linear_estimate, effective_variance)[:, None],
        ),
        axis=-1,
    ).astype(np.float32)
    observation_features = np.concatenate(
        (
            np.stack((y_data.real, y_data.imag), axis=-1),
            edge_mask.mean(axis=1, keepdims=True),
            np.full((y_data.shape[0], 1), noise_power, dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    packed = np.concatenate(
        (
            observation_features.reshape(-1),
            symbol_features.reshape(-1),
            edge_features.reshape(-1),
            edge_mask.reshape(-1),
        )
    ).astype(np.float32)
    return torch.from_numpy(packed).unsqueeze(0)


class FrozenReceiverRunner:
    """Run a requested frozen receiver on demand for one target-free frame."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        raw_dir: Path,
        config: Any,
        project_root: Path,
        environment_reference: SimpleEnvironmentReference,
        device: torch.device,
        cache_metadata: pd.DataFrame | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.raw_dir = Path(raw_dir)
        self.config = config
        self.project_root = project_root
        self.environment_reference = environment_reference
        self.device = device
        self.models = build_receiver_models(config, project_root, device)
        self.prior_iterations = read_locked_prior_iterations(project_root)
        all_metadata = (
            cache_metadata.reset_index(drop=True)
            if cache_metadata is not None
            else load_all_split_metadata(
                project_root
                / config.dataset.root
                / config.dataset.processed_dir
                / config.dataset.split_metadata_file
            )
        )
        cache_path = cache_path or (
            project_root
            / "experiments"
            / "pi_egnn"
            / "analytical_feature_cache.npz"
        )
        self.pi_cache, _ = load_or_build_feature_cache(
            all_metadata,
            self.raw_dir,
            config,
            cache_path,
            self.prior_iterations,
        )
        self.pi_cache_indices = {
            str(filename): index
            for index, filename in enumerate(self.pi_cache["sample_files"])
        }
        self.frame_cache: dict[int, dict[str, Any]] = {}
        self.prediction_cache: dict[tuple[int, str], np.ndarray] = {}
        self.receiver_inference_counts = {name: 0 for name in RECEIVER_NAMES}

    def _frame_inputs(self, sample_index: int) -> dict[str, Any]:
        if sample_index not in self.frame_cache:
            row = self.metadata.iloc[sample_index]
            filename = str(row["file"])
            with np.load(self.raw_dir / filename, allow_pickle=False) as sample:
                rx_dd = np.asarray(sample["rx_dd"])
                h_hat = np.asarray(sample["h_hat"])
            noise_power = 10.0 ** (-float(row["snr_db"]) / 10.0)
            y_data = self.pi_cache["y_data"][self.pi_cache_indices[filename]]
            mmse_estimate = self.pi_cache["mmse_estimate"][self.pi_cache_indices[filename]]
            environment_features = extract_simple_environment_features(
                rx_dd,
                h_hat,
                noise_power,
                self.config,
                mmse_estimate=mmse_estimate,
                y_data=y_data,
            )
            self.frame_cache[sample_index] = {
                "rx_dd": rx_dd,
                "h_hat": h_hat,
                "noise_power": noise_power,
                "y_data": y_data,
                "mmse_estimate": mmse_estimate,
                "environment_distance": self.environment_reference.score(
                    environment_features
                ),
            }
        return self.frame_cache[sample_index]

    def estimate(self, sample_index: int, receiver: str) -> tuple[np.ndarray, float, float]:
        """Return estimate, QPSK margin and environment diagnostic on demand."""

        if receiver not in RECEIVER_NAMES:
            raise KeyError(f"Unknown receiver {receiver!r}.")
        key = (sample_index, receiver)
        frame = self._frame_inputs(sample_index)
        if key not in self.prediction_cache:
            if receiver == "mmse":
                prediction = frame["mmse_estimate"]
            elif receiver == "oamp_dl":
                packed = _packed_oamp(
                    frame["y_data"], frame["h_hat"], frame["noise_power"]
                ).to(self.device)
                with torch.no_grad():
                    prediction = self.models[receiver](packed)[0].cpu().numpy()
            else:
                filename = str(self.metadata.iloc[sample_index]["file"])
                if receiver == "original_gnn":
                    packed = _pack_original_gnn(
                        frame["rx_dd"],
                        frame["h_hat"],
                        frame["y_data"],
                        frame["mmse_estimate"],
                        frame["noise_power"],
                        self.config,
                    )
                else:
                    packed = _pack_pi_egnn(
                        frame["h_hat"],
                        frame["y_data"],
                        frame["mmse_estimate"],
                        frame["noise_power"],
                        self.pi_cache,
                        self.pi_cache_indices[filename],
                        self.config,
                    )
                with torch.no_grad():
                    prediction = self.models[receiver](packed.to(self.device))[0].cpu().numpy()
            self.prediction_cache[key] = np.asarray(prediction)
            self.receiver_inference_counts[receiver] += 1
        prediction = self.prediction_cache[key]
        margin = float(np.mean(np.minimum(np.abs(prediction.real), np.abs(prediction.imag))))
        return prediction, margin, float(frame["environment_distance"])

    def candidate_margins(self, sample_index: int) -> dict[str, float]:
        """Compute all other bank margins only when the controller requests them."""

        return {
            receiver: self.estimate(sample_index, receiver)[1]
            for receiver in RECEIVER_NAMES
        }


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
