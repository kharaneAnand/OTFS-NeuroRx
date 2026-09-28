"""Pre-training cache, architecture, leakage, and solver checks for PI-EGNN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.pi_egnn import PIEGNN
from src.config.loader import load_config
from src.receivers.mmse import build_data_observation
from training.gnn.pi_egnn_cache import (
    load_all_split_metadata,
    load_or_build_feature_cache,
    read_locked_prior_iterations,
)
from training.gnn.pi_egnn_dataset import PIEGNNGraphDataset


PI_EGNN_SOURCE_FILES = (
    "models/gnn/pi_egnn.py",
    "training/gnn/pi_egnn_cache.py",
    "training/gnn/pi_egnn_dataset.py",
    "training/gnn/train_pi_egnn.py",
    "evaluations/gnn/evaluate_pi_egnn.py",
)


def check_attention_sums(
    trace: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
) -> int:
    checked = 0
    for direction, layers in trace.items():
        for layer_index, (weights, segment_ids) in enumerate(layers):
            occupied = torch.unique(segment_ids)
            sums = torch.zeros(
                int(segment_ids.max().item()) + 1,
                dtype=weights.dtype,
                device=weights.device,
            ).index_add(0, segment_ids, weights)
            if not torch.allclose(
                sums[occupied],
                torch.ones_like(sums[occupied]),
                rtol=1e-5,
                atol=1e-6,
            ):
                raise AssertionError(
                    f"Attention weights do not sum to one for {direction}, layer {layer_index}."
                )
            checked += int(occupied.numel())
    return checked


def check_no_true_channel_inputs() -> dict[str, object]:
    scanned = []
    for relative_path in PI_EGNN_SOURCE_FILES:
        path = PROJECT_ROOT / relative_path
        source = path.read_text(encoding="utf-8").lower()
        if "h_dd" in source:
            raise AssertionError(f"Forbidden true-channel reference found in {relative_path}.")
        scanned.append(relative_path)
    return {"passed": True, "files_scanned": scanned, "forbidden_token": "h_dd"}


def benchmark_solvers(
    h_hat: np.ndarray,
    noise_power: float,
    repeats: int,
) -> dict[str, object]:
    symbol_count = h_hat.shape[1]
    identity = np.eye(symbol_count, dtype=h_hat.dtype)
    normal_matrix = h_hat.conj().T @ h_hat + noise_power * identity
    rhs = h_hat.conj().T

    numpy_times = []
    numpy_solution = None
    for _ in range(repeats):
        start = time.perf_counter()
        numpy_solution = np.linalg.solve(normal_matrix, rhs)
        numpy_times.append(time.perf_counter() - start)

    result: dict[str, object] = {
        "numpy_solve_median_seconds": float(np.median(numpy_times)),
        "repeats": repeats,
        "matrix_shape": list(normal_matrix.shape),
        "torch_cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        result["torch_cuda"] = "skipped: CUDA is not available to this interpreter"
        return result

    torch_matrix_cpu = torch.from_numpy(np.ascontiguousarray(normal_matrix))
    torch_rhs_cpu = torch.from_numpy(np.ascontiguousarray(rhs))
    for _ in range(2):
        torch.linalg.solve(torch_matrix_cpu.cuda(), torch_rhs_cpu.cuda())
    torch.cuda.synchronize()

    transfer_solve_times = []
    solve_only_times = []
    matrix_gpu = torch_matrix_cpu.cuda()
    rhs_gpu = torch_rhs_cpu.cuda()
    torch_solution = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        matrix_gpu = torch_matrix_cpu.cuda()
        rhs_gpu = torch_rhs_cpu.cuda()
        torch_solution = torch.linalg.solve(matrix_gpu, rhs_gpu)
        torch.cuda.synchronize()
        transfer_solve_times.append(time.perf_counter() - start)

        torch.cuda.synchronize()
        start = time.perf_counter()
        torch_solution = torch.linalg.solve(matrix_gpu, rhs_gpu)
        torch.cuda.synchronize()
        solve_only_times.append(time.perf_counter() - start)

    torch_solution_np = torch_solution.cpu().numpy()
    result["torch_cuda_transfer_and_solve_median_seconds"] = float(
        np.median(transfer_solve_times)
    )
    result["torch_cuda_solve_only_median_seconds"] = float(
        np.median(solve_only_times)
    )
    result["max_abs_solution_difference"] = float(
        np.max(np.abs(torch_solution_np - numpy_solution))
    )
    result["torch_cuda_transfer_and_solve_faster"] = bool(
        np.median(transfer_solve_times) < np.median(numpy_times)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--benchmark-solvers",
        action="store_true",
        help="Compare NumPy solve with Torch CUDA solve when CUDA is available.",
    )
    parser.add_argument("--solver-repeats", type=int, default=7)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    metadata = load_all_split_metadata(metadata_path)
    iterations = read_locked_prior_iterations(PROJECT_ROOT)
    cache_path = PROJECT_ROOT / "experiments" / "pi_egnn" / "analytical_feature_cache.npz"
    cache, built = load_or_build_feature_cache(
        metadata, raw_dir, config, cache_path, iterations, verify_samples=5
    )

    dataset = PIEGNNGraphDataset(metadata.head(1), raw_dir, config, cache)
    packed, target = dataset[0]
    model = PIEGNN(
        observation_count=int(config.representation.expected_channel_shape.rows),
        symbol_count=int(config.representation.expected_channel_shape.cols),
        hidden_features=int(config.gnn.hidden_features),
        layers=int(config.gnn.message_passing_layers),
    ).to(device)
    model.eval()
    packed_device = packed.unsqueeze(0).to(device)
    target_device = target.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction, attention_trace = model(
            packed_device, return_attention=True
        )
        _, symbol_features, _, _ = model._unpack(packed_device)
        mmse_start = torch.complex(symbol_features[..., 0], symbol_features[..., 1])

    model_device = next(model.parameters()).device
    attention_devices = {
        str(weights.device)
        for layers in attention_trace.values()
        for weights, _ in layers
    }
    if (
        packed_device.device != model_device
        or target_device.device != model_device
        or attention_devices != {str(model_device)}
    ):
        raise AssertionError("Model, input, target, and attention tensors are not on one device.")

    expected_symbols = int(config.representation.expected_data_symbols)
    if prediction.shape != (1, expected_symbols) or not torch.is_complex(prediction):
        raise AssertionError(f"Unexpected output: {prediction.shape}, {prediction.dtype}.")
    if not torch.isfinite(prediction.real).all() or not torch.isfinite(prediction.imag).all():
        raise AssertionError("PI-EGNN output contains non-finite values.")
    if not torch.allclose(prediction, mmse_start, rtol=0.0, atol=1e-7):
        raise AssertionError("Zero-correction initialization is not equal to its MMSE start.")
    attention_neighborhoods = check_attention_sums(attention_trace)
    leakage_check = check_no_true_channel_inputs()

    report: dict[str, object] = {
        "training_started": False,
        "cache_path": str(cache_path),
        "cache_built_during_check": built,
        "locked_prior_iterations": iterations,
        "device": str(device),
        "model_input_target_attention_device_match": True,
        "output_shape": list(prediction.shape[1:]),
        "output_dtype": str(prediction.dtype),
        "all_output_values_finite": True,
        "attention_neighborhoods_checked": attention_neighborhoods,
        "attention_sums_to_one": True,
        "initial_output_equals_mmse": True,
        "h_dd_input_scan": leakage_check,
    }

    if args.benchmark_solvers:
        first_row = metadata.iloc[0]
        with np.load(raw_dir / str(first_row["file"]), allow_pickle=False) as sample:
            h_hat = np.asarray(sample["h_hat"])
        noise_power = 10.0 ** (-float(first_row["snr_db"]) / 10.0)
        report["solver_benchmark"] = benchmark_solvers(
            h_hat, noise_power, max(3, args.solver_repeats)
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
