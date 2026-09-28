"""Train one fixed GNN initialization ablation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.gnn.otfs_gnn import OTFSGNN
from src.config.loader import load_config
from training.dnn.trainer import DNNTrainer
from training.gnn.dataset import OTFSGraphDataset, load_split_metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--initialization",
        choices=("mmse", "neutral_zero"),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.reproducibility.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {config.reproducibility.seed}")
    print(f"Initialization: {args.initialization}")
    print(
        "Edge threshold: "
        f"{config.gnn.edge_threshold_fraction_of_max:.4f} * max(abs(H_hat))"
    )

    dataset_root = Path(config.dataset.root)
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )
    train_dataset = OTFSGraphDataset(
        load_split_metadata(metadata_path, "train"),
        raw_dir,
        config,
        args.initialization,
    )
    validation_dataset = OTFSGraphDataset(
        load_split_metadata(metadata_path, "validation"),
        raw_dir,
        config,
        args.initialization,
    )
    test_dataset = OTFSGraphDataset(
        load_split_metadata(metadata_path, "test"),
        raw_dir,
        config,
        args.initialization,
    )

    batch_size = int(config.gnn.training.batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = OTFSGNN(
        observation_count=int(
            config.representation.expected_channel_shape.rows
        ),
        symbol_count=int(
            config.representation.expected_channel_shape.cols
        ),
        hidden_features=int(config.gnn.hidden_features),
        layers=int(config.gnn.message_passing_layers),
    )

    if args.initialization == "mmse":
        output_directory = Path(config.gnn.output.mmse_directory)
    else:
        output_directory = Path(config.gnn.output.neutral_directory)

    trainer = DNNTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=int(config.gnn.training.epochs),
        learning_rate=float(config.gnn.training.learning_rate),
        weight_decay=float(config.gnn.training.weight_decay),
        gradient_clip_norm=float(config.gnn.training.gradient_clip_norm),
        early_stopping_patience=int(
            config.gnn.training.early_stopping_patience
        ),
        early_stopping_min_delta=float(
            config.gnn.training.early_stopping_min_delta
        ),
        scheduler_factor=float(config.gnn.training.scheduler_factor),
        scheduler_patience=int(config.gnn.training.scheduler_patience),
        checkpoint_path=output_directory / config.gnn.output.checkpoint_file,
        history_path=output_directory / config.gnn.output.history_file,
        device=device,
    )
    trainer.fit()

    print(
        "Best checkpoint saved to: "
        f"{output_directory / config.gnn.output.checkpoint_file}"
    )
    print(
        "Training history saved to: "
        f"{output_directory / config.gnn.output.history_file}"
    )
    print(f"Test samples available for evaluation: {len(test_dataset)}")


if __name__ == "__main__":
    main()
