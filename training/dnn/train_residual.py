"""Train the MMSE residual receiver without touching blind-DNN outputs."""

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

from models.dnn.mmse_residual_receiver import MMSEResidualReceiver
from src.config.loader import load_config
from training.dnn.dataset import load_split_metadata
from training.dnn.residual_dataset import MMSEResidualDataset
from training.dnn.trainer import DNNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MMSE residual OTFS receiver."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.reproducibility.seed))
    device = resolve_device()
    print(f"Device: {device}")

    dataset_root = Path(config.dataset.root)
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )

    train_metadata = load_split_metadata(metadata_path, "train")
    validation_metadata = load_split_metadata(metadata_path, "validation")
    test_metadata = load_split_metadata(metadata_path, "test")

    train_dataset = MMSEResidualDataset(train_metadata, raw_dir, config)
    validation_dataset = MMSEResidualDataset(
        validation_metadata,
        raw_dir,
        config,
    )
    test_dataset = MMSEResidualDataset(test_metadata, raw_dir, config)

    batch_size = int(config.residual.training.batch_size)
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

    model = MMSEResidualReceiver(
        output_symbols=int(config.representation.expected_data_symbols),
        hidden_dims=config.residual.hidden_dims,
        dropout=float(config.residual.dropout),
    )

    output_directory = Path(config.residual.output.directory)
    trainer = DNNTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=int(config.residual.training.epochs),
        learning_rate=float(config.residual.training.learning_rate),
        weight_decay=float(config.residual.training.weight_decay),
        gradient_clip_norm=float(config.residual.training.gradient_clip_norm),
        early_stopping_patience=int(
            config.residual.training.early_stopping_patience
        ),
        early_stopping_min_delta=float(
            config.residual.training.early_stopping_min_delta
        ),
        scheduler_factor=float(config.residual.training.scheduler_factor),
        scheduler_patience=int(config.residual.training.scheduler_patience),
        checkpoint_path=output_directory / config.residual.output.checkpoint_file,
        history_path=output_directory / config.residual.output.history_file,
        device=device,
    )

    trainer.fit()

    print(f"Best checkpoint saved to: {output_directory / config.residual.output.checkpoint_file}")
    print(f"Training history saved to: {output_directory / config.residual.output.history_file}")
    print(f"Test samples available for evaluation: {len(test_dataset)}")


if __name__ == "__main__":
    main()