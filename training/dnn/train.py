"""Production entry point for training the OTFS DNN receiver."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.loader import Config, load_config
from models.dnn.dnn_receiver import DNNReceiver
from training.dnn.dataset import (
    OTFSReceiverDataset,
    load_split_metadata,
)
from training.dnn.trainer import DNNTrainer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train the DNN OTFS receiver."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment YAML configuration.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Select the available computation device."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_datasets(
    config: Config,
) -> tuple[
    OTFSReceiverDataset,
    OTFSReceiverDataset,
    OTFSReceiverDataset,
]:
    """Build the predefined train, validation, and test datasets."""

    dataset_root = (
        Path(config.dataset.root)
    )

    raw_dir = (
        dataset_root
        / config.dataset.raw_dir
    )

    metadata_path = (
        dataset_root
        / config.dataset.processed_dir
        / config.dataset.split_metadata_file
    )

    input_shape = (
        int(config.representation.expected_grid_shape.rows),
        int(config.representation.expected_grid_shape.cols),
    )

    output_symbols = int(
        config.representation.expected_data_symbols
    )

    train_metadata = load_split_metadata(
        metadata_path,
        "train",
    )

    validation_metadata = load_split_metadata(
        metadata_path,
        "validation",
    )

    test_metadata = load_split_metadata(
        metadata_path,
        "test",
    )

    train_dataset = OTFSReceiverDataset(
        metadata=train_metadata,
        raw_dir=raw_dir,
        input_shape=input_shape,
        output_symbols=output_symbols,
    )

    validation_dataset = OTFSReceiverDataset(
        metadata=validation_metadata,
        raw_dir=raw_dir,
        input_shape=input_shape,
        output_symbols=output_symbols,
    )

    test_dataset = OTFSReceiverDataset(
        metadata=test_metadata,
        raw_dir=raw_dir,
        input_shape=input_shape,
        output_symbols=output_symbols,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def build_model(config: Config) -> DNNReceiver:
    """Build the configured DNN receiver."""

    input_shape = (
        int(config.representation.expected_grid_shape.rows),
        int(config.representation.expected_grid_shape.cols),
    )

    output_symbols = int(
        config.representation.expected_data_symbols
    )

    hidden_dims = [
        int(value)
        for value in config.dnn.hidden_dims
    ]

    return DNNReceiver(
        input_shape=input_shape,
        output_symbols=output_symbols,
        hidden_dims=hidden_dims,
        dropout=float(config.dnn.dropout),
    )


def main() -> None:
    """Run the complete DNN training pipeline."""

    args = parse_args()

    config = load_config(args.config)

    set_seed(
        int(config.reproducibility.seed)
    )

    device = resolve_device()

    print(f"Device: {device}")

    train_dataset, validation_dataset, test_dataset = (
        build_datasets(config)
    )

    print(
        "Dataset sizes: "
        f"train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, "
        f"test={len(test_dataset)}"
    )

    batch_size = int(
        config.dnn.training.batch_size
    )

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

    model = build_model(config)

    output_directory = Path(
        config.dnn.output.directory
    )

    checkpoint_path = (
        output_directory
        / config.dnn.output.checkpoint_file
    )

    history_path = (
        output_directory
        / config.dnn.output.history_file
    )

    trainer = DNNTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=int(
            config.dnn.training.epochs
        ),
        learning_rate=float(
            config.dnn.training.learning_rate
        ),
        weight_decay=float(
            config.dnn.training.weight_decay
        ),
        gradient_clip_norm=float(
            config.dnn.training.gradient_clip_norm
        ),
        early_stopping_patience=int(
            config.dnn.training.early_stopping_patience
        ),
        early_stopping_min_delta=float(
            config.dnn.training.early_stopping_min_delta
        ),
        scheduler_factor=float(
            config.dnn.training.scheduler_factor
        ),
        scheduler_patience=int(
            config.dnn.training.scheduler_patience
        ),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        device=device,
    )

    trainer.fit()

    print(
        f"Best checkpoint saved to: {checkpoint_path}"
    )

    print(
        f"Training history saved to: {history_path}"
    )

    print(
        f"Test samples available for evaluation: "
        f"{len(test_dataset)}"
    )


if __name__ == "__main__":
    main()