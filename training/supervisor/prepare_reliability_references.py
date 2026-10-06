"""Run frozen receivers on train/validation and fit reliability references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.supervisor.reliability_bank import collect_receiver_rows, RECEIVER_NAMES
from models.supervisor.reliability_detector import (
    RELIABILITY_FEATURE_NAMES,
    ReceiverReliabilityReference,
)
from models.supervisor.simple_environment_detector import SimpleEnvironmentReference
from src.config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_root = PROJECT_ROOT / config.dataset.root
    raw_dir = dataset_root / config.dataset.raw_dir
    metadata_path = dataset_root / config.dataset.processed_dir / config.dataset.split_metadata_file
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.loc[
        metadata["split"].astype(str).str.lower().isin({"train", "validation"})
    ].reset_index(drop=True)

    environment_reference_path = (
        PROJECT_ROOT / "experiments" / "environment_detector_simple" / "reference.json"
    )
    with environment_reference_path.open(encoding="utf-8") as file:
        environment_reference = SimpleEnvironmentReference.from_dict(json.load(file))

    rows = collect_receiver_rows(
        metadata,
        raw_dir,
        config,
        PROJECT_ROOT,
        environment_reference,
        device,
    )
    frame = pd.DataFrame(rows)
    references = {}
    for receiver in RECEIVER_NAMES:
        train_rows = frame.loc[
            (frame["receiver"] == receiver) & (frame["split"] == "train")
        ]
        feature_values = train_rows.loc[
            :, list(RELIABILITY_FEATURE_NAMES)
        ].to_numpy(dtype=float)
        references[receiver] = ReceiverReliabilityReference.fit(
            receiver, feature_values
        ).to_dict()

    output_directory = PROJECT_ROOT / "experiments" / "reliability_detector"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "references.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "reference_split": "train",
                "evaluation_split": "validation",
                "receivers": references,
                "receiver_checkpoints_reused": True,
                "retraining_performed": False,
                "live_inputs": ["receiver estimate", "environment distance"],
                "offline_only_labels": ["ber", "ser", "nmse"],
            },
            file,
            indent=2,
        )
    frame.to_csv(output_directory / "train_validation_features.csv", index=False)
    print(
        json.dumps(
            {
                "device": str(device),
                "receivers": list(RECEIVER_NAMES),
                "samples": len(metadata),
                "rows": len(frame),
                "reference_file": str(
                    (output_directory / "references.json").relative_to(PROJECT_ROOT)
                ),
                "features_file": str(
                    (output_directory / "train_validation_features.csv").relative_to(
                        PROJECT_ROOT
                    )
                ),
                "retraining_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
