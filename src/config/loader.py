"""
Central configuration loader for the BTP research pipeline.

This module provides a single interface for loading experiment
configuration files.

All experiment-level parameters should originate from the
configuration file rather than being duplicated throughout
individual scripts.
"""

from pathlib import Path
from typing import Any

import yaml



# Configuration object

class Config:
    """
    Lightweight configuration wrapper.

    Supports both dictionary-style and attribute-style access.
    Nested dictionaries are automatically converted to Config
    objects.
    """

    def __init__(self, values: dict[str, Any]) -> None:

        for key, value in values.items():

            if isinstance(value, dict):

                value = Config(value)

            elif isinstance(value, list):

                value = [
                    Config(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]

            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""

        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration back to a dictionary."""

        result = {}

        for key, value in self.__dict__.items():

            if isinstance(value, Config):

                result[key] = value.to_dict()

            elif isinstance(value, list):

                result[key] = [
                    item.to_dict()
                    if isinstance(item, Config)
                    else item
                    for item in value
                ]

            else:

                result[key] = value

        return result



# Validation

def validate_config(config: Config) -> None:
    """
    Validate the core experiment configuration.

    This catches configuration mistakes before an experiment
    starts producing data.
    """

    # Split ratios must form a complete partition.

    split_total = (
        config.split.train_ratio
        + config.split.validation_ratio
        + config.split.test_ratio
    )

    if not abs(split_total - 1.0) < 1e-9:

        raise ValueError(
            "Train/validation/test ratios must sum to 1.0."
        )

    # Grid dimensions must be positive.

    if config.otfs.M <= 0 or config.otfs.N <= 0:

        raise ValueError(
            "OTFS grid dimensions must be positive."
        )

    # Dataset size must be positive.

    if config.dataset.expected_samples <= 0:

        raise ValueError(
            "Expected dataset size must be positive."
        )

    # Number of paths must be positive.

    if config.channel.num_paths <= 0:

        raise ValueError(
            "Number of channel paths must be positive."
        )

    # SNR values must be numeric.

    if not config.channel.snr_db:

        raise ValueError(
            "At least one SNR operating point is required."
        )

    # Velocity values must be numeric.

    if not config.channel.velocity_kmh:

        raise ValueError(
            "At least one velocity operating point is required."
        )

        # DNN architecture must be valid.

    if not config.dnn.hidden_dims:

        raise ValueError(
            "DNN must contain at least one hidden layer."
        )

    if any(
        hidden_dim <= 0
        for hidden_dim in config.dnn.hidden_dims
    ):

        raise ValueError(
            "DNN hidden dimensions must be positive."
        )

    # Dropout must be within the valid range.

    if not 0.0 <= config.dnn.dropout < 1.0:

        raise ValueError(
            "DNN dropout must satisfy 0 <= dropout < 1."
        )

    # Training parameters must be valid.

    if config.dnn.training.epochs <= 0:

        raise ValueError(
            "DNN training epochs must be positive."
        )

    if config.dnn.training.batch_size <= 0:

        raise ValueError(
            "DNN batch size must be positive."
        )

    if config.dnn.training.learning_rate <= 0:

        raise ValueError(
            "DNN learning rate must be positive."
        )

    if config.dnn.training.weight_decay < 0:

        raise ValueError(
            "DNN weight decay cannot be negative."
        )

    if config.dnn.training.gradient_clip_norm <= 0:

        raise ValueError(
            "DNN gradient clip norm must be positive."
        )

    if config.dnn.training.early_stopping_patience <= 0:

        raise ValueError(
            "DNN early stopping patience must be positive."
        )

    if config.dnn.training.early_stopping_min_delta < 0:

        raise ValueError(
            "DNN early stopping minimum delta cannot be negative."
        )

    if not 0.0 < config.dnn.training.scheduler_factor < 1.0:

        raise ValueError(
            "DNN scheduler factor must satisfy 0 < factor < 1."
        )

    if config.dnn.training.scheduler_patience <= 0:

        raise ValueError(
            "DNN scheduler patience must be positive."
        )


# Loader

def load_config(
    config_path: str | Path,
) -> Config:
    """
    Load and validate an experiment configuration.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Validated configuration object.
    """

    config_path = Path(config_path)

    if not config_path.exists():

        raise FileNotFoundError(
            f"Configuration file not found:\n{config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        values = yaml.safe_load(file)

    if not isinstance(values, dict):

        raise ValueError(
            "Configuration file must contain a YAML mapping."
        )

    config = Config(values)

    validate_config(config)

    return config