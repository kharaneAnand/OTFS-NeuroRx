"""Informed neural correction model for the MMSE OTFS receiver."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class InformedResidualReceiver(nn.Module):
    """Predict a correction using MMSE, observations, residual, and channel strength."""

    def __init__(
        self,
        input_features: int,
        output_symbols: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        input_features = int(input_features)
        output_symbols = int(output_symbols)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        dropout = float(dropout)

        if input_features <= 0 or output_symbols <= 0:
            raise ValueError("Model dimensions must be positive.")
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive values.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        layers: list[nn.Module] = []
        previous_features = input_features

        for hidden_features in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_features, hidden_features),
                    nn.LayerNorm(hidden_features),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_features = hidden_features

        final_layer = nn.Linear(previous_features, output_symbols * 2)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.network = nn.Sequential(*layers)
        self.output_symbols = output_symbols
        self.input_features = input_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim not in (1, 2):
            raise ValueError("features must have shape (F,) or (B, F).")
        if features.shape[-1] != self.input_features:
            raise ValueError(
                f"Unexpected feature count: {features.shape[-1]} "
                f"!= {self.input_features}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("features contain non-finite values.")

        output = self.network(features)
        real, imag = torch.chunk(output, chunks=2, dim=-1)
        return torch.complex(real, imag)