"""Neural residual correction on top of an MMSE OTFS estimate."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MMSEResidualReceiver(nn.Module):
    """Predict a complex correction for an MMSE symbol estimate."""

    def __init__(
        self,
        output_symbols: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        output_symbols = int(output_symbols)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        dropout = float(dropout)

        if output_symbols <= 0:
            raise ValueError("output_symbols must be positive.")

        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive values.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.output_symbols = output_symbols
        self.input_features = output_symbols * 2
        self.output_features = output_symbols * 2

        layers: list[nn.Module] = []
        previous_features = self.input_features

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

        layers.append(
            nn.Linear(previous_features, self.output_features)
        )
        self.network = nn.Sequential(*layers)

    def forward(self, mmse_estimate: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(mmse_estimate):
            raise TypeError("mmse_estimate must be complex-valued.")

        if mmse_estimate.ndim not in (1, 2):
            raise ValueError(
                "mmse_estimate must have shape (S,) or (B, S)."
            )

        if mmse_estimate.shape[-1] != self.output_symbols:
            raise ValueError(
                "Unexpected MMSE estimate length: "
                f"{mmse_estimate.shape[-1]} != {self.output_symbols}."
            )

        if not torch.isfinite(mmse_estimate.real).all():
            raise ValueError("mmse_estimate has non-finite real values.")

        if not torch.isfinite(mmse_estimate.imag).all():
            raise ValueError("mmse_estimate has non-finite imaginary values.")

        features = torch.cat(
            (
                mmse_estimate.real,
                mmse_estimate.imag,
            ),
            dim=-1,
        )
        correction = self.network(features)
        real, imag = torch.chunk(correction, chunks=2, dim=-1)
        return torch.complex(real, imag)