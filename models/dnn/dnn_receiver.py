"""Fully connected neural receiver for OTFS symbol recovery."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DNNReceiver(nn.Module):
    """Recover transmitted OTFS symbols from the received DD grid."""

    def __init__(
        self,
        input_shape: Sequence[int],
        output_symbols: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        input_shape = tuple(int(value) for value in input_shape)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        output_symbols = int(output_symbols)
        dropout = float(dropout)

        if len(input_shape) != 2:
            raise ValueError(
                "input_shape must contain exactly two dimensions."
            )

        if any(value <= 0 for value in input_shape):
            raise ValueError(
                "All input dimensions must be positive."
            )

        if output_symbols <= 0:
            raise ValueError(
                "output_symbols must be positive."
            )

        if not hidden_dims:
            raise ValueError(
                "hidden_dims must contain at least one layer."
            )

        if any(value <= 0 for value in hidden_dims):
            raise ValueError(
                "All hidden dimensions must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must satisfy 0 <= dropout < 1."
            )

        self.input_shape = input_shape
        self.output_symbols = output_symbols
        self.input_features = (
            input_shape[0] * input_shape[1] * 2
        )
        self.output_features = output_symbols * 2

        layers: list[nn.Module] = []

        previous_features = self.input_features

        for hidden_features in hidden_dims:
            layers.extend(
                [
                    nn.Linear(
                        previous_features,
                        hidden_features,
                    ),
                    nn.LayerNorm(hidden_features),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

            previous_features = hidden_features

        layers.append(
            nn.Linear(
                previous_features,
                self.output_features,
            )
        )

        self.network = nn.Sequential(*layers)

    def _validate_input(
        self,
        rx_dd: torch.Tensor,
    ) -> None:
        """Validate the received DD grid before inference."""

        if not torch.is_complex(rx_dd):
            raise TypeError(
                "rx_dd must be a complex-valued tensor."
            )

        if rx_dd.ndim not in (2, 3):
            raise ValueError(
                "rx_dd must have shape (M, N) or (B, M, N)."
            )

        if tuple(rx_dd.shape[-2:]) != self.input_shape:
            raise ValueError(
                "Unexpected rx_dd shape. "
                f"Expected trailing dimensions {self.input_shape}, "
                f"got {tuple(rx_dd.shape[-2:])}."
            )

        if not torch.isfinite(rx_dd.real).all():
            raise ValueError(
                "rx_dd contains non-finite real values."
            )

        if not torch.isfinite(rx_dd.imag).all():
            raise ValueError(
                "rx_dd contains non-finite imaginary values."
            )

    def _encode_input(
        self,
        rx_dd: torch.Tensor,
    ) -> torch.Tensor:
        """Convert complex DD samples into real-valued features."""

        real = rx_dd.real.reshape(
            *rx_dd.shape[:-2],
            -1,
        )

        imag = rx_dd.imag.reshape(
            *rx_dd.shape[:-2],
            -1,
        )

        return torch.cat(
            (real, imag),
            dim=-1,
        )

    def _decode_output(
        self,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Convert real-valued network output into complex symbols."""

        real, imag = torch.chunk(
            output,
            chunks=2,
            dim=-1,
        )

        return torch.complex(
            real,
            imag,
        )

    def forward(
        self,
        rx_dd: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the transmitted complex-valued DD data symbols."""

        self._validate_input(rx_dd)

        features = self._encode_input(rx_dd)
        output = self.network(features)

        return self._decode_output(output)