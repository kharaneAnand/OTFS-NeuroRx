"""Unfolded OAMP-DL detector for the OTFS data-domain system."""

from __future__ import annotations

import torch
from torch import nn


class OAMPDLDetector(nn.Module):
    """Learn scalar controls for a model-driven OAMP iteration stack."""

    def __init__(
        self,
        observation_count: int,
        symbol_count: int,
        iterations: int,
    ) -> None:
        super().__init__()

        self.observation_count = int(observation_count)
        self.symbol_count = int(symbol_count)
        self.iterations = int(iterations)

        if self.observation_count <= 0 or self.symbol_count <= 0:
            raise ValueError("System dimensions must be positive.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")

        self.step_logits = nn.Parameter(
            torch.zeros(self.iterations)
        )
        self.damping_logits = nn.Parameter(
            torch.zeros(self.iterations)
        )
        self.variance_logits = nn.Parameter(
            torch.zeros(self.iterations)
        )

    def _unpack(
        self,
        packed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if packed.ndim != 2:
            raise ValueError("packed input must have shape (B, F).")

        matrix_size = self.observation_count * self.symbol_count
        expected_features = self.observation_count + matrix_size + 1
        if packed.shape[-1] != expected_features:
            raise ValueError(
                f"Unexpected packed feature count: {packed.shape[-1]} "
                f"!= {expected_features}."
            )

        y = packed[:, :self.observation_count]
        h = packed[
            :, self.observation_count:self.observation_count + matrix_size
        ].reshape(
            -1,
            self.observation_count,
            self.symbol_count,
        )
        noise_power = packed[:, -1].real.abs().clamp_min(1e-8)
        return y, h, noise_power

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        y, h, noise_power = self._unpack(packed)
        h_h = h.conj().transpose(-2, -1)
        column_power = (h.abs() ** 2).sum(dim=-2).clamp_min(1e-8)
        estimate = torch.zeros(
            packed.shape[0],
            self.symbol_count,
            dtype=packed.dtype,
            device=packed.device,
        )

        for iteration in range(self.iterations):
            step = torch.nn.functional.softplus(
                self.step_logits[iteration]
            )
            damping = torch.sigmoid(
                self.damping_logits[iteration]
            )
            variance_scale = torch.nn.functional.softplus(
                self.variance_logits[iteration]
            )

            residual = y - torch.bmm(h, estimate.unsqueeze(-1)).squeeze(-1)
            matched_update = torch.bmm(
                h_h,
                residual.unsqueeze(-1),
            ).squeeze(-1) / column_power
            linear_estimate = estimate + step * matched_update
            effective_variance = (
                noise_power * variance_scale
                + torch.mean(torch.abs(linear_estimate - estimate) ** 2, dim=-1)
            ).clamp_min(1e-8)
            denoised_real = torch.tanh(
                linear_estimate.real / effective_variance.unsqueeze(-1)
            ) / torch.sqrt(torch.tensor(2.0, device=packed.device))
            denoised_imag = torch.tanh(
                linear_estimate.imag / effective_variance.unsqueeze(-1)
            ) / torch.sqrt(torch.tensor(2.0, device=packed.device))
            denoised = torch.complex(denoised_real, denoised_imag)
            estimate = (
                (1.0 - damping) * estimate
                + damping * denoised
            )

        return estimate