"""Bipartite message-passing GNN for OTFS detection."""

from __future__ import annotations

import torch
from torch import nn


class OTFSGNN(nn.Module):
    """Process received-observation and transmitted-symbol nodes."""

    def __init__(
        self,
        observation_count: int,
        symbol_count: int,
        hidden_features: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.observation_count = int(observation_count)
        self.symbol_count = int(symbol_count)
        self.hidden_features = int(hidden_features)
        self.layers = int(layers)

        if min(
            self.observation_count,
            self.symbol_count,
            self.hidden_features,
            self.layers,
        ) <= 0:
            raise ValueError("GNN dimensions must be positive.")

        self.observation_encoder = nn.Linear(4, hidden_features)
        self.symbol_encoder = nn.Linear(4, hidden_features)
        self.symbol_to_observation = nn.ModuleList()
        self.observation_update = nn.ModuleList()
        self.observation_to_symbol = nn.ModuleList()
        self.symbol_update = nn.ModuleList()

        for _ in range(self.layers):
            self.symbol_to_observation.append(
                nn.Sequential(
                    nn.Linear(hidden_features + 2, hidden_features),
                    nn.GELU(),
                    nn.Linear(hidden_features, hidden_features),
                )
            )
            self.observation_update.append(
                nn.Sequential(
                    nn.Linear(hidden_features * 2 + 4, hidden_features),
                    nn.LayerNorm(hidden_features),
                    nn.GELU(),
                    nn.Linear(hidden_features, hidden_features),
                )
            )
            self.observation_to_symbol.append(
                nn.Sequential(
                    nn.Linear(hidden_features + 2, hidden_features),
                    nn.GELU(),
                    nn.Linear(hidden_features, hidden_features),
                )
            )
            self.symbol_update.append(
                nn.Sequential(
                    nn.Linear(hidden_features * 2 + 4, hidden_features),
                    nn.LayerNorm(hidden_features),
                    nn.GELU(),
                    nn.Linear(hidden_features, hidden_features),
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 2),
        )

    def _unpack(self, packed: torch.Tensor):
        observation_features = self.observation_count * 4
        symbol_features = self.symbol_count * 4
        edge_features = self.observation_count * self.symbol_count * 2
        mask_features = self.observation_count * self.symbol_count
        expected = (
            observation_features
            + symbol_features
            + edge_features
            + mask_features
        )
        if packed.ndim != 2 or packed.shape[-1] != expected:
            raise ValueError(
                f"Unexpected packed graph shape: {tuple(packed.shape)}; "
                f"expected feature count {expected}."
            )

        offset = 0
        observation = packed[:, offset:offset + observation_features].reshape(
            -1, self.observation_count, 4
        )
        offset += observation_features
        symbols = packed[:, offset:offset + symbol_features].reshape(
            -1, self.symbol_count, 4
        )
        offset += symbol_features
        edges = packed[:, offset:offset + edge_features].reshape(
            -1, self.observation_count, self.symbol_count, 2
        )
        offset += edge_features
        mask = packed[:, offset:offset + mask_features].reshape(
            -1, self.observation_count, self.symbol_count
        )
        return observation, symbols, edges, mask

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        observation_input, symbol_input, edges, mask = self._unpack(packed)
        observation_state = self.observation_encoder(observation_input)
        symbol_state = self.symbol_encoder(symbol_input)
        for layer in range(self.layers):
            symbol_messages = self.symbol_to_observation[layer](
                torch.cat(
                    (
                        symbol_state.unsqueeze(1).expand(
                            -1,
                            self.observation_count,
                            -1,
                            -1,
                        ),
                        edges,
                    ),
                    dim=-1,
                )
            )
            observation_messages = (
                symbol_messages * mask.unsqueeze(-1)
            ).sum(dim=2)
            observation_messages = observation_messages / mask.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
            observation_state = observation_state + self.observation_update[layer](
                torch.cat(
                    (
                        observation_state,
                        observation_messages,
                        observation_input,
                    ),
                    dim=-1,
                )
            )

            observation_messages = self.observation_to_symbol[layer](
                torch.cat(
                    (
                        observation_state.unsqueeze(2).expand(
                            -1,
                            -1,
                            self.symbol_count,
                            -1,
                        ),
                        edges,
                    ),
                    dim=-1,
                )
            )
            symbol_messages = (
                observation_messages * mask.unsqueeze(-1)
            ).sum(dim=1)
            symbol_messages = symbol_messages / mask.sum(
                dim=1,
                keepdim=True,
            ).transpose(1, 2).clamp_min(1.0)
            symbol_state = symbol_state + self.symbol_update[layer](
                torch.cat(
                    (
                        symbol_state,
                        symbol_messages,
                        symbol_input,
                    ),
                    dim=-1,
                )
            )

        output = self.readout(symbol_state)
        return torch.complex(output[..., 0], output[..., 1])
