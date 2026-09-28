"""Prior-information-enhanced bipartite GNN for OTFS detection."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class OAMPTraceStep:
    linear_estimate: np.ndarray
    estimate: np.ndarray
    effective_variance: float
    elapsed_seconds: float
    linear_solve_seconds: float
    operator_setup_seconds: float
    iterations_seconds: float


def oamp_prior_trace(
    y_data: np.ndarray,
    h_hat: np.ndarray,
    noise_power: float,
    iterations: int = 3,
) -> list[OAMPTraceStep]:
    """Return analytical OAMP estimates and scalar variances at each step."""

    y_data = np.asarray(y_data)
    h_hat = np.asarray(h_hat)
    if y_data.ndim != 1 or h_hat.ndim != 2:
        raise ValueError("Expected 1-D y_data and 2-D h_hat.")
    if h_hat.shape[0] != y_data.shape[0]:
        raise ValueError("h_hat rows must match y_data length.")
    if iterations <= 0 or noise_power < 0:
        raise ValueError("iterations must be positive and noise non-negative.")

    start = time.perf_counter()
    symbol_count = h_hat.shape[1]
    identity = np.eye(symbol_count, dtype=h_hat.dtype)
    gram = h_hat.conj().T @ h_hat
    right_hand_side = h_hat.conj().T
    solve_start = time.perf_counter()
    inverse = np.linalg.solve(
        gram + noise_power * identity,
        right_hand_side,
    )
    linear_solve_seconds = time.perf_counter() - solve_start
    orthogonalization = symbol_count / np.trace(inverse @ h_hat).real
    linear_operator = orthogonalization * inverse
    propagated_noise = noise_power * np.mean(
        np.abs((linear_operator @ linear_operator.conj().T).real)
    )

    estimate = np.zeros(symbol_count, dtype=h_hat.dtype)
    epsilon = np.finfo(float).eps
    trace: list[OAMPTraceStep] = []
    operator_setup_seconds = time.perf_counter() - start - linear_solve_seconds
    iterations_start = time.perf_counter()

    for _ in range(iterations):
        residual = y_data - h_hat @ estimate
        linear_estimate = estimate + linear_operator @ residual
        effective_variance = max(
            float(np.mean(np.abs(linear_estimate - estimate) ** 2) + propagated_noise),
            epsilon,
        )
        scale = 1.0 / np.sqrt(2.0)
        estimate = (
            scale * np.tanh(np.sqrt(2.0) * linear_estimate.real / effective_variance)
            + 1j * scale * np.tanh(
                np.sqrt(2.0) * linear_estimate.imag / effective_variance
            )
        )
        trace.append(
            OAMPTraceStep(
                linear_estimate=linear_estimate.copy(),
                estimate=estimate.copy(),
                effective_variance=effective_variance,
                elapsed_seconds=time.perf_counter() - start,
                linear_solve_seconds=linear_solve_seconds,
                operator_setup_seconds=operator_setup_seconds,
                iterations_seconds=time.perf_counter() - iterations_start,
            )
        )

    return trace


def qpsk_max_posterior(
    linear_estimate: np.ndarray,
    effective_variance: float,
) -> np.ndarray:
    """Maximum QPSK posterior mass under complex Gaussian variance v."""

    scale = 1.0 / np.sqrt(2.0)
    constellation = np.asarray(
        [scale * (real + 1j * imag) for real in (-1, 1) for imag in (-1, 1)]
    )
    distances_squared = np.abs(
        np.asarray(linear_estimate)[..., None] - constellation
    ) ** 2
    logits = -distances_squared / max(
        float(effective_variance), np.finfo(float).eps
    )
    logits -= np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.max(probabilities, axis=-1)


class PIEGNN(nn.Module):
    """Bipartite GNN with OAMP priors and active-edge attention."""

    observation_feature_count = 4
    symbol_feature_count = 10

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
        if min(self.observation_count, self.symbol_count, self.hidden_features, self.layers) <= 0:
            raise ValueError("PI-EGNN dimensions must be positive.")

        hidden = self.hidden_features
        self.observation_encoder = nn.Linear(self.observation_feature_count, hidden)
        self.symbol_encoder = nn.Linear(self.symbol_feature_count, hidden)
        self.symbol_to_observation_attention = nn.ModuleList()
        self.observation_to_symbol_attention = nn.ModuleList()
        self.symbol_to_observation_message = nn.ModuleList()
        self.observation_update = nn.ModuleList()
        self.observation_to_symbol_message = nn.ModuleList()
        self.symbol_update = nn.ModuleList()

        for _ in range(self.layers):
            self.symbol_to_observation_attention.append(self._attention_network(hidden))
            self.observation_to_symbol_attention.append(self._attention_network(hidden))
            self.symbol_to_observation_message.append(self._message_network(hidden))
            self.observation_update.append(
                self._update_network(hidden, self.observation_feature_count)
            )
            self.observation_to_symbol_message.append(self._message_network(hidden))
            self.symbol_update.append(
                self._update_network(hidden, self.symbol_feature_count)
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    @staticmethod
    def _attention_network(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _message_network(hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    @staticmethod
    def _update_network(hidden: int, node_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden * 2 + node_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def _unpack(self, packed: torch.Tensor):
        observation_features = self.observation_count * self.observation_feature_count
        symbol_features = self.symbol_count * self.symbol_feature_count
        edge_features = self.observation_count * self.symbol_count * 2
        mask_features = self.observation_count * self.symbol_count
        expected = observation_features + symbol_features + edge_features + mask_features
        if packed.ndim != 2 or packed.shape[-1] != expected:
            raise ValueError(
                f"Unexpected packed PI-EGNN shape {tuple(packed.shape)}; "
                f"expected {expected} features."
            )

        offset = 0
        observation = packed[:, :observation_features].reshape(
            -1, self.observation_count, self.observation_feature_count
        )
        offset += observation_features
        symbols = packed[:, offset:offset + symbol_features].reshape(
            -1, self.symbol_count, self.symbol_feature_count
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

    @staticmethod
    def _segment_softmax(
        logits: torch.Tensor,
        segment_ids: torch.Tensor,
        segment_count: int,
    ) -> torch.Tensor:
        maxima = torch.full(
            (segment_count,),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        maxima.scatter_reduce_(
            0, segment_ids, logits, reduce="amax", include_self=True
        )
        exponentials = torch.exp(logits - maxima[segment_ids])
        denominators = torch.zeros_like(maxima).index_add(
            0, segment_ids, exponentials
        )
        return exponentials / denominators[segment_ids].clamp_min(1e-8)

    def forward(
        self,
        packed: torch.Tensor,
        return_attention: bool = False,
    ):
        observation_input, symbol_input, edges, mask = self._unpack(packed)
        observation_state = self.observation_encoder(observation_input)
        symbol_state = self.symbol_encoder(symbol_input)
        batch_indices, observation_indices, symbol_indices = torch.nonzero(
            mask > 0, as_tuple=True
        )
        batch_size = packed.shape[0]
        edge_count = batch_indices.numel()
        edge_features = edges[
            batch_indices, observation_indices, symbol_indices
        ]
        observation_segment = batch_indices * self.observation_count + observation_indices
        symbol_segment = batch_indices * self.symbol_count + symbol_indices
        attention_trace: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
            "symbol_to_observation": [],
            "observation_to_symbol": [],
        }

        for layer in range(self.layers):
            edge_context = torch.cat(
                (
                    observation_state[batch_indices, observation_indices],
                    symbol_state[batch_indices, symbol_indices],
                    edge_features,
                ),
                dim=-1,
            )
            symbol_to_observation_weights = self._segment_softmax(
                self.symbol_to_observation_attention[layer](edge_context).squeeze(-1),
                observation_segment,
                batch_size * self.observation_count,
            )
            if return_attention:
                attention_trace["symbol_to_observation"].append(
                    (symbol_to_observation_weights, observation_segment)
                )
            symbol_messages = self.symbol_to_observation_message[layer](
                torch.cat(
                    (
                        symbol_state[batch_indices, symbol_indices],
                        edge_features,
                    ),
                    dim=-1,
                )
            )
            aggregated = torch.zeros(
                batch_size * self.observation_count,
                self.hidden_features,
                dtype=symbol_messages.dtype,
                device=symbol_messages.device,
            ).index_add(
                0,
                observation_segment,
                symbol_messages * symbol_to_observation_weights.unsqueeze(-1),
            ).reshape(batch_size, self.observation_count, self.hidden_features)
            observation_state = observation_state + self.observation_update[layer](
                torch.cat((observation_state, aggregated, observation_input), dim=-1)
            )

            edge_context = torch.cat(
                (
                    observation_state[batch_indices, observation_indices],
                    symbol_state[batch_indices, symbol_indices],
                    edge_features,
                ),
                dim=-1,
            )
            observation_to_symbol_weights = self._segment_softmax(
                self.observation_to_symbol_attention[layer](edge_context).squeeze(-1),
                symbol_segment,
                batch_size * self.symbol_count,
            )
            if return_attention:
                attention_trace["observation_to_symbol"].append(
                    (observation_to_symbol_weights, symbol_segment)
                )
            observation_messages = self.observation_to_symbol_message[layer](
                torch.cat(
                    (
                        observation_state[batch_indices, observation_indices],
                        edge_features,
                    ),
                    dim=-1,
                )
            )
            aggregated = torch.zeros(
                batch_size * self.symbol_count,
                self.hidden_features,
                dtype=observation_messages.dtype,
                device=observation_messages.device,
            ).index_add(
                0,
                symbol_segment,
                observation_messages * observation_to_symbol_weights.unsqueeze(-1),
            ).reshape(batch_size, self.symbol_count, self.hidden_features)
            symbol_state = symbol_state + self.symbol_update[layer](
                torch.cat((symbol_state, aggregated, symbol_input), dim=-1)
            )

        correction = self.readout(symbol_state)
        initial = symbol_input[..., :2]
        output = initial + correction
        complex_output = torch.complex(output[..., 0], output[..., 1])
        if return_attention:
            return complex_output, attention_trace
        return complex_output