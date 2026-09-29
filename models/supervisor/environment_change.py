"""Anytime-valid, fixed-reference environment shift monitoring for OTFS frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from models.supervisor.environment_features import FEATURE_NAMES


RANK_FEATURE_NAMES = ("order", "dispersion")


def make_regime_key(snr_db: int | float, velocity_kmh: int | float) -> str:
    """Build a stable context key from independently known operating metadata."""

    return f"snr={float(snr_db):g}|velocity={float(velocity_kmh):g}"


class _RankFeatureBet:
    def __init__(self, name: str, rank_count: int) -> None:
        self.name = name
        if name == "order":
            self.values = np.arange(rank_count, dtype=np.float64) / (rank_count - 1) - 0.5
        elif name == "dispersion":
            self.values = np.abs(
                2.0 * np.arange(rank_count, dtype=np.float64) / (rank_count - 1) - 1.0
            ) - 0.5
        else:
            raise ValueError(f"Unknown rank feature: {name}")
        self.log_wealth = 0.0
        self.lambda_value = 0.0
        self.gradient_square_sum = 1.0

    def update(
        self,
        rank_index: int,
        predictive_probabilities: np.ndarray,
        next_probabilities: np.ndarray,
    ) -> float:
        predictive_mean = float(np.dot(predictive_probabilities, self.values))
        payoff = float(self.values[rank_index] - predictive_mean)
        factor = 1.0 + self.lambda_value * payoff
        if factor <= 0.0:
            raise FloatingPointError("PRM betting factor must remain positive.")
        self.log_wealth += float(np.log(factor))
        gradient = payoff / factor
        self.gradient_square_sum += gradient * gradient
        next_mean = float(np.dot(next_probabilities, self.values))
        bound = float(np.max(np.abs(self.values - next_mean)))
        radius = min(1.0, 0.75 / bound) if bound > 0.0 else 1.0
        self.lambda_value = float(
            np.clip(
                self.lambda_value + 4.5 * gradient / self.gradient_square_sum,
                -radius,
                radius,
            )
        )
        return self.log_wealth


class _PredictiveRankMonitor:
    """PRM ONS monitor for one scalar score and one fixed reference sample."""

    def __init__(
        self,
        reference_scores: np.ndarray,
        alpha: float,
        rng: np.random.Generator,
    ) -> None:
        reference_scores = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
        if reference_scores.size < 2:
            raise ValueError("Each PRM reference needs at least two frames.")
        if not np.isfinite(reference_scores).all():
            raise ValueError("PRM reference contains non-finite scores.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one.")

        self.reference_count = int(reference_scores.size)
        self.reference_scores = reference_scores.copy()
        self.reference_tie_breakers = rng.random(self.reference_count)
        order = np.lexsort((self.reference_tie_breakers, self.reference_scores))
        self.sorted_scores = self.reference_scores[order]
        self.sorted_tie_breakers = self.reference_tie_breakers[order]
        self.counts = np.zeros(self.reference_count + 1, dtype=np.int64)
        self.order_bet = _RankFeatureBet("order", self.reference_count + 1)
        self.dispersion_bet = _RankFeatureBet("dispersion", self.reference_count + 1)
        self.alpha = float(alpha)
        self.threshold = 1.0 / self.alpha
        self.rng = rng
        self.observations = 0
        self.alarmed = False

    def _rank_index(self, score: float) -> int:
        tie_breaker = float(self.rng.random())
        left = int(np.searchsorted(self.sorted_scores, score, side="left"))
        right = int(np.searchsorted(self.sorted_scores, score, side="right"))
        equal_ties = self.sorted_tie_breakers[left:right]
        return left + int(np.searchsorted(equal_ties, tie_breaker, side="left"))

    def update(self, score: float) -> dict[str, float | bool | int]:
        if not np.isfinite(score):
            raise ValueError("Incoming environment score must be finite.")
        rank_index = self._rank_index(float(score))
        predictive_probabilities = (1.0 + self.counts) / (
            self.reference_count + self.observations + 1.0
        )
        next_counts = self.counts.copy()
        next_counts[rank_index] += 1
        next_probabilities = (1.0 + next_counts) / (
            self.reference_count + self.observations + 2.0
        )
        order_log_wealth = self.order_bet.update(
            rank_index, predictive_probabilities, next_probabilities
        )
        dispersion_log_wealth = self.dispersion_bet.update(
            rank_index, predictive_probabilities, next_probabilities
        )
        self.counts = next_counts
        self.observations += 1
        log_portfolio_wealth = float(
            np.logaddexp(order_log_wealth, dispersion_log_wealth) - np.log(2.0)
        )
        self.alarmed = self.alarmed or log_portfolio_wealth >= np.log(self.threshold)
        return {
            "rank": rank_index + 1,
            "order_log_wealth": order_log_wealth,
            "dispersion_log_wealth": dispersion_log_wealth,
            "portfolio_log_wealth": log_portfolio_wealth,
            "log_threshold": float(np.log(self.threshold)),
            "alarm": self.alarmed,
            "alarm_latched": self.alarmed,
            "observations": self.observations,
        }


@dataclass(frozen=True)
class EnvironmentDecision:
    status: str
    reference_id: str
    known_context_change: bool
    unsupported_context: bool
    alarmed_scores: tuple[str, ...]
    evidence: dict[str, dict[str, float | bool | int]]
    false_alarm_scope: str


class EnvironmentChangeDetector:
    """Monitor one active known context or a pooled operating distribution.

    References are independent training-split frames. A regime ID must be
    supplied by trusted metadata; unknown/inferred IDs use the pooled reference.
    """

    def __init__(
        self,
        references: dict[str, dict[str, np.ndarray]],
        alpha: float = 0.05,
        seed: int | None = None,
        score_names: tuple[str, ...] = FEATURE_NAMES,
        familywise_contexts: bool = True,
    ) -> None:
        if not references:
            raise ValueError("At least one environment reference is required.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one.")
        self.references = references
        self.alpha = float(alpha)
        self.score_names = tuple(score_names)
        self.familywise_contexts = bool(familywise_contexts)
        self.rng = np.random.default_rng(seed)
        self.context_count = len(references)
        context_divisor = self.context_count if self.familywise_contexts else 1
        self.alpha_per_monitor = self.alpha / (context_divisor * len(self.score_names))
        self.monitors: dict[str, dict[str, _PredictiveRankMonitor]] = {}
        for reference_id, score_arrays in references.items():
            missing = set(self.score_names).difference(score_arrays)
            if missing:
                raise ValueError(
                    f"Reference {reference_id!r} lacks scores: {sorted(missing)}"
                )
            self.monitors[reference_id] = {
                score_name: _PredictiveRankMonitor(
                    np.asarray(score_arrays[score_name]),
                    self.alpha_per_monitor,
                    self.rng,
                )
                for score_name in self.score_names
            }
        self.previous_context: str | None = None

    def update(
        self,
        features: dict[str, float],
        regime_id: str | None = None,
    ) -> EnvironmentDecision:
        missing = set(self.score_names).difference(features)
        if missing:
            raise ValueError(f"Incoming frame lacks scores: {sorted(missing)}")

        requested_id = regime_id or "pooled"
        unsupported_context = requested_id not in self.monitors
        reference_id = (
            "pooled"
            if unsupported_context and "pooled" in self.monitors
            else requested_id
        )
        if reference_id not in self.monitors:
            raise KeyError(f"No reference available for context {requested_id!r}.")
        known_context_change = (
            self.previous_context is not None
            and requested_id != self.previous_context
            and not unsupported_context
            and self.previous_context in self.monitors
        )

        evidence = {
            score_name: self.monitors[reference_id][score_name].update(
                float(features[score_name])
            )
            for score_name in self.score_names
        }
        alarmed = tuple(
            score_name
            for score_name, result in evidence.items()
            if bool(result["alarm"])
        )
        if alarmed:
            status = "SHIFT_DETECTED"
        elif unsupported_context:
            status = "POOLED_NO_SHIFT_EVIDENCE_UNSUPPORTED_CONTEXT"
        else:
            status = "NO_SHIFT_EVIDENCE"
        self.previous_context = requested_id
        if self.familywise_contexts:
            scope = (
                "Anytime marginal family-wise alpha over all configured references and scores, "
                "under the paper's i.i.d. frame-level null and fixed-feature assumptions."
            )
        else:
            scope = (
                "Anytime marginal alpha within each active reference, split over its scores. "
                "There is no combined false-alarm bound across context references."
            )
        scope += " Does not certify receiver accuracy or conditional validity for a realized reference."
        return EnvironmentDecision(
            status=status,
            reference_id=reference_id,
            known_context_change=known_context_change,
            unsupported_context=unsupported_context,
            alarmed_scores=alarmed,
            evidence=evidence,
            false_alarm_scope=scope,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize monitor state for process restart without retraining."""

        monitor_states = {}
        for reference_id, score_monitors in self.monitors.items():
            monitor_states[reference_id] = {}
            for score_name, monitor in score_monitors.items():
                monitor_states[reference_id][score_name] = {
                    "reference_tie_breakers": monitor.reference_tie_breakers.copy(),
                    "counts": monitor.counts.copy(),
                    "observations": monitor.observations,
                    "alarmed": monitor.alarmed,
                    "order": {
                        "log_wealth": monitor.order_bet.log_wealth,
                        "lambda_value": monitor.order_bet.lambda_value,
                        "gradient_square_sum": monitor.order_bet.gradient_square_sum,
                    },
                    "dispersion": {
                        "log_wealth": monitor.dispersion_bet.log_wealth,
                        "lambda_value": monitor.dispersion_bet.lambda_value,
                        "gradient_square_sum": monitor.dispersion_bet.gradient_square_sum,
                    },
                }
        return {
            "version": 1,
            "alpha": self.alpha,
            "score_names": self.score_names,
            "familywise_contexts": self.familywise_contexts,
            "previous_context": self.previous_context,
            "monitor_states": monitor_states,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("Unsupported detector state version.")
        if (
            float(state["alpha"]) != self.alpha
            or tuple(state["score_names"]) != self.score_names
            or bool(state["familywise_contexts"]) != self.familywise_contexts
        ):
            raise ValueError("Saved state does not match this detector configuration.")
        for reference_id, score_states in state["monitor_states"].items():
            if reference_id not in self.monitors:
                raise ValueError(f"Saved state contains unknown reference {reference_id!r}.")
            for score_name, saved in score_states.items():
                monitor = self.monitors[reference_id][score_name]
                tie_breakers = np.asarray(
                    saved["reference_tie_breakers"], dtype=np.float64
                )
                if tie_breakers.shape != monitor.reference_tie_breakers.shape:
                    raise ValueError("Saved tie-breakers have the wrong shape.")
                monitor.reference_tie_breakers = tie_breakers.copy()
                order = np.lexsort(
                    (monitor.reference_tie_breakers, monitor.reference_scores)
                )
                monitor.sorted_scores = monitor.reference_scores[order]
                monitor.sorted_tie_breakers = monitor.reference_tie_breakers[order]
                counts = np.asarray(saved["counts"], dtype=np.int64)
                if counts.shape != monitor.counts.shape:
                    raise ValueError("Saved rank-count vector has the wrong shape.")
                monitor.counts = counts.copy()
                monitor.observations = int(saved["observations"])
                monitor.alarmed = bool(saved["alarmed"])
                for bet_name, bet in (
                    ("order", monitor.order_bet),
                    ("dispersion", monitor.dispersion_bet),
                ):
                    bet_state = saved[bet_name]
                    bet.log_wealth = float(bet_state["log_wealth"])
                    bet.lambda_value = float(bet_state["lambda_value"])
                    bet.gradient_square_sum = float(
                        bet_state["gradient_square_sum"]
                    )
        self.previous_context = state.get("previous_context")
        self.rng.bit_generator.state = state["rng_state"]
