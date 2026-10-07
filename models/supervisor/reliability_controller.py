"""Rule-based active-receiver controller using confidence margin only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Keep the bank names local if the reliability module is used independently.
RECEIVERS = ("mmse", "oamp_dl", "original_gnn", "pi_egnn")


@dataclass(frozen=True)
class ControllerThresholds:
    low_z_threshold: float = -1.0
    switch_z_gap: float = 0.5
    consecutive_low_frames: int = 2
    minimum_frames_between_switches: int = 10


@dataclass(frozen=True)
class ControllerDecision:
    frame_index: int
    active_before: str
    active_after: str
    decision: str
    active_margin: float
    active_confidence_z: float
    environment_distance: float
    low_z_threshold: float
    switch_z_gap: float | None
    low_streak: int
    candidate_margins: dict[str, float]
    candidate_confidence_z_scores: dict[str, float]
    candidate_validated_ber: dict[str, float]
    candidate_ber_ceiling: float | None
    candidate_gate_results: dict[str, dict[str, bool]]
    recommended_adaptation_model: str | None
    reason: str


class ReliabilityController:
    """Stateful frame sequencer; it never uses labels or triggers retraining."""

    def __init__(
        self,
        references: dict[str, dict[str, object]],
        thresholds: ControllerThresholds | None = None,
        initial_model: str = "oamp_dl",
        validated_ber: dict[str, float] | None = None,
        performance_margin: float = 0.005,
        best_reference_receiver: str = "oamp_dl",
    ) -> None:
        self.references = references
        self.thresholds = thresholds or ControllerThresholds()
        self.validated_ber = {
            name: float(value) for name, value in (validated_ber or {}).items()
        }
        self.performance_margin = float(performance_margin)
        self.best_reference_receiver = best_reference_receiver
        if self.performance_margin < 0.0:
            raise ValueError("performance_margin must be non-negative.")
        if self.validated_ber and (
            self.best_reference_receiver not in self.validated_ber
            or set(references).difference(self.validated_ber)
        ):
            raise ValueError("Every candidate and the best reference need a validated BER value.")
        self.active_model = initial_model
        self.low_streak = 0
        self.frames_since_switch = self.thresholds.minimum_frames_between_switches
        self.frame_index = 0
        if self.active_model not in references:
            raise ValueError(f"Unknown initial model: {self.active_model}")

    def _mean_std(self, receiver: str) -> tuple[float, float]:
        reference = self.references[receiver]
        mean = float(reference["mean"][0])
        standard_deviation = max(float(reference["standard_deviation"][0]), 1e-8)
        return mean, standard_deviation

    def step(
        self,
        active_margin: float,
        environment_distance: float,
        candidate_provider: Callable[[], dict[str, float]] | None = None,
    ) -> ControllerDecision:
        """Process one frame; candidate_provider is called only after 2 low frames."""

        if not np.isfinite(active_margin) or not np.isfinite(environment_distance):
            raise ValueError("Controller inputs must be finite.")
        self.frame_index += 1
        self.frames_since_switch += 1
        active_before = self.active_model
        mean, standard_deviation = self._mean_std(active_before)
        active_confidence_z = (float(active_margin) - mean) / standard_deviation

        if active_confidence_z < self.thresholds.low_z_threshold:
            self.low_streak += 1
        else:
            self.low_streak = 0

        if self.low_streak < self.thresholds.consecutive_low_frames:
            return ControllerDecision(
                frame_index=self.frame_index,
                active_before=active_before,
                active_after=self.active_model,
                decision="KEEP",
                active_margin=float(active_margin),
                active_confidence_z=float(active_confidence_z),
                environment_distance=float(environment_distance),
                low_z_threshold=self.thresholds.low_z_threshold,
                switch_z_gap=None,
                low_streak=self.low_streak,
                candidate_margins={},
                candidate_confidence_z_scores={},
                candidate_validated_ber={},
                candidate_ber_ceiling=None,
                candidate_gate_results={},
                recommended_adaptation_model=None,
                reason="active confidence is not low for two consecutive frames",
            )

        if candidate_provider is None:
            raise ValueError("candidate_provider is required after the low-confidence guard fires.")
        candidate_margins = {
            name: float(value)
            for name, value in candidate_provider().items()
            if name in self.references
        }
        candidate_margins[active_before] = float(active_margin)
        candidate_confidence_z_scores = {}
        for name, margin in candidate_margins.items():
            candidate_mean, candidate_std = self._mean_std(name)
            candidate_confidence_z_scores[name] = (margin - candidate_mean) / candidate_std
        eligible = {
            name: score
            for name, score in candidate_confidence_z_scores.items()
            if name != active_before
            and score >= active_confidence_z + self.thresholds.switch_z_gap
        }
        alternative_margins = {
            name: margin
            for name, margin in candidate_margins.items()
            if name != active_before
        }
        if not alternative_margins:
            raise ValueError("candidate_provider returned no alternative receiver margins.")
        best_candidate = max(alternative_margins, key=alternative_margins.get)

        candidate_gate_results: dict[str, dict[str, bool]] = {}
        candidate_ber_ceiling = None
        if self.validated_ber:
            active_ber = self.validated_ber[active_before]
            best_ber = self.validated_ber[self.best_reference_receiver]
            candidate_ber_ceiling = min(
                active_ber + self.performance_margin,
                best_ber + self.performance_margin,
            )
            candidate_gate_results = {
                name: {
                    "passes_confidence_z_gap": name in eligible,
                    "passes_active_ber_noninferiority": (
                        self.validated_ber[name]
                        <= active_ber + self.performance_margin
                    ),
                    "passes_best_receiver_ber_floor": (
                        self.validated_ber[name]
                        <= best_ber + self.performance_margin
                    ),
                }
                for name in alternative_margins
            }
            eligible = {
                name: score
                for name, score in eligible.items()
                if candidate_gate_results[name]["passes_active_ber_noninferiority"]
                and candidate_gate_results[name]["passes_best_receiver_ber_floor"]
            }
        if eligible and self.frames_since_switch >= self.thresholds.minimum_frames_between_switches:
            new_active = max(eligible, key=eligible.get)
            self.active_model = new_active
            self.frames_since_switch = 0
            self.low_streak = 0
            return ControllerDecision(
                frame_index=self.frame_index,
                active_before=active_before,
                active_after=new_active,
                decision="SWITCH",
                active_margin=float(active_margin),
                active_confidence_z=float(active_confidence_z),
                environment_distance=float(environment_distance),
                low_z_threshold=self.thresholds.low_z_threshold,
                switch_z_gap=self.thresholds.switch_z_gap,
                low_streak=self.thresholds.consecutive_low_frames,
                candidate_margins=candidate_margins,
                candidate_confidence_z_scores=candidate_confidence_z_scores,
                candidate_validated_ber={
                    name: self.validated_ber[name]
                    for name in alternative_margins
                    if name in self.validated_ber
                },
                candidate_ber_ceiling=candidate_ber_ceiling,
                candidate_gate_results=candidate_gate_results,
                recommended_adaptation_model=None,
                reason="candidate exceeds active confidence by the fixed switch margin",
            )

        return ControllerDecision(
            frame_index=self.frame_index,
            active_before=active_before,
            active_after=self.active_model,
            decision="ADAPT",
            active_margin=float(active_margin),
                active_confidence_z=float(active_confidence_z),
            environment_distance=float(environment_distance),
                low_z_threshold=self.thresholds.low_z_threshold,
                switch_z_gap=self.thresholds.switch_z_gap,
            low_streak=self.low_streak,
            candidate_margins=candidate_margins,
                candidate_confidence_z_scores=candidate_confidence_z_scores,
            candidate_validated_ber={
                name: self.validated_ber[name]
                for name in alternative_margins
                if name in self.validated_ber
            },
            candidate_ber_ceiling=candidate_ber_ceiling,
            candidate_gate_results=candidate_gate_results,
            recommended_adaptation_model=best_candidate,
            reason=(
                "no candidate clears the fixed switch margin or the switch cooldown "
                "blocks a change; adaptation is only recommended"
            ),
        )
