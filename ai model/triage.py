"""Confidence-aware triage logic from the project research paper.

This module is model-agnostic: a trained CheXpert classifier supplies one
probability per observation, and this module produces an auditable triage
record for the clinical UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TriageResult:
    finding: str
    confidence: float
    severity: float
    risk: float
    priority: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_risk(severity: float, confidence: float, alpha: float = 0.7, beta: float = 0.3) -> float:
    """Calculate R = alpha*S + beta*(1-C), as defined in the paper."""
    if not 0 <= severity <= 1:
        raise ValueError("severity must be between 0 and 1")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if alpha < 0 or beta < 0 or abs(alpha + beta - 1) > 1e-9:
        raise ValueError("alpha and beta must be non-negative and sum to 1")
    return alpha * severity + beta * (1 - confidence)


def priority_for_risk(risk: float, high_threshold: float = 0.70, low_threshold: float = 0.35) -> str:
    """Map the risk score to the paper's three triage priorities."""
    if not 0 <= risk <= 1:
        raise ValueError("risk must be between 0 and 1")
    if not 0 <= low_threshold < high_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 1")
    if risk >= high_threshold:
        return "High"
    if risk >= low_threshold:
        return "Medium"
    return "Low"


def triage_prediction(
    probabilities: Mapping[str, float] | Sequence[float],
    severity_by_finding: Mapping[str, float],
    labels: Sequence[str] | None = None,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    high_threshold: float = 0.70,
    low_threshold: float = 0.35,
) -> TriageResult:
    """Convert CheXpert probabilities into one explainable triage result."""
    if isinstance(probabilities, Mapping):
        named_probabilities = dict(probabilities)
    else:
        if labels is None or len(labels) != len(probabilities):
            raise ValueError("labels must match a sequence of probabilities")
        named_probabilities = dict(zip(labels, probabilities))

    if not named_probabilities:
        raise ValueError("at least one prediction probability is required")
    if any(not 0 <= value <= 1 for value in named_probabilities.values()):
        raise ValueError("all prediction probabilities must be between 0 and 1")

    finding, confidence = max(named_probabilities.items(), key=lambda item: item[1])
    if finding not in severity_by_finding:
        raise KeyError(f"missing severity for predicted finding: {finding}")
    severity = float(severity_by_finding[finding])
    risk = calculate_risk(severity, confidence, alpha, beta)
    priority = priority_for_risk(risk, high_threshold, low_threshold)
    return TriageResult(finding, float(confidence), severity, risk, priority)
