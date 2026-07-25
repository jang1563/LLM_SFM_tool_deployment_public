"""Deterministic finite-grid risk certificates for C5 trust routing."""

from __future__ import annotations

import math
from typing import Any, Sequence


def select_hoeffding_certificate(
    observations: Sequence[tuple[float, bool]],
    *,
    alpha: float,
    delta: float,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Select the highest-coverage threshold with a union-bound certificate.

    Each observation is ``(confidence, success)``. The finite threshold family
    is treated as fixed before labels are scored, and the Hoeffding radius is
    corrected uniformly by the number of candidate thresholds.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    candidates = tuple(thresholds)
    if not candidates:
        raise ValueError("thresholds must be non-empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("thresholds must be unique")
    for index, threshold in enumerate(candidates):
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold[{index}] outside [0, 1]")
    if not observations:
        raise ValueError("observations must be non-empty")

    normalized: list[tuple[float, bool]] = []
    for index, (confidence, success) in enumerate(observations):
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"observation[{index}] confidence outside [0, 1]")
        if not isinstance(success, bool):
            raise ValueError(f"observation[{index}] success must be bool")
        normalized.append((confidence, success))

    evaluated: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in candidates:
        trusted = [
            success
            for confidence, success in normalized
            if confidence >= threshold
        ]
        if not trusted:
            continue
        failures = sum(not success for success in trusted)
        empirical_risk = failures / len(trusted)
        radius = math.sqrt(
            math.log(len(candidates) / delta) / (2 * len(trusted))
        )
        upper_bound = min(1.0, empirical_risk + radius)
        candidate = {
            "threshold": threshold,
            "trusted": len(trusted),
            "failures": failures,
            "empirical_risk": empirical_risk,
            "risk_upper_bound": upper_bound,
        }
        evaluated.append(candidate)
        if upper_bound <= alpha:
            eligible.append(candidate)

    closest = (
        min(
            evaluated,
            key=lambda candidate: (
                candidate["risk_upper_bound"],
                -candidate["trusted"],
                candidate["threshold"],
            ),
        )
        if evaluated
        else None
    )
    if not eligible:
        return {
            "alpha": alpha,
            "delta": delta,
            "candidate_count": len(candidates),
            "certified": False,
            "threshold": None,
            "calibration_trusted": 0,
            "calibration_failures": 0,
            "calibration_empirical_risk": None,
            "calibration_risk_upper_bound": None,
            "closest_candidate": _compact_candidate(closest),
        }

    best = min(
        eligible,
        key=lambda candidate: (
            -candidate["trusted"],
            candidate["threshold"],
        ),
    )
    return {
        "alpha": alpha,
        "delta": delta,
        "candidate_count": len(candidates),
        "certified": True,
        "threshold": best["threshold"],
        "calibration_trusted": best["trusted"],
        "calibration_failures": best["failures"],
        "calibration_empirical_risk": round(best["empirical_risk"], 6),
        "calibration_risk_upper_bound": round(best["risk_upper_bound"], 6),
        "closest_candidate": _compact_candidate(closest),
    }


def threshold_policy_metrics(
    observations: Sequence[tuple[float, bool]],
    *,
    threshold: float | None,
) -> dict[str, Any]:
    """Return target-level risk and coverage for a confidence threshold."""

    trusted = (
        list(observations)
        if threshold is None
        else [
            observation
            for observation in observations
            if observation[0] >= threshold
        ]
    )
    failures = sum(not success for _, success in trusted)
    count = len(observations)
    return {
        "targets": count,
        "trusted": len(trusted),
        "verify_or_defer": count - len(trusted),
        "failures_among_trusted": failures,
        "failure_rate_among_trusted": (
            round(failures / len(trusted), 6) if trusted else 0.0
        ),
        "coverage": round(len(trusted) / count, 6) if count else 0.0,
    }


def _compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "threshold": candidate["threshold"],
        "trusted": candidate["trusted"],
        "failures": candidate["failures"],
        "empirical_risk": round(candidate["empirical_risk"], 6),
        "risk_upper_bound": round(candidate["risk_upper_bound"], 6),
    }
