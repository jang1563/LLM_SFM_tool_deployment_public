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

    Each observation is ``(score, success)``. The finite threshold family
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
        if not math.isfinite(confidence):
            raise ValueError(f"observation[{index}] score must be finite")
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


def select_exact_binomial_certificate(
    observations: Sequence[tuple[float, bool]],
    *,
    alpha: float,
    delta: float,
    thresholds: Sequence[float],
    multiplicity: int | None = None,
) -> dict[str, Any]:
    """Select a threshold using exact Bernoulli tests with Bonferroni control.

    For each fixed threshold, the null hypothesis is that conditional failure
    risk exceeds ``alpha``. The lower-tail binomial p-value is compared with
    ``delta / multiplicity``. Threshold tests may be dependent; Bonferroni
    therefore remains valid for the finite, preregistered family.
    """

    normalized, candidates = _validated_inputs(
        observations,
        alpha=alpha,
        delta=delta,
        thresholds=thresholds,
    )
    family_size = len(candidates) if multiplicity is None else multiplicity
    if (
        isinstance(family_size, bool)
        or not isinstance(family_size, int)
        or family_size < len(candidates)
        or family_size < 1
    ):
        raise ValueError("multiplicity must cover every candidate threshold")
    per_candidate_delta = delta / family_size

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
        p_value = _binomial_lower_tail(
            failures,
            len(trusted),
            alpha,
        )
        upper_bound = exact_binomial_upper_bound(
            failures,
            len(trusted),
            delta=per_candidate_delta,
        )
        candidate = {
            "threshold": threshold,
            "trusted": len(trusted),
            "failures": failures,
            "empirical_risk": empirical_risk,
            "risk_upper_bound": upper_bound,
            "risk_test_p_value": p_value,
        }
        evaluated.append(candidate)
        if p_value <= per_candidate_delta:
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
    common = {
        "alpha": alpha,
        "delta": delta,
        "candidate_count": family_size,
        "evaluated_threshold_count": len(candidates),
        "per_candidate_delta": per_candidate_delta,
        "bound": "exact one-sided binomial with Bonferroni correction",
    }
    if not eligible:
        return {
            **common,
            "certified": False,
            "threshold": None,
            "calibration_trusted": 0,
            "calibration_failures": 0,
            "calibration_empirical_risk": None,
            "calibration_risk_upper_bound": None,
            "calibration_risk_test_p_value": None,
            "closest_candidate": _compact_exact_candidate(closest),
        }

    best = min(
        eligible,
        key=lambda candidate: (
            -candidate["trusted"],
            candidate["threshold"],
        ),
    )
    return {
        **common,
        "certified": True,
        "threshold": best["threshold"],
        "calibration_trusted": best["trusted"],
        "calibration_failures": best["failures"],
        "calibration_empirical_risk": round(best["empirical_risk"], 6),
        "calibration_risk_upper_bound": round(best["risk_upper_bound"], 6),
        "calibration_risk_test_p_value": round(
            best["risk_test_p_value"],
            12,
        ),
        "closest_candidate": _compact_exact_candidate(closest),
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


def fixed_threshold_hoeffding_metrics(
    observations: Sequence[tuple[float, bool]],
    *,
    threshold: float | None,
    delta: float,
) -> dict[str, Any]:
    """Return fixed-policy metrics with a non-search Hoeffding upper bound."""

    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or math.isnan(threshold)
    ):
        raise ValueError("threshold must be numeric and not NaN")
    if not observations:
        raise ValueError("observations must be non-empty")
    for index, (score, success) in enumerate(observations):
        if not math.isfinite(score):
            raise ValueError(f"observation[{index}] score must be finite")
        if not isinstance(success, bool):
            raise ValueError(f"observation[{index}] success must be bool")
    metrics = threshold_policy_metrics(observations, threshold=threshold)
    trusted = metrics["trusted"]
    if trusted == 0:
        risk_upper_bound = None
    else:
        risk_upper_bound = min(
            1.0,
            metrics["failures_among_trusted"] / trusted
            + math.sqrt(math.log(1 / delta) / (2 * trusted)),
        )
        risk_upper_bound = round(risk_upper_bound, 6)
    return {
        **metrics,
        "delta": delta,
        "risk_upper_bound": risk_upper_bound,
        "bound": "fixed-threshold Hoeffding",
    }


def fixed_threshold_exact_binomial_metrics(
    observations: Sequence[tuple[float, bool]],
    *,
    threshold: float | None,
    alpha: float,
    delta: float,
) -> dict[str, Any]:
    """Return fixed-policy metrics with an exact one-sided binomial test."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or math.isnan(threshold)
    ):
        raise ValueError("threshold must be numeric and not NaN")
    _validate_observations(observations)
    metrics = threshold_policy_metrics(observations, threshold=threshold)
    trusted = metrics["trusted"]
    if trusted == 0:
        upper_bound = None
        p_value = None
        passed = False
    else:
        failures = metrics["failures_among_trusted"]
        upper_bound = round(
            exact_binomial_upper_bound(failures, trusted, delta=delta),
            6,
        )
        p_value = round(
            _binomial_lower_tail(failures, trusted, alpha),
            12,
        )
        passed = p_value <= delta
    return {
        **metrics,
        "alpha": alpha,
        "delta": delta,
        "risk_upper_bound": upper_bound,
        "risk_test_p_value": p_value,
        "risk_test_passed": passed,
        "bound": "fixed-threshold exact one-sided binomial",
    }


def exact_binomial_upper_bound(
    failures: int,
    trials: int,
    *,
    delta: float,
) -> float:
    """Invert the exact binomial lower tail for a one-sided risk bound."""

    if (
        isinstance(trials, bool)
        or not isinstance(trials, int)
        or trials < 1
    ):
        raise ValueError("trials must be a positive integer")
    if (
        isinstance(failures, bool)
        or not isinstance(failures, int)
        or failures < 0
        or failures > trials
    ):
        raise ValueError("failures must be an integer in [0, trials]")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if failures == trials:
        return 1.0

    lower = failures / trials
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        if _binomial_lower_tail(failures, trials, midpoint) > delta:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _binomial_lower_tail(successes: int, trials: int, probability: float) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    return sum(
        math.comb(trials, value)
        * probability**value
        * (1 - probability) ** (trials - value)
        for value in range(successes + 1)
    )


def _validated_inputs(
    observations: Sequence[tuple[float, bool]],
    *,
    alpha: float,
    delta: float,
    thresholds: Sequence[float],
) -> tuple[list[tuple[float, bool]], tuple[float, ...]]:
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
    return _validate_observations(observations), candidates


def _validate_observations(
    observations: Sequence[tuple[float, bool]],
) -> list[tuple[float, bool]]:
    if not observations:
        raise ValueError("observations must be non-empty")
    normalized: list[tuple[float, bool]] = []
    for index, (confidence, success) in enumerate(observations):
        if not math.isfinite(confidence):
            raise ValueError(f"observation[{index}] score must be finite")
        if not isinstance(success, bool):
            raise ValueError(f"observation[{index}] success must be bool")
        normalized.append((confidence, success))
    return normalized


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


def _compact_exact_candidate(
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    compact = _compact_candidate(candidate)
    if compact is None or candidate is None:
        return None
    return {
        **compact,
        "risk_test_p_value": round(candidate["risk_test_p_value"], 12),
    }
