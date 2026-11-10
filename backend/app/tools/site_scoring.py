
"""
Deterministic site scoring utilities for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Scoring criteria use a 0-5 scale:
    0 = unacceptable / absent
    1 = very weak
    2 = below expectations
    3 = acceptable
    4 = strong
    5 = excellent

The scale and weights are illustrative POC defaults. Clinical operations
SMEs should validate them before operational or sponsor-facing use.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


CRITERION_WEIGHTS: dict[str, float] = {
    "patient_access_score": 0.30,
    "trial_experience_score": 0.20,
    "staffing_score": 0.15,
    "facility_score": 0.15,
    "regulatory_readiness_score": 0.20,
}

MIN_SCORE = 0.0
MAX_SCORE = 5.0
MIN_DATA_COVERAGE_PERCENT = 80.0

# Illustrative bands; validate with clinical operations SMEs.
SCORE_BANDS = (
    (80.0, "high"),
    (60.0, "moderate"),
    (0.0, "low"),
)


class SiteScoringError(ValueError):
    """Raised when site scoring inputs are structurally invalid."""


def _is_finite_number(value: Any) -> bool:
    """True only for finite numeric values; booleans are not accepted."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Validate that criterion weights are nonnegative and sum to 1."""
    if not weights:
        raise SiteScoringError("At least one scoring criterion is required.")

    normalized: dict[str, float] = {}

    for criterion, weight in weights.items():
        if not isinstance(criterion, str) or not criterion.strip():
            raise SiteScoringError("Criterion names must be non-empty strings.")

        if not _is_finite_number(weight) or float(weight) < 0:
            raise SiteScoringError(
                f"Weight for '{criterion}' must be finite and nonnegative."
            )

        normalized[criterion] = float(weight)

    total_weight = sum(normalized.values())

    if total_weight <= 0 or not math.isclose(
        total_weight, 1.0, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise SiteScoringError(
            f"Criterion weights must sum to 1.0; got {total_weight:.6f}."
        )

    return normalized


def _score_band(score: Optional[float]) -> str:
    """Return an illustrative feasibility band."""
    if score is None:
        return "insufficient_data"

    for minimum, band in SCORE_BANDS:
        if score >= minimum:
            return band

    return "low"


def score_site(
    site: Mapping[str, Any],
    weights: Optional[Mapping[str, float]] = None,
    minimum_coverage_percent: float = MIN_DATA_COVERAGE_PERCENT,
) -> dict[str, Any]:
    """
    Calculate an explainable site feasibility score.

    Expected site fields:
        site_id
        site_name (optional)
        patient_access_score (0-5)
        trial_experience_score (0-5)
        staffing_score (0-5)
        facility_score (0-5)
        regulatory_readiness_score (0-5)

    Returns:
        A normalized score, data coverage, criterion-level details,
        findings, and a human-review flag.

    Missing criteria are excluded from the score and coverage denominator
    remains the sum of all configured weights. Invalid supplied values
    are reported and excluded.
    """
    if not isinstance(site, Mapping):
        raise SiteScoringError("site must be a mapping.")

    site_id = site.get("site_id")
    if not isinstance(site_id, str) or not site_id.strip():
        raise SiteScoringError("site_id must be a non-empty string.")

    if (
        isinstance(minimum_coverage_percent, bool)
        or not _is_finite_number(minimum_coverage_percent)
        or not 0.0 <= float(minimum_coverage_percent) <= 100.0
    ):
        raise SiteScoringError(
            "minimum_coverage_percent must be between 0 and 100."
        )

    active_weights = _validate_weights(
        weights if weights is not None else CRITERION_WEIGHTS
    )

    weighted_score = 0.0
    available_weight = 0.0
    criterion_details: dict[str, dict[str, Any]] = {}
    missing_criteria: list[str] = []
    invalid_criteria: list[str] = []
    findings: list[dict[str, str]] = []

    for criterion, weight in active_weights.items():
        raw_value = site.get(criterion)

        if raw_value is None:
            missing_criteria.append(criterion)
            criterion_details[criterion] = {
                "value": None,
                "weight": weight,
                "status": "missing",
                "weighted_contribution": None,
            }
            continue

        if (
            not _is_finite_number(raw_value)
            or not MIN_SCORE <= float(raw_value) <= MAX_SCORE
        ):
            invalid_criteria.append(criterion)
            criterion_details[criterion] = {
                "value": raw_value,
                "weight": weight,
                "status": "invalid",
                "weighted_contribution": None,
            }
            findings.append(
                {
                    "code": "INVALID_CRITERION_SCORE",
                    "severity": "high",
                    "message": (
                        f"{criterion} must be a finite number from "
                        f"{MIN_SCORE:g} to {MAX_SCORE:g}."
                    ),
                }
            )
            continue

        value = float(raw_value)
        contribution = (value / MAX_SCORE) * weight * 100.0

        weighted_score += contribution
        available_weight += weight

        criterion_details[criterion] = {
            "value": value,
            "weight": weight,
            "status": "valid",
            "weighted_contribution": round(contribution, 2),
        }

    total_weight = sum(active_weights.values())
    coverage_percent = (
        (available_weight / total_weight) * 100.0
        if total_weight > 0
        else 0.0
    )

    # Normalize against available weight so missing criteria are not
    # silently treated as zero. Coverage is returned separately.
    score = (
        (weighted_score / available_weight) * 100.0
        if available_weight > 0
        else None
    )

    score = round(score, 2) if score is not None else None
    coverage_percent = round(coverage_percent, 2)

    if missing_criteria:
        findings.append(
            {
                "code": "MISSING_CRITERIA",
                "severity": "medium",
                "message": (
                    "Missing scoring criteria: "
                    + ", ".join(missing_criteria)
                ),
            }
        )

    if coverage_percent < float(minimum_coverage_percent):
        findings.append(
            {
                "code": "LOW_DATA_COVERAGE",
                "severity": "high",
                "message": (
                    f"Scoring data coverage is {coverage_percent:.2f}%; "
                    f"minimum required is "
                    f"{float(minimum_coverage_percent):.2f}%."
                ),
            }
        )

    if score is None:
        findings.append(
            {
                "code": "NO_VALID_CRITERIA",
                "severity": "high",
                "message": "No valid scoring criteria are available.",
            }
        )

    # A numeric score can still be provisional or unsuitable for ranking.
    requires_human_review = (
        score is None
        or coverage_percent < float(minimum_coverage_percent)
        or bool(invalid_criteria)
        or bool(missing_criteria)
    )

    return {
        "site_id": site_id,
        "site_name": site.get("site_name"),
        "score": score,
        "score_band": _score_band(score),
        "data_coverage_percent": coverage_percent,
        "available_weight": round(available_weight, 4),
        "total_weight": round(total_weight, 4),
        "criterion_details": criterion_details,
        "missing_criteria": missing_criteria,
        "invalid_criteria": invalid_criteria,
        "findings": findings,
        "requires_human_review": requires_human_review,
        "scoring_version": "site-score-v1",
    }


def score_sites(
    sites: Sequence[Mapping[str, Any]],
    weights: Optional[Mapping[str, float]] = None,
    minimum_coverage_percent: float = MIN_DATA_COVERAGE_PERCENT,
) -> dict[str, Any]:
    """
    Score multiple sites and produce a transparent ranking.

    Sites with inadequate coverage or invalid criteria are retained in
    the output but are not eligible for the supported ranking.
    """
    if isinstance(sites, (str, bytes)) or not isinstance(sites, Sequence):
        raise SiteScoringError("sites must be a sequence of site records.")

    results = [
        score_site(
            site=site,
            weights=weights,
            minimum_coverage_percent=minimum_coverage_percent,
        )
        for site in sites
    ]

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for result in results:
        site_id = result["site_id"]
        if site_id in seen_ids:
            duplicate_ids.add(site_id)
        seen_ids.add(site_id)

    for result in results:
        if result["site_id"] in duplicate_ids:
            result["requires_human_review"] = True
            result["findings"].append(
                {
                    "code": "DUPLICATE_SITE_ID",
                    "severity": "high",
                    "message": (
                        f"Duplicate site_id '{result['site_id']}' "
                        "prevents unambiguous site identification."
                    ),
                }
            )

    eligible = [
        result
        for result in results
        if (
            result["score"] is not None
            and result["data_coverage_percent"] >= minimum_coverage_percent
            and not result["invalid_criteria"]
            and not result["missing_criteria"]
            and result["site_id"] not in duplicate_ids
        )
    ]

    # Stable sorting: score descending, then site_id ascending.
    eligible.sort(
        key=lambda result: (
            -result["score"],
            result["site_id"],
        )
    )

    for rank, result in enumerate(eligible, start=1):
        result["rank"] = rank

    eligible_ids = {result["site_id"] for result in eligible}
    for result in results:
        if result["site_id"] not in eligible_ids:
            result["rank"] = None

    return {
        "scoring_version": "site-score-v1",
        "site_count": len(results),
        "ranked_site_count": len(eligible),
        "sites": results,
        "ranking": [
            {
                "rank": result["rank"],
                "site_id": result["site_id"],
                "site_name": result["site_name"],
                "score": result["score"],
                "score_band": result["score_band"],
                "data_coverage_percent": result["data_coverage_percent"],
            }
            for result in eligible
        ],
        "limitations": [
            "Weights and score bands are illustrative POC defaults.",
            "Scores depend on the accuracy and comparability of supplied inputs.",
            "A higher score does not establish regulatory or operational approval.",
            "Final site selection requires qualified clinical operations review.",
        ],
    }
