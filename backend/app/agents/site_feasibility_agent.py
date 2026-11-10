
"""
Site Feasibility Agent
Multi-Agent Clinical Trial Feasibility Assessment

Responsibilities:
- Evaluate supplied site readiness and suitability indicators.
- Calculate a transparent, weighted site feasibility score.
- Capture monthly recruitment capacity.
- Identify operational risks and missing information.
- Return site-level results for downstream recruitment analysis.

Scoring:
- Each supplied criterion uses a 0-5 scale.
- Weights are normalized across valid criteria.
- Missing criteria are excluded from scoring, not treated as zero.
- Scores are illustrative and require domain validation.
"""

import logging
import math
from numbers import Real
from typing import Any

from app.config.settings import settings
logger = settings.get_logger()

# Illustrative weights. Validate these with clinical operations SMEs.
SITE_SCORE_WEIGHTS = {
    "patient_access_score": 0.30,
    "trial_experience_score": 0.20,
    "staffing_score": 0.15,
    "facility_score": 0.15,
    "regulatory_readiness_score": 0.20,
}

MINIMUM_SCORE_COVERAGE = 0.80


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a value from a dictionary or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _is_valid_number(value: Any) -> bool:
    """Check whether a value is a finite numeric value, excluding booleans."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_score(value: Any) -> bool:
    """Site criteria must use a numeric score between 0 and 5."""
    return _is_valid_number(value) and 0 <= value <= 5


def _validate_capacity(value: Any) -> bool:
    """Monthly enrollment capacity must be nonnegative."""
    return _is_valid_number(value) and value >= 0


def _extract_sites(request: Any) -> list:
    """Extract proposed sites from the request."""
    sites = _get_value(request, "sites", None)

    if sites is None:
        sites = _get_value(request, "site_data", [])

    if sites is None:
        return []

    if not isinstance(sites, list):
        raise ValueError("The sites field must be a list.")

    return sites


def _score_site(site: dict) -> dict:
    """
    Calculate a weighted score from the available valid criteria.

    The score is normalized to 0-100. Coverage indicates how much of
    the configured scoring weight is supported by valid input data.
    """
    weighted_total = 0.0
    available_weight = 0.0
    invalid_criteria = []
    missing_criteria = []

    for criterion, weight in SITE_SCORE_WEIGHTS.items():
        value = site.get(criterion)

        if value is None:
            missing_criteria.append(criterion)
            continue

        if not _validate_score(value):
            invalid_criteria.append(criterion)
            continue

        weighted_total += (float(value) / 5.0) * weight
        available_weight += weight

    if available_weight == 0:
        score = None
        coverage = 0.0
    else:
        score = round(
            (weighted_total / available_weight) * 100.0,
            2,
        )
        coverage = round(available_weight * 100.0, 2)

    sufficiently_supported = (
        available_weight >= MINIMUM_SCORE_COVERAGE
    )

    if not sufficiently_supported:
        suitability = "insufficient_data"
    elif score >= 80:
        suitability = "higher_relative_suitability"
    elif score >= 60:
        suitability = "moderate_relative_suitability"
    else:
        suitability = "lower_relative_suitability"

    return {
        "score": score,
        "score_coverage_percent": coverage,
        "suitability": suitability,
        "missing_criteria": missing_criteria,
        "invalid_criteria": invalid_criteria,
        "score_is_provisional": not sufficiently_supported,
    }


def _deterministic_site_feasibility_agent(state: dict) -> dict:
    """
    Evaluate the sites supplied in the feasibility request.

    Expected request fields:
        study_id
        target_enrollment
        planned_months
        sites: list[dict]

    Each site may contain:
        site_id
        site_name
        capacity_per_month
        patient_access_score
        trial_experience_score
        staffing_score
        facility_score
        regulatory_readiness_score
        evidence_refs

    Returns:
        Partial LangGraph state containing site_results.
    """
    request = state.get("request")

    if request is None:
        raise ValueError(
            "Site feasibility analysis requires a feasibility request."
        )

    study_id = _get_value(request, "study_id", "unknown")
    sites = _extract_sites(request)

    if not sites:
        logger.warning(
            "No proposed sites supplied | study_id=%s",
            study_id,
        )

        return {
            "site_results": [],
            "site_assessment_summary": {
                "study_id": study_id,
                "sites_evaluated": 0,
                "sites_with_supported_scores": 0,
                "missing_information": [
                    "No proposed sites were supplied."
                ],
                "requires_human_review": True,
            },
        }

    site_results = []
    site_ids = set()

    for index, raw_site in enumerate(sites, start=1):
        if not isinstance(raw_site, dict):
            site_results.append({
                "site_id": f"unknown-{index}",
                "score": None,
                "score_coverage_percent": 0.0,
                "suitability": "insufficient_data",
                "capacity_per_month": None,
                "findings": [{
                    "category": "site_data_quality",
                    "severity": "high",
                    "finding": "Site input is not a dictionary.",
                    "requires_human_review": True,
                }],
                "evidence_refs": [],
                "requires_human_review": True,
            })
            continue

        site_id = str(raw_site.get("site_id", "")).strip()

        if not site_id:
            site_id = f"site-{index}"

        findings = []
        assumptions = []
        missing_information = []

        if site_id in site_ids:
            findings.append({
                "category": "site_data_quality",
                "severity": "high",
                "finding": "Duplicate site identifier detected.",
                "requires_human_review": True,
            })
        site_ids.add(site_id)

        score_result = _score_site(raw_site)

        if score_result["score_is_provisional"]:
            findings.append({
                "category": "site_data_quality",
                "severity": "medium",
                "finding": (
                    "The site score is provisional because insufficient "
                    "scoring criteria have valid values."
                ),
                "rationale": (
                    f"Valid score weight coverage: "
                    f"{score_result['score_coverage_percent']}%."
                ),
                "recommended_action": (
                    "Collect and validate missing site readiness indicators."
                ),
                "requires_human_review": True,
            })

        if score_result["invalid_criteria"]:
            findings.append({
                "category": "site_data_quality",
                "severity": "high",
                "finding": "One or more site scores are invalid.",
                "rationale": (
                    "Criteria must use the defined 0-5 numeric scale."
                ),
                "invalid_criteria": score_result["invalid_criteria"],
                "requires_human_review": True,
            })

        if score_result["missing_criteria"]:
            missing_information.extend(
                score_result["missing_criteria"]
            )

        capacity = raw_site.get("capacity_per_month")

        if capacity is None:
            missing_information.append("capacity_per_month")
            validated_capacity = None
        elif not _validate_capacity(capacity):
            validated_capacity = None
            findings.append({
                "category": "site_capacity",
                "severity": "high",
                "finding": (
                    "Monthly site capacity is missing or invalid."
                ),
                "rationale": (
                    "Capacity must be a finite, nonnegative number."
                ),
                "requires_human_review": True,
            })
        else:
            validated_capacity = float(capacity)
            assumptions.append(
                "Reported monthly capacity is assumed to be valid "
                "for the planned recruitment period."
            )

        evidence_refs = raw_site.get("evidence_refs", [])

        if not isinstance(evidence_refs, list):
            evidence_refs = []
            findings.append({
                "category": "evidence_quality",
                "severity": "medium",
                "finding": "Site evidence references must be a list.",
                "requires_human_review": True,
            })

        if not evidence_refs:
            findings.append({
                "category": "evidence_quality",
                "severity": "medium",
                "finding": (
                    "No supporting evidence references were supplied "
                    "for this site's assessment."
                ),
                "requires_human_review": True,
            })

        result = {
            "site_id": site_id,
            "site_name": raw_site.get("site_name"),
            "score": score_result["score"],
            "score_coverage_percent": (
                score_result["score_coverage_percent"]
            ),
            "suitability": score_result["suitability"],
            "capacity_per_month": validated_capacity,
            "missing_criteria": score_result["missing_criteria"],
            "invalid_criteria": score_result["invalid_criteria"],
            "missing_information": list(
                dict.fromkeys(missing_information)
            ),
            "findings": findings,
            "evidence_refs": evidence_refs,
            "assumptions": assumptions,
            "score_weights": SITE_SCORE_WEIGHTS.copy(),
            "requires_human_review": True,
        }

        site_results.append(result)

    supported_count = sum(
        1
        for result in site_results
        if (
            result.get("score") is not None
            and not result.get("requires_human_review", False)
        )
    )

    summary = {
        "study_id": study_id,
        "sites_evaluated": len(site_results),
        "sites_with_supported_scores": supported_count,
        "sites_requiring_review": len(site_results) - supported_count,
        "requires_human_review": True,
        "limitations": [
            "Weights and score bands are illustrative and require SME validation.",
            "A high score does not establish regulatory approval or site activation.",
            "Scores depend on the accuracy and recency of supplied data.",
            "Missing criteria are excluded and the remaining weights normalized.",
        ],
    }

    logger.info(
        "Site feasibility analysis completed | study_id=%s | sites=%d",
        study_id,
        len(site_results),
    )

    return {
        "site_results": site_results,
        "site_assessment_summary": summary,
    }


def site_feasibility_agent(state: dict) -> dict:
    """Run deterministic checks, then attach advisory structured LLM insights.

    Deterministic calculations and safety/review flags remain authoritative.
    """
    from app.agents.llm_enrichment import enrich_result

    output = _deterministic_site_feasibility_agent(state)
    return enrich_result(
        state=state,
        agent_name="site_feasibility_agent",
        output=output,
        output_key="site_assessment_summary",
    )
