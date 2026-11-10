
"""
Recruitment Risk Agent
Multi-Agent Clinical Trial Feasibility Assessment

Responsibilities:
- Calculate required monthly enrollment.
- Aggregate supplied site recruitment capacity.
- Compare capacity with the study enrollment target and timeline.
- Incorporate patient population findings when available.
- Identify recruitment risks, missing information and assumptions.

This is a deterministic screening agent, not a validated enrollment
forecasting model.
"""

import logging
import math
from numbers import Real
from typing import Any

from app.config.settings import settings
logger = settings.get_logger()


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dictionary or Pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _valid_number(value: Any) -> bool:
    """Return True for finite, nonnegative numeric values."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _severity_for_gap(
    required_monthly: float,
    available_monthly: float,
) -> str:
    """Classify a capacity gap using deterministic rules."""
    if available_monthly <= 0:
        return "critical"

    if available_monthly < required_monthly:
        return "high"

    return "low"


def _deterministic_recruitment_risk_agent(state: dict) -> dict:
    """
    Assess recruitment risk using current workflow state.

    Expected state inputs:
        request:
            study_id
            target_enrollment
            planned_months

        site_results:
            list of site dictionaries, each optionally containing
            site_id and capacity_per_month

        population_result:
            optional output from patient_population_agent

    Returns:
        Partial LangGraph state containing recruitment_result.
    """
    request = state.get("request")

    if request is None:
        raise ValueError(
            "Recruitment risk analysis requires a feasibility request."
        )

    study_id = _get_value(request, "study_id", "unknown")
    target = _get_value(request, "target_enrollment")
    planned_months = _get_value(request, "planned_months")

    findings = []
    assumptions = []
    missing_information = []

    # Validate enrollment target.
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target <= 0
    ):
        raise ValueError(
            "target_enrollment must be a positive integer."
        )

    # Validate the recruitment period.
    if (
        isinstance(planned_months, bool)
        or not isinstance(planned_months, int)
        or planned_months <= 0
    ):
        raise ValueError(
            "planned_months must be a positive integer."
        )

    required_monthly = target / planned_months

    # Aggregate capacities from site feasibility findings.
    site_results = state.get("site_results", [])

    if not isinstance(site_results, list):
        site_results = []
        missing_information.append(
            "Site results were not provided in the expected list format."
        )

    valid_capacities = []
    invalid_sites = []

    for site in site_results:
        if not isinstance(site, dict):
            invalid_sites.append("unknown")
            continue

        site_id = site.get("site_id", "unknown")
        capacity = site.get("capacity_per_month")

        if capacity is None:
            invalid_sites.append(str(site_id))
            continue

        if not _valid_number(capacity):
            invalid_sites.append(str(site_id))
            continue

        valid_capacities.append(float(capacity))

    if invalid_sites:
        missing_information.append(
            "Monthly recruitment capacity is missing or invalid for "
            f"{len(invalid_sites)} site result(s)."
        )

    available_monthly = (
        sum(valid_capacities)
        if valid_capacities
        else None
    )

    expected_capacity = None
    capacity_gap = None
    capacity_status = "insufficient_data"

    if available_monthly is not None:
        expected_capacity = available_monthly * planned_months
        capacity_gap = required_monthly - available_monthly

        if available_monthly == 0:
            capacity_status = "no_reported_capacity"
        elif available_monthly < required_monthly:
            capacity_status = "capacity_below_required_rate"
        else:
            capacity_status = "capacity_meets_required_rate"

        assumptions.append(
            "Site capacities are treated as monthly enrollment rates "
            "and are assumed to be additive."
        )

        assumptions.append(
            "Site capacity is assumed to remain constant throughout "
            "the planned recruitment period."
        )

        if capacity_status == "capacity_below_required_rate":
            findings.append({
                "category": "recruitment_capacity",
                "severity": _severity_for_gap(
                    required_monthly,
                    available_monthly,
                ),
                "finding": (
                    "Reported site capacity is below the required "
                    "monthly enrollment rate."
                ),
                "rationale": (
                    f"Required: {required_monthly:.2f} participants/month; "
                    f"reported capacity: {available_monthly:.2f} "
                    "participants/month."
                ),
                "evidence_refs": [],
                "assumptions": list(assumptions),
                "recommended_action": (
                    "Validate site enrollment rates and evaluate additional "
                    "sites, recruitment strategies, or a revised timeline."
                ),
                "requires_human_review": True,
            })

        elif capacity_status == "no_reported_capacity":
            findings.append({
                "category": "recruitment_capacity",
                "severity": "critical",
                "finding": "No positive monthly recruitment capacity was reported.",
                "rationale": (
                    "The supplied site capacity is zero; recruitment "
                    "feasibility cannot be established."
                ),
                "evidence_refs": [],
                "assumptions": list(assumptions),
                "recommended_action": (
                    "Confirm site activation status and provide "
                    "evidence-supported recruitment capacity."
                ),
                "requires_human_review": True,
            })

        else:
            findings.append({
                "category": "recruitment_capacity",
                "severity": "low",
                "finding": (
                    "Reported monthly site capacity meets or exceeds "
                    "the arithmetic enrollment requirement."
                ),
                "rationale": (
                    "This comparison does not account for activation "
                    "delays, competing trials, screening failures, or "
                    "monthly variation in enrollment."
                ),
                "evidence_refs": [],
                "assumptions": list(assumptions),
                "recommended_action": (
                    "Validate historical site performance and "
                    "recruitment assumptions."
                ),
                "requires_human_review": True,
            })

    else:
        missing_information.append(
            "No valid site-level monthly recruitment capacity is available."
        )

    # Incorporate the population agent's findings, if present.
    population_result = state.get("population_result", {})

    if not isinstance(population_result, dict):
        population_result = {}
        missing_information.append(
            "Population result was not provided in the expected format."
        )

    recruitable_population = population_result.get(
        "recruitable_population_estimate"
    )

    population_status = population_result.get(
        "population_status",
        "insufficient_data",
    )

    if recruitable_population is not None:
        if not _valid_number(recruitable_population):
            missing_information.append(
                "Recruitable population estimate is invalid."
            )
        elif recruitable_population < target:
            findings.append({
                "category": "patient_availability",
                "severity": "high",
                "finding": (
                    "The supplied recruitable population estimate "
                    "is below target enrollment."
                ),
                "rationale": (
                    f"Estimated recruitable population: "
                    f"{recruitable_population:g}; target: {target}."
                ),
                "evidence_refs": population_result.get(
                    "evidence_refs", []
                ),
                "assumptions": population_result.get(
                    "assumptions", []
                ),
                "recommended_action": (
                    "Validate population estimates, eligibility criteria, "
                    "geographic coverage and recruitment assumptions."
                ),
                "requires_human_review": True,
            })
    else:
        missing_information.append(
            "A supported recruitable patient population estimate "
            "is unavailable."
        )

    # Identify additional risks only when the request explicitly supplies
    # the relevant indicators.
    competing_trials = _get_value(
        request, "competing_trials_count"
    )

    if competing_trials is not None:
        if (
            isinstance(competing_trials, bool)
            or not isinstance(competing_trials, int)
            or competing_trials < 0
        ):
            missing_information.append(
                "competing_trials_count must be a nonnegative integer."
            )
        elif competing_trials > 0:
            findings.append({
                "category": "competition",
                "severity": "medium",
                "finding": (
                    "Competing trials were reported in the recruitment area."
                ),
                "rationale": (
                    f"Reported competing trial count: {competing_trials}."
                ),
                "evidence_refs": [],
                "assumptions": [],
                "recommended_action": (
                    "Assess overlapping eligibility criteria, site workload "
                    "and expected participant competition."
                ),
                "requires_human_review": True,
            })

    result = {
        "study_id": study_id,
        "target_enrollment": target,
        "planned_months": planned_months,
        "required_monthly_enrollment": round(required_monthly, 2),
        "available_monthly_capacity": (
            round(available_monthly, 2)
            if available_monthly is not None
            else None
        ),
        "capacity_gap_per_month": (
            round(capacity_gap, 2)
            if capacity_gap is not None
            else None
        ),
        "expected_capacity_over_period": (
            round(expected_capacity, 2)
            if expected_capacity is not None
            else None
        ),
        "capacity_status": capacity_status,
        "population_status": population_status,
        "findings": findings,
        "assumptions": assumptions,
        "missing_information": list(dict.fromkeys(missing_information)),
        "requires_human_review": True,
        "limitations": [
            "Capacity arithmetic is not a validated enrollment forecast.",
            "Site capacities may not be independent or constant.",
            "Actual enrollment depends on activation, screening, eligibility, "
            "competition and recruitment execution.",
        ],
    }

    logger.info(
        "Recruitment risk assessment completed | study_id=%s | status=%s",
        study_id,
        capacity_status,
    )

    return {"recruitment_result": result}


def recruitment_risk_agent(state: dict) -> dict:
    """Run deterministic checks, then attach advisory structured LLM insights.

    Deterministic calculations and safety/review flags remain authoritative.
    """
    from app.agents.llm_enrichment import enrich_result

    output = _deterministic_recruitment_risk_agent(state)
    return enrich_result(
        state=state,
        agent_name="recruitment_risk_agent",
        output=output,
        output_key="recruitment_result",
    )
