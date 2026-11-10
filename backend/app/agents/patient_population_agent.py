
"""
Patient Population Agent
Multi-Agent Clinical Trial Feasibility Assessment

Responsibilities:
- Inspect supplied patient population data.
- Distinguish estimated, eligible, and recruitable populations.
- Calculate potential recruitment availability when inputs support it.
- Identify missing assumptions and population feasibility risks.
- Return structured findings for LangGraph orchestration.

This agent does not independently establish clinical eligibility
or provide validated epidemiological predictions.
"""

import logging
from numbers import Real
from typing import Any

from app.config.settings import settings
logger = settings.get_logger()


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dictionary or Pydantic-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _as_dict(value: Any) -> dict:
    """Normalize dictionary-like inputs without guessing their contents."""
    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        result = value.model_dump()
        return result if isinstance(result, dict) else {}

    return {}


def _valid_nonnegative_number(value: Any) -> bool:
    """Accept finite, nonnegative numeric values, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False

    try:
        return value >= 0 and value < float("inf")
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_fraction(value: Any) -> bool:
    """Validate a proportion between zero and one, inclusive."""
    return (
        _valid_nonnegative_number(value)
        and value <= 1
    )


def _deterministic_patient_population_agent(state: dict) -> dict:
    """
    Analyze supplied patient population information.

    Expected optional request fields:
        target_enrollment: int
        population_data: dict

    population_data may contain:
        geographic_population
        prevalence_rate
        eligible_population_estimate
        recruitable_population_estimate
        recruitment_fraction
        evidence_refs
        assumptions
        data_source

    Returns:
        A partial LangGraph state containing population_result.

    Missing data is reported explicitly rather than imputed.
    """
    request = state.get("request")

    if request is None:
        raise ValueError(
            "Patient population analysis requires a feasibility request."
        )

    study_id = _get_value(request, "study_id", "unknown")
    target_enrollment = _get_value(request, "target_enrollment")
    population_data = _as_dict(
        _get_value(request, "population_data", {})
    )

    findings = []
    assumptions = []
    missing_information = []
    limitations = []

    evidence_refs = population_data.get("evidence_refs", [])
    if not isinstance(evidence_refs, list):
        evidence_refs = []
        missing_information.append(
            "Population evidence references must be supplied as a list."
        )

    # Validate the required enrollment target.
    if (
        isinstance(target_enrollment, bool)
        or not isinstance(target_enrollment, int)
        or target_enrollment <= 0
    ):
        missing_information.append(
            "A positive target enrollment is required."
        )
        target_enrollment = None

    # Extract explicitly supplied population estimates.
    geographic_population = population_data.get(
        "geographic_population"
    )
    prevalence_rate = population_data.get("prevalence_rate")
    eligible_population = population_data.get(
        "eligible_population_estimate"
    )
    recruitable_population = population_data.get(
        "recruitable_population_estimate"
    )
    recruitment_fraction = population_data.get(
        "recruitment_fraction"
    )

    # Validate the geographic denominator, if provided.
    if geographic_population is not None:
        if not _valid_nonnegative_number(geographic_population):
            missing_information.append(
                "Geographic population must be a nonnegative number."
            )
            geographic_population = None

    # Estimate disease population only when both inputs are supplied.
    disease_population_estimate = None

    if prevalence_rate is not None:
        if not _valid_fraction(prevalence_rate):
            missing_information.append(
                "Prevalence rate must be a proportion between 0 and 1."
            )
            prevalence_rate = None

    if (
        geographic_population is not None
        and prevalence_rate is not None
    ):
        disease_population_estimate = (
            geographic_population * prevalence_rate
        )
        assumptions.append(
            "The supplied prevalence rate applies to the specified "
            "geographic population and reference period."
        )
    elif (
        geographic_population is not None
        or prevalence_rate is not None
    ):
        missing_information.append(
            "Both geographic population and prevalence rate are required "
            "to calculate a disease population estimate."
        )

    # Validate eligible population estimate.
    if eligible_population is not None:
        if not _valid_nonnegative_number(eligible_population):
            missing_information.append(
                "Eligible population estimate must be nonnegative."
            )
            eligible_population = None

    # Validate recruitable population estimate.
    if recruitable_population is not None:
        if not _valid_nonnegative_number(recruitable_population):
            missing_information.append(
                "Recruitable population estimate must be nonnegative."
            )
            recruitable_population = None

    # Calculate a recruitable estimate only if an explicit eligible
    # population and a documented recruitment fraction are supplied.
    if (
        recruitable_population is None
        and eligible_population is not None
        and recruitment_fraction is not None
    ):
        if _valid_fraction(recruitment_fraction):
            recruitable_population = (
                eligible_population * recruitment_fraction
            )
            assumptions.append(
                "The supplied recruitment fraction applies to the "
                "eligible population for this trial's recruitment period."
            )
        else:
            missing_information.append(
                "Recruitment fraction must be a proportion between 0 and 1."
            )

    # Report missing inputs needed for a meaningful assessment.
    if eligible_population is None:
        missing_information.append(
            "An eligibility-adjusted patient population estimate is needed."
        )

    if recruitable_population is None:
        missing_information.append(
            "A supported recruitable population estimate is unavailable."
        )

    # Evaluate the target only when a comparable recruitable estimate
    # is available. This is a screening check, not a forecast.
    population_status = "insufficient_data"

    if (
        target_enrollment is not None
        and recruitable_population is not None
    ):
        if recruitable_population < target_enrollment:
            population_status = "potential_population_gap"

            findings.append({
                "category": "patient_population",
                "severity": "high",
                "finding": (
                    "The supplied recruitable population estimate is "
                    "below the target enrollment."
                ),
                "rationale": (
                    f"Estimated recruitable population: "
                    f"{recruitable_population:g}; target enrollment: "
                    f"{target_enrollment}."
                ),
                "evidence_refs": list(evidence_refs),
                "assumptions": list(assumptions),
                "recommended_action": (
                    "Validate eligibility assumptions, recruitment "
                    "geography, data sources, and the recruitment fraction."
                ),
                "requires_human_review": True,
            })

        else:
            population_status = "population_estimate_meets_target"

            findings.append({
                "category": "patient_population",
                "severity": "unknown",
                "finding": (
                    "The supplied recruitable population estimate meets "
                    "or exceeds target enrollment."
                ),
                "rationale": (
                    "This arithmetic comparison does not establish "
                    "that the target can be recruited within the planned "
                    "timeline."
                ),
                "evidence_refs": list(evidence_refs),
                "assumptions": list(assumptions),
                "recommended_action": (
                    "Validate site access, competing trials, enrollment "
                    "rates, eligibility criteria, and recruitment duration."
                ),
                "requires_human_review": True,
            })

    if not population_data:
        missing_information.append(
            "No population_data was supplied with the request."
        )

    if not evidence_refs:
        limitations.append(
            "No population evidence references were supplied."
        )

    limitations.extend([
        "Estimates depend on the supplied data and assumptions.",
        "Disease prevalence does not equal protocol-eligible population.",
        "Eligible population does not equal realistically recruitable patients.",
        "This agent does not validate epidemiological sources or forecast "
        "actual enrollment performance.",
    ])

    result = {
        "population_result": {
            "study_id": study_id,
            "target_enrollment": target_enrollment,
            "geographic_population": geographic_population,
            "prevalence_rate": prevalence_rate,
            "disease_population_estimate": disease_population_estimate,
            "eligible_population_estimate": eligible_population,
            "recruitable_population_estimate": recruitable_population,
            "population_status": population_status,
            "findings": findings,
            "evidence_refs": list(evidence_refs),
            "assumptions": assumptions,
            "missing_information": list(dict.fromkeys(missing_information)),
            "limitations": limitations,
            "requires_human_review": True,
        }
    }

    logger.info(
        "Patient population analysis completed | study_id=%s | status=%s",
        study_id,
        population_status,
    )

    return result


def patient_population_agent(state: dict) -> dict:
    """Run deterministic checks, then attach advisory structured LLM insights.

    Deterministic calculations and safety/review flags remain authoritative.
    """
    from app.agents.llm_enrichment import enrich_result

    output = _deterministic_patient_population_agent(state)
    return enrich_result(
        state=state,
        agent_name="patient_population_agent",
        output=output,
        output_key="population_result",
    )
