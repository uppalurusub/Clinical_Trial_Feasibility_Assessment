
"""
Cost and Timeline Agent
Multi-Agent Clinical Trial Feasibility Assessment

Responsibilities:
- Assess the proposed enrollment timeline.
- Estimate duration using supplied recruitment capacity.
- Calculate indicative study costs when sufficient cost inputs exist.
- Identify timeline gaps, missing cost inputs, and assumptions.
- Return structured findings for LangGraph orchestration.

Important:
- Duration calculations are illustrative capacity calculations,
  not validated recruitment forecasts.
- Costs are calculated only from explicitly supplied cost inputs.
- This agent does not establish a validated clinical trial budget.
"""

import logging
import math
from numbers import Real
from typing import Any


from app.config.settings import settings
logger = settings.get_logger()


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a value from a dictionary or Pydantic-style object."""
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _valid_number(value: Any) -> bool:
    """Validate a finite, nonnegative numeric value."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _valid_positive_integer(value: Any) -> bool:
    """Validate a positive integer, excluding booleans."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    )


def _deterministic_cost_timeline_agent(state: dict) -> dict:
    """
    Analyze study timeline and indicative cost.

    Expected state:
        request:
            study_id
            target_enrollment
            planned_months
            optional cost_data

        recruitment_result:
            available_monthly_capacity
            capacity_status

        site_results:
            list of site assessments

    Optional request.cost_data:
        site_startup_cost
        cost_per_participant
        monthly_operating_cost
        contingency_rate
        currency

    Cost assumptions are interpreted in the same currency and
    accounting scope supplied by the user.
    """
    request = state.get("request")

    if request is None:
        raise ValueError(
            "Cost and timeline analysis requires a feasibility request."
        )

    study_id = _get_value(request, "study_id", "unknown")
    target = _get_value(request, "target_enrollment")
    planned_months = _get_value(request, "planned_months")

    if not _valid_positive_integer(target):
        raise ValueError(
            "target_enrollment must be a positive integer."
        )

    if not _valid_positive_integer(planned_months):
        raise ValueError(
            "planned_months must be a positive integer."
        )

    findings = []
    assumptions = []
    missing_information = []
    limitations = []

    required_monthly_enrollment = target / planned_months

    # ---------------------------------------------------------
    # 1. Analyze recruitment capacity and indicative duration
    # ---------------------------------------------------------
    recruitment_result = state.get("recruitment_result", {})

    if not isinstance(recruitment_result, dict):
        recruitment_result = {}
        missing_information.append(
            "Recruitment results are unavailable or malformed."
        )

    available_capacity = recruitment_result.get(
        "available_monthly_capacity"
    )

    estimated_enrollment_months = None
    estimated_timeline_status = "insufficient_data"

    if available_capacity is not None:
        if not _valid_number(available_capacity):
            missing_information.append(
                "Available monthly recruitment capacity is invalid."
            )
        elif available_capacity <= 0:
            estimated_timeline_status = "no_reported_capacity"

            findings.append({
                "category": "timeline",
                "severity": "critical",
                "finding": (
                    "The supplied recruitment capacity does not support "
                    "an enrollment duration estimate."
                ),
                "rationale": (
                    "No positive monthly enrollment capacity was supplied."
                ),
                "evidence_refs": [],
                "assumptions": [],
                "recommended_action": (
                    "Validate site activation and recruitment capacity."
                ),
                "requires_human_review": True,
            })
        else:
            estimated_enrollment_months = math.ceil(
                target / float(available_capacity)
            )

            assumptions.append(
                "Monthly site capacity is assumed to remain constant "
                "throughout enrollment."
            )

            if estimated_enrollment_months > planned_months:
                estimated_timeline_status = "potential_timeline_gap"

                findings.append({
                    "category": "timeline",
                    "severity": "high",
                    "finding": (
                        "The capacity-based enrollment duration exceeds "
                        "the planned recruitment period."
                    ),
                    "rationale": (
                        f"Illustrative enrollment duration: "
                        f"{estimated_enrollment_months} months; "
                        f"planned duration: {planned_months} months."
                    ),
                    "evidence_refs": [],
                    "assumptions": list(assumptions),
                    "recommended_action": (
                        "Validate site-level enrollment rates and evaluate "
                        "additional sites, recruitment strategies, or "
                        "a revised timeline."
                    ),
                    "requires_human_review": True,
                })
            else:
                estimated_timeline_status = "capacity_within_planned_period"

                findings.append({
                    "category": "timeline",
                    "severity": "low",
                    "finding": (
                        "The illustrative capacity calculation falls "
                        "within the planned enrollment period."
                    ),
                    "rationale": (
                        "This does not account for site activation delays, "
                        "screen failures, pauses, or variation in enrollment."
                    ),
                    "evidence_refs": [],
                    "assumptions": list(assumptions),
                    "recommended_action": (
                        "Validate the calculation against historical "
                        "site performance and the study schedule."
                    ),
                    "requires_human_review": True,
                })
    else:
        missing_information.append(
            "Available monthly recruitment capacity is required "
            "for a capacity-based timeline estimate."
        )

    # ---------------------------------------------------------
    # 2. Estimate cost using explicit assumptions
    # ---------------------------------------------------------
    cost_data = _get_value(request, "cost_data", {}) or {}

    if not isinstance(cost_data, dict):
        cost_data = {}
        missing_information.append(
            "Cost data must be supplied as a dictionary."
        )

    site_results = state.get("site_results", [])

    if not isinstance(site_results, list):
        site_results = []

    # Only valid site IDs count as proposed sites for startup-cost
    # calculations. The cost assumptions must still match the
    # intended accounting scope.
    proposed_site_count = sum(
        1
        for site in site_results
        if isinstance(site, dict) and site.get("site_id")
    )

    site_startup_cost = cost_data.get("site_startup_cost")
    cost_per_participant = cost_data.get("cost_per_participant")
    monthly_operating_cost = cost_data.get("monthly_operating_cost")
    contingency_rate = cost_data.get("contingency_rate", 0.0)
    currency = cost_data.get("currency")

    invalid_cost_fields = []

    for field_name, value in (
        ("site_startup_cost", site_startup_cost),
        ("cost_per_participant", cost_per_participant),
        ("monthly_operating_cost", monthly_operating_cost),
        ("contingency_rate", contingency_rate),
    ):
        if value is not None and not _valid_number(value):
            invalid_cost_fields.append(field_name)

    if _valid_number(contingency_rate) and contingency_rate > 1:
        invalid_cost_fields.append("contingency_rate")

    for field_name in invalid_cost_fields:
        missing_information.append(
            f"{field_name} is invalid; provide a finite, nonnegative "
            "value and use a contingency_rate between 0 and 1."
        )

    valid_startup_cost = (
        site_startup_cost is not None
        and _valid_number(site_startup_cost)
        and "site_startup_cost" not in invalid_cost_fields
    )

    valid_participant_cost = (
        cost_per_participant is not None
        and _valid_number(cost_per_participant)
        and "cost_per_participant" not in invalid_cost_fields
    )

    valid_monthly_cost = (
        monthly_operating_cost is not None
        and _valid_number(monthly_operating_cost)
        and "monthly_operating_cost" not in invalid_cost_fields
    )

    valid_contingency = (
        _valid_number(contingency_rate)
        and contingency_rate <= 1
        and "contingency_rate" not in invalid_cost_fields
    )

    # The calculation requires all three core cost components.
    cost_estimate = None
    cost_breakdown = None

    if not valid_startup_cost:
        missing_information.append(
            "A valid site_startup_cost is required for the cost estimate."
        )

    if not valid_participant_cost:
        missing_information.append(
            "A valid cost_per_participant is required for the cost estimate."
        )

    if not valid_monthly_cost:
        missing_information.append(
            "A valid monthly_operating_cost is required for the cost estimate."
        )

    if proposed_site_count <= 0:
        missing_information.append(
            "At least one site result with a site_id is required "
            "to calculate startup costs."
        )

    if (
        valid_startup_cost
        and valid_participant_cost
        and valid_monthly_cost
        and valid_contingency
        and proposed_site_count > 0
        and estimated_enrollment_months is not None
    ):
        startup_total = (
            float(site_startup_cost) * proposed_site_count
        )

        participant_total = (
            float(cost_per_participant) * target
        )

        operating_total = (
            float(monthly_operating_cost)
            * estimated_enrollment_months
        )

        subtotal = (
            startup_total
            + participant_total
            + operating_total
        )

        contingency_amount = subtotal * float(contingency_rate)
        cost_estimate = round(subtotal + contingency_amount, 2)

        cost_breakdown = {
            "site_startup_total": round(startup_total, 2),
            "participant_cost_total": round(participant_total, 2),
            "operating_cost_total": round(operating_total, 2),
            "subtotal": round(subtotal, 2),
            "contingency_rate": float(contingency_rate),
            "contingency_amount": round(contingency_amount, 2),
            "estimated_total": cost_estimate,
            "currency": currency,
            "site_count_used": proposed_site_count,
            "estimated_enrollment_months_used": (
                estimated_enrollment_months
            ),
        }

        assumptions.append(
            "The startup cost is assumed to apply independently to each "
            "proposed site."
        )

        assumptions.append(
            "Participant costs are assumed to apply to every target "
            "participant, and monthly operating costs to every estimated "
            "enrollment month."
        )

        assumptions.append(
            "All cost components are assumed to use the same currency, "
            "price basis, and accounting scope."
        )

        findings.append({
            "category": "cost",
            "severity": "unknown",
            "finding": (
                "An indicative cost estimate was calculated from "
                "the supplied assumptions."
            ),
            "rationale": (
                "The result is a formula-based estimate, not a validated "
                "trial budget or vendor quotation."
            ),
            "evidence_refs": [],
            "assumptions": list(assumptions),
            "recommended_action": (
                "Validate assumptions with clinical operations, finance, "
                "site budgets, and applicable vendor estimates."
            ),
            "requires_human_review": True,
        })
    else:
        limitations.append(
            "No complete cost estimate was calculated because one or more "
            "required cost inputs or timeline inputs are unavailable."
        )

    # ---------------------------------------------------------
    # 3. Return the shared workflow result
    # ---------------------------------------------------------
    result = {
        "study_id": study_id,
        "target_enrollment": target,
        "planned_months": planned_months,
        "required_monthly_enrollment": round(
            required_monthly_enrollment, 2
        ),
        "available_monthly_capacity": (
            float(available_capacity)
            if (
                available_capacity is not None
                and _valid_number(available_capacity)
            )
            else None
        ),
        "estimated_enrollment_months": estimated_enrollment_months,
        "timeline_status": estimated_timeline_status,
        "estimated_cost": cost_estimate,
        "cost_breakdown": cost_breakdown,
        "currency": currency,
        "findings": findings,
        "assumptions": assumptions,
        "missing_information": list(
            dict.fromkeys(missing_information)
        ),
        "limitations": limitations + [
            "The timeline calculation assumes constant monthly capacity.",
            "The cost calculation excludes any cost categories not explicitly "
            "represented by the supplied cost inputs.",
            "No contingency or inflation assumption is validated by this agent.",
        ],
        "requires_human_review": True,
    }

    logger.info(
        "Cost and timeline analysis completed | study_id=%s | "
        "timeline_status=%s | cost_estimated=%s",
        study_id,
        estimated_timeline_status,
        cost_estimate is not None,
    )

    return {"timeline_cost_result": result}


def cost_timeline_agent(state: dict) -> dict:
    """Run deterministic checks, then attach advisory structured LLM insights.

    Deterministic calculations and safety/review flags remain authoritative.
    """
    from app.agents.llm_enrichment import enrich_result

    output = _deterministic_cost_timeline_agent(state)
    return enrich_result(
        state=state,
        agent_name="cost_timeline_agent",
        output=output,
        output_key="timeline_cost_result",
    )
