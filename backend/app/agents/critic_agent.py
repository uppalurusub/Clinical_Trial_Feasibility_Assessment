
"""
Critic Agent
Multi-Agent Clinical Trial Feasibility Assessment

Responsibilities:
- Cross-check specialist outputs for contradictions.
- Validate recruitment and timeline arithmetic.
- Identify missing information and unsupported findings.
- Flag evidence gaps and incomplete agent results.
- Produce a structured quality-control result for LangGraph routing.

This is a deterministic critic. It does not independently validate
clinical evidence or replace review by clinical research experts.
"""

import logging
import math
from numbers import Real
from typing import Any

from app.config.settings import settings
logger = settings.get_logger()


def _valid_number(value: Any) -> bool:
    """Check for finite numeric values, excluding booleans."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_nonnegative_number(value: Any) -> bool:
    return _valid_number(value) and value >= 0


def _add_issue(
    issues: list[dict],
    *,
    category: str,
    severity: str,
    finding: str,
    rationale: str,
    recommended_action: str,
    evidence_refs: list | None = None,
) -> None:
    """Append a consistently structured quality-control finding."""
    issues.append({
        "category": category,
        "severity": severity,
        "finding": finding,
        "rationale": rationale,
        "recommended_action": recommended_action,
        "evidence_refs": evidence_refs or [],
        "requires_human_review": True,
    })


def _collect_findings(state: dict) -> list[dict]:
    """Collect findings produced by specialist agents."""
    collected = []

    result_keys = (
        "protocol_result",
        "population_result",
        "site_assessment_summary",
        "recruitment_result",
        "timeline_cost_result",
    )

    for key in result_keys:
        result = state.get(key)

        if result is None:
            continue

        if not isinstance(result, dict):
            continue

        findings = result.get("findings", [])

        if not isinstance(findings, list):
            continue

        for finding in findings:
            if isinstance(finding, dict):
                collected.append(finding)

    for site in state.get("site_results", []):
        if not isinstance(site, dict):
            continue

        for finding in site.get("findings", []):
            if isinstance(finding, dict):
                collected.append(finding)

    return collected


def _deterministic_critic_agent(state: dict) -> dict:
    """
    Inspect the shared feasibility state and return critic_result.

    Expected inputs:
        request
        protocol_result
        population_result
        site_results
        recruitment_result
        timeline_cost_result

    Returns:
        {"critic_result": {...}}
    """
    request = state.get("request")

    if request is None:
        raise ValueError(
            "Critic agent requires the original feasibility request."
        )

    issues: list[dict] = []
    unsupported_claims: list[str] = []
    missing_information: list[str] = []

    target = getattr(request, "target_enrollment", None)
    planned_months = getattr(request, "planned_months", None)

    if isinstance(request, dict):
        target = request.get("target_enrollment", target)
        planned_months = request.get("planned_months", planned_months)

    # ---------------------------------------------------------
    # 1. Check whether specialist outputs are available
    # ---------------------------------------------------------
    required_results = (
        "protocol_result",
        "site_results",
        "recruitment_result",
        "timeline_cost_result",
    )

    for key in required_results:
        if key not in state:
            missing_information.append(
                f"Specialist output is missing: {key}."
            )
        elif not isinstance(state[key], (dict, list)):
            missing_information.append(
                f"Specialist output has an invalid structure: {key}."
            )

    # Population analysis may be absent in earlier versions of the POC.
    if "population_result" not in state:
        missing_information.append(
            "Patient population analysis was not supplied to the critic."
        )

    # ---------------------------------------------------------
    # 2. Recruitment arithmetic consistency
    # ---------------------------------------------------------
    recruitment = state.get("recruitment_result", {})
    if not isinstance(recruitment, dict):
        recruitment = {}

    if (
        isinstance(target, int)
        and not isinstance(target, bool)
        and target > 0
        and isinstance(planned_months, int)
        and not isinstance(planned_months, bool)
        and planned_months > 0
    ):
        required_rate = target / planned_months
        reported_rate = recruitment.get(
            "required_monthly_enrollment"
        )

        if reported_rate is None:
            missing_information.append(
                "Required monthly enrollment rate is missing."
            )
        elif not _valid_nonnegative_number(reported_rate):
            _add_issue(
                issues,
                category="recruitment_consistency",
                severity="high",
                finding="Required monthly enrollment rate is invalid.",
                rationale=(
                    "The recruitment agent returned a nonnumeric, negative, "
                    "or non-finite rate."
                ),
                recommended_action=(
                    "Recalculate the required rate from target enrollment "
                    "and planned months."
                ),
            )
        elif not math.isclose(
            float(reported_rate),
            required_rate,
            rel_tol=1e-6,
            abs_tol=1e-4,
        ):
            _add_issue(
                issues,
                category="recruitment_consistency",
                severity="high",
                finding=(
                    "Recruitment agent rate conflicts with the study inputs."
                ),
                rationale=(
                    f"Expected {required_rate:.2f} participants/month, "
                    f"but received {float(reported_rate):.2f}."
                ),
                recommended_action=(
                    "Reconcile the enrollment target, planned duration, "
                    "and recruitment calculation."
                ),
            )

        available_capacity = recruitment.get(
            "available_monthly_capacity"
        )

        if available_capacity is not None:
            if not _valid_nonnegative_number(available_capacity):
                _add_issue(
                    issues,
                    category="recruitment_consistency",
                    severity="high",
                    finding="Available monthly recruitment capacity is invalid.",
                    rationale=(
                        "Capacity must be a finite, nonnegative number."
                    ),
                    recommended_action=(
                        "Validate the site-level capacities and recalculate "
                        "the combined monthly capacity."
                    ),
                )
            elif float(available_capacity) < required_rate:
                _add_issue(
                    issues,
                    category="recruitment_timeline_conflict",
                    severity="high",
                    finding=(
                        "Reported site capacity is below the required "
                        "enrollment rate."
                    ),
                    rationale=(
                        f"Required rate: {required_rate:.2f}/month; "
                        f"reported capacity: "
                        f"{float(available_capacity):.2f}/month."
                    ),
                    recommended_action=(
                        "Review the recruitment assumptions, site capacity, "
                        "number of sites, and planned duration."
                    ),
                )

    # ---------------------------------------------------------
    # 3. Patient population consistency
    # ---------------------------------------------------------
    population = state.get("population_result", {})

    if isinstance(population, dict):
        eligible = population.get("eligible_population_estimate")
        recruitable = population.get("recruitable_population_estimate")

        if (
            eligible is not None
            and _valid_nonnegative_number(eligible)
            and recruitable is not None
            and _valid_nonnegative_number(recruitable)
            and float(recruitable) > float(eligible)
        ):
            _add_issue(
                issues,
                category="population_consistency",
                severity="high",
                finding=(
                    "Recruitable population exceeds the supplied "
                    "eligible population estimate."
                ),
                rationale=(
                    f"Eligible estimate: {eligible}; "
                    f"recruitable estimate: {recruitable}."
                ),
                recommended_action=(
                    "Validate the population definitions and recruitment "
                    "fraction before using these estimates."
                ),
            )

        if (
            isinstance(target, int)
            and not isinstance(target, bool)
            and target > 0
            and recruitable is not None
            and _valid_nonnegative_number(recruitable)
            and float(recruitable) < target
        ):
            _add_issue(
                issues,
                category="population_feasibility",
                severity="high",
                finding=(
                    "Supplied recruitable population is below target enrollment."
                ),
                rationale=(
                    f"Recruitable estimate: {recruitable}; "
                    f"target enrollment: {target}."
                ),
                recommended_action=(
                    "Validate the estimate, eligibility assumptions, "
                    "and recruitment geography."
                ),
            )

    # ---------------------------------------------------------
    # 4. Site data quality and capacity aggregation
    # ---------------------------------------------------------
    sites = state.get("site_results", [])

    if isinstance(sites, list) and sites:
        site_ids = [
            site.get("site_id")
            for site in sites
            if isinstance(site, dict) and site.get("site_id")
        ]

        if len(site_ids) != len(set(site_ids)):
            _add_issue(
                issues,
                category="site_data_quality",
                severity="high",
                finding="Duplicate site identifiers were detected.",
                rationale=(
                    "Duplicate site records can inflate the reported "
                    "combined recruitment capacity."
                ),
                recommended_action=(
                    "Deduplicate site records before aggregating capacity."
                ),
            )

        valid_capacities = [
            float(site["capacity_per_month"])
            for site in sites
            if (
                isinstance(site, dict)
                and _valid_nonnegative_number(
                    site.get("capacity_per_month")
                )
            )
        ]

        recruitment_capacity = recruitment.get(
            "available_monthly_capacity"
        )

        if (
            valid_capacities
            and _valid_nonnegative_number(recruitment_capacity)
        ):
            summed_capacity = sum(valid_capacities)

            if not math.isclose(
                summed_capacity,
                float(recruitment_capacity),
                rel_tol=1e-6,
                abs_tol=1e-4,
            ):
                _add_issue(
                    issues,
                    category="site_capacity_consistency",
                    severity="high",
                    finding=(
                        "Recruitment capacity does not match the sum of "
                        "valid site-level capacities."
                    ),
                    rationale=(
                        f"Site sum: {summed_capacity:.2f}/month; "
                        f"recruitment result: "
                        f"{float(recruitment_capacity):.2f}/month."
                    ),
                    recommended_action=(
                        "Reconcile site inclusion rules and capacity inputs."
                    ),
                )

    # ---------------------------------------------------------
    # 5. Timeline consistency
    # ---------------------------------------------------------
    timeline = state.get("timeline_cost_result", {})

    if isinstance(timeline, dict):
        estimated_months = timeline.get(
            "estimated_enrollment_months"
        )

        reported_capacity = timeline.get(
            "available_monthly_capacity"
        )

        if (
            isinstance(target, int)
            and not isinstance(target, bool)
            and target > 0
            and _valid_number(reported_capacity)
            and reported_capacity > 0
            and estimated_months is not None
        ):
            expected_months = math.ceil(
                target / float(reported_capacity)
            )

            if (
                isinstance(estimated_months, bool)
                or not isinstance(estimated_months, int)
                or estimated_months != expected_months
            ):
                _add_issue(
                    issues,
                    category="timeline_consistency",
                    severity="high",
                    finding=(
                        "The timeline estimate conflicts with the "
                        "reported target and monthly capacity."
                    ),
                    rationale=(
                        f"Expected illustrative duration: "
                        f"{expected_months} months; "
                        f"reported duration: {estimated_months}."
                    ),
                    recommended_action=(
                        "Recalculate the capacity-based duration and "
                        "check the source inputs."
                    ),
                )

    # ---------------------------------------------------------
    # 6. Evidence and assumption checks
    # ---------------------------------------------------------
    for finding in _collect_findings(state):
        text = finding.get("finding", "Unnamed finding")
        evidence_refs = finding.get("evidence_refs", [])

        if not isinstance(evidence_refs, list):
            unsupported_claims.append(
                f"{text}: evidence references are malformed."
            )
        elif not evidence_refs:
            unsupported_claims.append(
                f"{text}: no evidence reference was supplied."
            )

    for key in (
        "protocol_result",
        "population_result",
        "recruitment_result",
        "timeline_cost_result",
    ):
        result = state.get(key)

        if not isinstance(result, dict):
            continue

        missing = result.get("missing_information", [])

        if isinstance(missing, list):
            for item in missing:
                if isinstance(item, str) and item.strip():
                    missing_information.append(
                        f"{key}: {item.strip()}"
                    )

    missing_information = list(dict.fromkeys(missing_information))
    unsupported_claims = list(dict.fromkeys(unsupported_claims))

    # ---------------------------------------------------------
    # 7. Determine whether human review is required
    # ---------------------------------------------------------
    critical_or_high_issues = any(
        issue["severity"] in ("critical", "high")
        for issue in issues
    )

    issues_found = bool(issues or unsupported_claims or missing_information)

    # This POC always requires a human reviewer. A clean critic result
    # means only that no issue was detected by these checks.
    review_required = True

    if critical_or_high_issues:
        summary = (
            "High-priority feasibility or consistency issues were detected. "
            "Expert review is required."
        )
    elif issues_found:
        summary = (
            "Potential evidence gaps, missing inputs, or quality issues "
            "were detected. Expert review is required."
        )
    else:
        summary = (
            "No issues were detected by the configured deterministic checks. "
            "This does not establish clinical correctness or trial feasibility."
        )

    critic_result = {
        "issues_found": issues_found,
        "contradictions": issues,
        "unsupported_claims": unsupported_claims,
        "missing_information": missing_information,
        "review_required": review_required,
        "summary": summary,
        "checks_performed": [
            "Required specialist outputs",
            "Required monthly enrollment arithmetic",
            "Recruitment capacity versus required rate",
            "Population estimate consistency",
            "Site identifier uniqueness",
            "Site capacity aggregation",
            "Capacity-based timeline arithmetic",
            "Evidence reference presence",
            "Missing information aggregation",
        ],
        "limitations": [
            "Evidence reference presence does not establish source validity.",
            "These checks do not validate clinical assumptions or source content.",
            "A clean result is not clinical approval.",
        ],
    }

    logger.info(
        "Critic analysis completed | study_id=%s | issues_found=%s | "
        "issue_count=%d | missing_count=%d",
        (
            request.get("study_id", "unknown")
            if isinstance(request, dict)
            else getattr(request, "study_id", "unknown")
        ),
        issues_found,
        len(issues),
        len(missing_information),
    )

    return {"critic_result": critic_result}


def critic_agent(state: dict) -> dict:
    """Run deterministic checks, then attach advisory structured LLM insights.

    Deterministic calculations and safety/review flags remain authoritative.
    """
    from app.agents.llm_enrichment import enrich_result

    output = _deterministic_critic_agent(state)
    return enrich_result(
        state=state,
        agent_name="critic_agent",
        output=output,
        output_key="critic_result",
    )
