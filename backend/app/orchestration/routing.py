"""
Fail-safe conditional routing for the feasibility workflow.

No reanalysis loop is wired because the supplied project files do not
include a targeted reanalysis node. The supplied critic agent currently
sets review_required=True for every run, so the normal outcome is human
review rather than automatic report readiness.
"""
from __future__ import annotations


from typing import Any

from app.orchestration.state import FeasibilityState

from app.config.settings import settings
logger = settings.get_logger()


def route_after_supervisor(state: FeasibilityState) -> str:
    """Route to the first available specialist or escalate malformed input."""
    if state.get("request") is None:
        logger.warning("Request missing; routing to human review.")
        return "human_review"
    return "patient_population"


def route_after_critic(state: FeasibilityState) -> str:
    """
    Return 'human_review' or 'report_ready'.

    Missing/malformed critic output, detected issues, unsupported claims,
    missing information, or an explicit review requirement all escalate.
    """
    critic = state.get("critic_result")
    if not isinstance(critic, dict):
        logger.warning("Critic result missing or malformed.")
        return "human_review"

    contradictions = critic.get("contradictions", [])
    unsupported = critic.get("unsupported_claims", [])
    missing = critic.get("missing_information", [])
    review_required = critic.get("review_required", True)
    issues_found = critic.get("issues_found", True)

    if not isinstance(contradictions, list):
        return "human_review"
    if not isinstance(unsupported, list) or not isinstance(missing, list):
        return "human_review"
    if not isinstance(review_required, bool) or not isinstance(issues_found, bool):
        return "human_review"
    if any(not isinstance(item, dict) for item in contradictions):
        return "human_review"

    critical_issue = any(
        item.get("severity") == "critical" for item in contradictions
    )
    unresolved = (
        critical_issue
        or bool(contradictions)
        or bool(unsupported)
        or bool(missing)
        or review_required
        or issues_found
    )
    if unresolved:
        logger.info(
            "Assessment requires human review | study_id=%s",
            state.get("study_id", "unknown"),
        )
        return "human_review"

    logger.info(
        "Critic checks passed; report may be assembled for review | study_id=%s",
        state.get("study_id", "unknown"),
    )
    return "report_ready"


def route_after_reanalysis(state: FeasibilityState) -> str:
    """Reanalysis is not wired; safely escalate instead of routing to a missing node."""
    return "human_review"


def route_after_human_review(state: FeasibilityState) -> str:
    """Only an explicit approved decision can proceed to report readiness."""
    return "report_ready" if state.get("review_status") == "approved" else "end"


def route_after_validation(state: FeasibilityState) -> str:
    """Route malformed or incomplete specialist state to human review."""
    if state.get("workflow_error"):
        return "human_review"
    if not isinstance(state.get("site_results"), list):
        return "human_review"
    if not isinstance(state.get("population_result"), dict):
        return "human_review"
    return "critic"
