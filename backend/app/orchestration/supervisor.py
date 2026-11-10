"""
Deterministic supervisor nodes for clinical trial feasibility orchestration.

The supervisor initializes state and documents the sequence. It does not
make clinical decisions or approve an assessment.
"""
from __future__ import annotations


from typing import Any

from app.orchestration.state import FeasibilityState

from app.config.settings import settings
logger = settings.get_logger()

SPECIALIST_EXECUTION_PLAN = [
    "patient_population",
    "sites",
    "recruitment",
    "timeline_cost",
    "critic",
]


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def supervisor_node(state: FeasibilityState) -> dict[str, Any]:
    """Initialize workflow metadata and return a partial state update."""
    request = state.get("request")
    if request is None:
        raise ValueError("Supervisor requires a validated feasibility request.")

    study_id = state.get("study_id") or _get_value(request, "study_id")
    if not study_id:
        raise ValueError("A study_id is required.")

    logger.info("Supervisor initialized | study_id=%s", study_id)
    return {
        "study_id": str(study_id),
        "orchestration_engine": "langgraph",
        "workflow_status": "running",
        "review_status": "not_started",
        "execution_plan": list(SPECIALIST_EXECUTION_PLAN),
        "reanalysis_attempts": 0,
        "missing_information": [],
        "limitations": [
            "Protocol and document-evidence agent files were not supplied for this graph update.",
            "Site scoring weights and score bands are illustrative and require SME validation.",
            "Capacity calculations are not validated enrollment forecasts.",
            "Cost estimates are formula-based and are not validated trial budgets.",
            "Automated analysis does not replace authorized expert review.",
        ],
    }


def supervisor_plan(state: FeasibilityState) -> list[str]:
    """Return the execution plan documented by the supervisor."""
    if state.get("request") is None:
        raise ValueError("Cannot plan without a feasibility request.")
    return list(SPECIALIST_EXECUTION_PLAN)


def human_review_node(state: FeasibilityState) -> dict[str, Any]:
    """Mark the assessment as awaiting review; this node does not collect approval."""
    logger.info(
        "Assessment requires human review | study_id=%s",
        state.get("study_id", "unknown"),
    )
    return {
        "workflow_status": "pending_human_review",
        "review_status": "pending_human_review",
    }


def report_ready_node(state: FeasibilityState) -> dict[str, Any]:
    """Mark report assembly readiness, not human approval."""
    logger.info(
        "Assessment ready for report assembly | study_id=%s",
        state.get("study_id", "unknown"),
    )
    return {
        "workflow_status": "report_ready",
        "review_status": "pending_human_review",
    }
