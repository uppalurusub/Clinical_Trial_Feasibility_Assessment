
"""
Business service for Multi-Agent Clinical Trial Feasibility Assessment.

Responsibilities:
- Execute the feasibility workflow.
- Assemble and store the assessment report.
- Record human review decisions.
- Maintain basic audit events.

POC limitations:
- Assessment and audit data are stored in memory.
- Use synthetic data only.
- Replace in-memory storage with PostgreSQL before production.
"""

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from app.orchestration.graph import run_workflow


# POC-only in-memory storage.
_ASSESSMENTS: dict[str, dict] = {}
_AUDIT_EVENTS: list[dict] = []
_LOCK = RLock()


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _record_audit_event(
    event_type: str,
    assessment_id: str,
    details: dict,
) -> None:
    """
    Record an audit event without storing protocol text,
    prompts, or other potentially sensitive content.
    """
    event = {
        "event_id": str(uuid4()),
        "timestamp_utc": _utc_now(),
        "event_type": event_type,
        "assessment_id": assessment_id,
        "details": deepcopy(details),
    }

    with _LOCK:
        _AUDIT_EVENTS.append(event)


def run_assessment(request) -> dict:
    """
    Execute the workflow and create a feasibility assessment report.

    The request is expected to follow FeasibilityRequest
    from app.schemas.
    """
    # Run the specialist workflow implemented in orchestrator.py.
    result = run_workflow(request)

    assessment_id = str(uuid4())

    recruitment_result = result.get("recruitment_result", {})
    timeline_result = result.get("timeline_cost_result", {})
    critic_result = result.get("critic_result", {})

    report = {
        "assessment_id": assessment_id,
        "study_id": request.study_id,
        "study_title": request.title,
        "therapeutic_area": request.therapeutic_area,
        "status": "pending_human_review",
        "created_at_utc": _utc_now(),

        "orchestration_engine": result.get(
            "orchestration_engine",
            "unknown",
        ),

        "protocol_assessment": result.get(
            "protocol_result", {}
        ),
        "evidence": result.get("evidence", []),
        "site_rankings": result.get("site_results", []),
        "recruitment_assessment": recruitment_result,
        "timeline_cost_assessment": timeline_result,
        "critic_review": critic_result,

        "limitations": [
            "Synthetic evidence and demonstration data only.",
            "Protocol completeness checks use simple keyword matching.",
            "Site scores are illustrative and not validated.",
            "Recruitment calculations are capacity checks, not validated forecasts.",
            "Cost estimates are not generated without validated cost inputs.",
            "Human review is required; this is not trial authorization.",
        ],
    }

    with _LOCK:
        _ASSESSMENTS[assessment_id] = deepcopy(report)

    _record_audit_event(
        event_type="assessment_created",
        assessment_id=assessment_id,
        details={
            "study_id": request.study_id,
            "status": report["status"],
            "orchestration_engine": report["orchestration_engine"],
        },
    )

    return deepcopy(report)


def get_assessment(assessment_id: str) -> dict:
    """Retrieve a stored assessment by ID."""
    with _LOCK:
        report = _ASSESSMENTS.get(assessment_id)

        if report is None:
            raise KeyError(assessment_id)

        return deepcopy(report)


def record_review(assessment_id: str, request) -> dict:
    """
    Record a human review decision.

    Expected request fields:
    - decision: approved, rejected, or needs_revision
    - reviewer: reviewer identifier
    - comments: optional review comments
    """
    with _LOCK:
        report = _ASSESSMENTS.get(assessment_id)

        if report is None:
            raise KeyError(assessment_id)

        decision = request.decision

        if report["status"] != "pending_human_review":
            raise ValueError(
                "This assessment is no longer pending human review."
            )

        review = {
            "decision": decision,
            "reviewer": request.reviewer,
            "comments": request.comments,
            "reviewed_at_utc": _utc_now(),
        }

        report["review"] = review
        report["status"] = decision

        # Do not place free-text reviewer comments in audit details.
        audit_details = {
            "study_id": report["study_id"],
            "decision": decision,
            "reviewer": request.reviewer,
        }

        response = {
            "assessment_id": assessment_id,
            "status": report["status"],
            "review": deepcopy(review),
        }

    _record_audit_event(
        event_type="human_review_recorded",
        assessment_id=assessment_id,
        details=audit_details,
    )

    return response


def get_audit_events() -> list[dict]:
    """Return a snapshot of POC audit events."""
    with _LOCK:
        return deepcopy(_AUDIT_EVENTS)
