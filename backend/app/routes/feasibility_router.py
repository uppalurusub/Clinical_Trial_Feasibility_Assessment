


from fastapi import APIRouter, HTTPException, status

from app.schemas.feasibility_request import FeasibilityRequest, ReviewRequest
from app.services.feasibility_service import (
    get_audit_events,
    record_review,
    run_assessment,
)

from app.config.settings import settings
logger = settings.get_logger()

router = APIRouter(tags=["Clinical Trial Feasibility"])


@router.post(
    "/api/v1/feasibility/assess",
    status_code=status.HTTP_200_OK,
    summary="Run a clinical trial feasibility assessment",
)
def assess_feasibility(request: FeasibilityRequest):
    """
    Validate the feasibility payload using FeasibilityRequest, execute
    the feasibility workflow, and return the assessment report.

    The current POC uses synthetic evidence and illustrative calculations.
    """
    try:
        return run_assessment(request)

    except Exception:
        logger.exception(
            "Feasibility assessment failed | study_id=%s",
            request.study_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Assessment failed. Check server logs for diagnostic details.",
        ) from None


@router.post(
    "/api/v1/feasibility/{assessment_id}/review",
    summary="Record a human review decision",
)
def review_feasibility(
    assessment_id: str,
    request: ReviewRequest,
):
    """
    Record an approval, rejection, or request for revision.
    """
    try:
        return record_review(assessment_id, request)

    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Assessment not found.",
        ) from None

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from None

    except Exception:
        logger.exception(
            "Feasibility review failed | assessment_id=%s",
            assessment_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to record the review.",
        ) from None


@router.get(
    "/api/v1/audit",
    summary="Retrieve feasibility audit events",
)
def list_audit_events():
    """
    Return audit events for the current POC.

    Production deployments must enforce authorization and use durable,
    access-controlled audit storage.
    """
    try:
        return {"events": get_audit_events()}

    except Exception:
        logger.exception("Failed to retrieve feasibility audit events")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve audit events.",
        ) from None
