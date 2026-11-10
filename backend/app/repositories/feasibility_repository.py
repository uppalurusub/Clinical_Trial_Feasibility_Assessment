
"""
Repository data contracts for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Compatible with Pydantic v2.

These models define the data exchanged between the service layer and
repository implementations. They do not execute database queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


RepositoryStatus = Literal[
    "success",
    "not_found",
    "conflict",
    "failure",
]


class RepositoryModel(BaseModel):
    """Common repository contract configuration."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class RepositoryResult(RepositoryModel):
    """Standard result returned by repository operations."""

    status: RepositoryStatus
    message: str = Field(default="", max_length=2000)
    record_id: Optional[str] = Field(default=None, max_length=100)


class AssessmentRecord(RepositoryModel):
    """Persistable assessment record."""

    assessment_id: str = Field(..., min_length=1, max_length=100)
    study_id: str = Field(..., min_length=1, max_length=100)

    title: str = Field(..., min_length=1, max_length=300)
    therapeutic_area: str = Field(..., min_length=1, max_length=150)

    assessment_status: Literal[
        "pending",
        "running",
        "completed",
        "partial",
        "failed",
    ] = "pending"

    feasibility_status: Literal[
        "potentially_feasible",
        "feasible_with_risks",
        "potentially_not_feasible",
        "insufficient_data",
        "pending_human_review",
    ] = "insufficient_data"

    report_json: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    version: int = Field(default=1, ge=1)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Timestamps must include a timezone.")
        return value.astimezone(timezone.utc)


class AssessmentCreate(RepositoryModel):
    """Input contract for creating a new assessment record."""

    assessment_id: str = Field(..., min_length=1, max_length=100)
    study_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=300)
    therapeutic_area: str = Field(..., min_length=1, max_length=150)

    report_json: dict[str, Any] = Field(default_factory=dict)


class AssessmentUpdate(RepositoryModel):
    """Partial update contract for an existing assessment."""

    assessment_status: Optional[
        Literal["pending", "running", "completed", "partial", "failed"]
    ] = None

    feasibility_status: Optional[
        Literal[
            "potentially_feasible",
            "feasible_with_risks",
            "potentially_not_feasible",
            "insufficient_data",
            "pending_human_review",
        ]
    ] = None

    report_json: Optional[dict[str, Any]] = None

    # Optimistic concurrency: supply the version last read.
    expected_version: int = Field(..., ge=1)


class ReviewRecord(RepositoryModel):
    """Persisted human review decision."""

    review_id: str = Field(..., min_length=1, max_length=100)
    assessment_id: str = Field(..., min_length=1, max_length=100)

    reviewer_id: str = Field(..., min_length=1, max_length=200)

    decision: Literal[
        "approved",
        "rejected",
        "changes_requested",
    ]

    comments: Optional[str] = Field(default=None, max_length=4000)

    reviewed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("reviewed_at")
    @classmethod
    def normalize_review_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("reviewed_at must include a timezone.")
        return value.astimezone(timezone.utc)


class ReviewCreate(RepositoryModel):
    """Input for recording a human review decision."""

    review_id: str = Field(..., min_length=1, max_length=100)
    assessment_id: str = Field(..., min_length=1, max_length=100)
    reviewer_id: str = Field(..., min_length=1, max_length=200)

    decision: Literal[
        "approved",
        "rejected",
        "changes_requested",
    ]

    comments: Optional[str] = Field(default=None, max_length=4000)

    expected_assessment_version: int = Field(..., ge=1)


class AuditEventRecord(RepositoryModel):
    """
    Audit metadata for assessment and review operations.

    Do not put protocol text, patient identifiers, credentials, prompts,
    model responses, or other sensitive content in details.
    """

    event_id: str = Field(..., min_length=1, max_length=100)
    assessment_id: Optional[str] = Field(default=None, max_length=100)

    actor_id: str = Field(..., min_length=1, max_length=200)

    action: str = Field(..., min_length=1, max_length=100)
    outcome: Literal["success", "denied", "failure"]

    correlation_id: Optional[str] = Field(default=None, max_length=100)

    # Allow only operational metadata, not document or patient content.
    details: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )

    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("occurred_at")
    @classmethod
    def normalize_event_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must include a timezone.")
        return value.astimezone(timezone.utc)


class AssessmentQuery(RepositoryModel):
    """Filters and pagination for assessment searches."""

    study_id: Optional[str] = Field(default=None, max_length=100)

    assessment_status: Optional[
        Literal["pending", "running", "completed", "partial", "failed"]
    ] = None

    feasibility_status: Optional[
        Literal[
            "potentially_feasible",
            "feasible_with_risks",
            "potentially_not_feasible",
            "insufficient_data",
            "pending_human_review",
        ]
    ] = None

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class AssessmentPage(RepositoryModel):
    """Paginated assessment search result."""

    items: list[AssessmentRecord] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ReviewPage(RepositoryModel):
    """Paginated human review history."""

    items: list[ReviewRecord] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class AuditEventPage(RepositoryModel):
    """Paginated audit-event query result."""

    items: list[AuditEventRecord] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
