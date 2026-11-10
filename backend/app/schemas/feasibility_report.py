
"""
Final feasibility report schemas for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Compatible with Pydantic v2.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


Severity = Literal["low", "medium", "high", "critical"]

FeasibilityStatus = Literal[
    "potentially_feasible",
    "feasible_with_risks",
    "potentially_not_feasible",
    "insufficient_data",
    "pending_human_review",
]

ReviewStatus = Literal[
    "pending_human_review",
    "approved",
    "rejected",
    "changes_requested",
]

AssessmentStatus = Literal[
    "completed",
    "partial",
    "failed",
]


class ReportModel(BaseModel):
    """Common configuration for report models."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ReportEvidenceReference(ReportModel):
    """Traceable source reference supporting a report statement."""

    evidence_id: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1, max_length=500)

    page_number: Optional[int] = Field(default=None, ge=1)
    section: Optional[str] = Field(default=None, max_length=300)
    chunk_id: Optional[str] = Field(default=None, max_length=200)

    relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Retrieval relevance, not a probability of truth.",
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def validate_relevance_score(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("relevance_score must be finite numeric.")
        return value


class ReportFinding(ReportModel):
    """Consolidated finding from one or more assessment agents."""

    finding_id: Optional[str] = Field(default=None, max_length=200)
    code: str = Field(..., min_length=1, max_length=100)
    category: Literal[
        "protocol",
        "population",
        "site",
        "recruitment",
        "timeline",
        "cost",
        "data_quality",
        "evidence",
        "compliance",
        "other",
    ] = "other"

    severity: Severity = "medium"
    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., min_length=1, max_length=4000)

    basis: Literal[
        "source_evidence",
        "deterministic_calculation",
        "assumption",
        "model_inference",
        "human_input",
    ] = "source_evidence"

    evidence_refs: list[ReportEvidenceReference] = Field(
        default_factory=list,
        max_length=100,
    )

    recommended_action: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    requires_human_review: bool = False


class SiteFeasibilitySummary(ReportModel):
    """Site-level feasibility and scoring information."""

    site_id: str = Field(..., min_length=1, max_length=100)
    site_name: Optional[str] = Field(default=None, max_length=200)

    score: Optional[float] = Field(default=None, ge=0, le=100)
    score_band: Optional[
        Literal["high", "moderate", "low", "insufficient_data"]
    ] = None

    data_coverage_percent: Optional[float] = Field(
        default=None, ge=0, le=100
    )
    capacity_per_month: Optional[float] = Field(default=None, ge=0)

    rank: Optional[int] = Field(default=None, ge=1)
    findings: list[ReportFinding] = Field(default_factory=list)
    evidence_refs: list[ReportEvidenceReference] = Field(
        default_factory=list
    )

    requires_human_review: bool = True

    @field_validator(
        "score",
        "data_coverage_percent",
        "capacity_per_month",
        mode="before",
    )
    @classmethod
    def validate_site_numbers(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Site values must be finite numbers.")
        return value


class EnrollmentForecastSummary(ReportModel):
    """Enrollment target, capacity, and forecast summary."""

    target_enrollment: int = Field(..., ge=1)
    enrolled_to_date: int = Field(default=0, ge=0)
    planned_months: int = Field(..., ge=1)
    elapsed_months: int = Field(default=0, ge=0)

    remaining_enrollment: int = Field(..., ge=0)
    remaining_months: int = Field(..., ge=0)

    required_monthly_rate_from_start: float = Field(..., ge=0)
    required_monthly_rate_from_now: Optional[float] = Field(
        default=None, ge=0
    )

    reported_monthly_capacity: Optional[float] = Field(
        default=None, ge=0
    )
    effective_monthly_capacity: Optional[float] = Field(
        default=None, ge=0
    )

    estimated_months_to_target: Optional[int] = Field(
        default=None, ge=0
    )
    projected_enrollment_by_deadline: Optional[float] = Field(
        default=None, ge=0
    )
    projected_enrollment_gap: Optional[float] = Field(
        default=None, ge=0
    )

    capacity_status: str = Field(
        default="insufficient_data",
        max_length=100,
    )

    historical_average_monthly_enrollment: Optional[float] = Field(
        default=None, ge=0
    )

    forecast_method: Optional[str] = Field(
        default=None, max_length=100
    )

    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_enrollment_consistency(
        self,
    ) -> "EnrollmentForecastSummary":
        if self.enrolled_to_date > self.target_enrollment:
            raise ValueError(
                "enrolled_to_date cannot exceed target_enrollment."
            )

        if self.elapsed_months > self.planned_months:
            raise ValueError(
                "elapsed_months cannot exceed planned_months."
            )

        expected_remaining = (
            self.target_enrollment - self.enrolled_to_date
        )
        if self.remaining_enrollment != expected_remaining:
            raise ValueError(
                "remaining_enrollment must equal target minus enrolled."
            )

        expected_months = self.planned_months - self.elapsed_months
        if self.remaining_months != expected_months:
            raise ValueError(
                "remaining_months must equal planned minus elapsed."
            )

        return self


class CostForecastSummary(ReportModel):
    """Indicative study cost breakdown."""

    currency: str = Field(default="USD", min_length=3, max_length=3)

    site_startup_total: Optional[float] = Field(default=None, ge=0)
    participant_cost_total: Optional[float] = Field(default=None, ge=0)
    operating_cost_total: Optional[float] = Field(default=None, ge=0)

    base_subtotal: Optional[float] = Field(default=None, ge=0)
    contingency_rate: Optional[float] = Field(
        default=None, ge=0, le=1
    )
    contingency_amount: Optional[float] = Field(default=None, ge=0)

    estimated_total_cost: Optional[float] = Field(default=None, ge=0)
    remaining_cost_estimate: Optional[float] = Field(
        default=None, ge=0
    )
    cost_per_target_participant: Optional[float] = Field(
        default=None, ge=0
    )

    estimate_status: Literal[
        "estimated",
        "partial",
        "not_available",
    ] = "not_available"

    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.upper()
        if not normalized.isalpha():
            raise ValueError("currency must contain three letters.")
        return normalized

    @field_validator(
        "site_startup_total",
        "participant_cost_total",
        "operating_cost_total",
        "base_subtotal",
        "contingency_rate",
        "contingency_amount",
        "estimated_total_cost",
        "remaining_cost_estimate",
        "cost_per_target_participant",
        mode="before",
    )
    @classmethod
    def validate_cost_numbers(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Cost values must be finite numbers.")
        return value


class CriticSummary(ReportModel):
    """Cross-agent validation summary."""

    consistency_checks_passed: int = Field(default=0, ge=0)
    consistency_checks_failed: int = Field(default=0, ge=0)

    contradictions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)

    review_required: bool = True


class HumanReviewSummary(ReportModel):
    """Human review state and reviewer comments."""

    status: ReviewStatus = "pending_human_review"
    reviewer: Optional[str] = Field(default=None, max_length=200)
    comments: Optional[str] = Field(default=None, max_length=4000)
    reviewed_at: Optional[str] = Field(default=None, max_length=100)


class FeasibilityReport(ReportModel):
    """Final report returned by the feasibility assessment workflow."""

    assessment_id: str = Field(..., min_length=1, max_length=100)
    study_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=300)
    therapeutic_area: str = Field(..., min_length=1, max_length=150)

    assessment_status: AssessmentStatus = "completed"

    feasibility_status: FeasibilityStatus = "insufficient_data"
    executive_summary: str = Field(
        ...,
        min_length=1,
        max_length=8000,
    )

    protocol_summary: Optional[str] = Field(
        default=None, max_length=8000
    )
    population_summary: Optional[str] = Field(
        default=None, max_length=8000
    )

    enrollment_forecast: Optional[EnrollmentForecastSummary] = None
    site_assessments: list[SiteFeasibilitySummary] = Field(
        default_factory=list,
        max_length=500,
    )
    cost_forecast: Optional[CostForecastSummary] = None

    findings: list[ReportFinding] = Field(
        default_factory=list,
        max_length=1000,
    )

    missing_information: list[str] = Field(
        default_factory=list,
        max_length=500,
    )
    assumptions: list[str] = Field(
        default_factory=list,
        max_length=500,
    )
    limitations: list[str] = Field(
        default_factory=list,
        max_length=500,
    )

    evidence_refs: list[ReportEvidenceReference] = Field(
        default_factory=list,
        max_length=1000,
    )

    critic_summary: Optional[CriticSummary] = None
    human_review: HumanReviewSummary = Field(
        default_factory=HumanReviewSummary
    )

    requires_human_review: bool = True

    generated_at: Optional[str] = Field(
        default=None,
        max_length=100,
        description="ISO 8601 timestamp supplied by the service layer.",
    )

    @model_validator(mode="after")
    def validate_report_review_state(self) -> "FeasibilityReport":
        # An assessment cannot be treated as approved just because
        # automated agents completed successfully.
        if self.human_review.status != "approved":
            self.requires_human_review = True

        if self.critic_summary and self.critic_summary.review_required:
            self.requires_human_review = True

        if self.feasibility_status == "pending_human_review":
            self.requires_human_review = True

        if self.assessment_status == "failed":
            self.requires_human_review = True

        return self
