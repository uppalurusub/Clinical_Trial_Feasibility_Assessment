
"""
Standardized agent result schemas for the Multi-Agent Clinical Trial
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
)


Severity = Literal["low", "medium", "high", "critical"]

FindingCategory = Literal[
    "protocol",
    "population",
    "site",
    "recruitment",
    "timeline",
    "cost",
    "data_quality",
    "evidence",
    "compliance",
    "orchestration",
    "other",
]

AgentStatus = Literal[
    "success",
    "partial",
    "failed",
    "skipped",
]


class AgentResultBase(BaseModel):
    """Base validation and serialization configuration."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class EvidenceReference(AgentResultBase):
    """Reference to evidence supporting an agent finding."""

    evidence_id: str = Field(..., min_length=1, max_length=200)
    source: str = Field(..., min_length=1, max_length=500)

    page_number: Optional[int] = Field(default=None, ge=1)
    section: Optional[str] = Field(default=None, max_length=300)
    chunk_id: Optional[str] = Field(default=None, max_length=200)

    # Optional retrieval score; it is not a probability of truth.
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def validate_relevance_score(cls, value: Any) -> Any:
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "relevance_score must be a finite numeric value."
                )
        return value


class AgentFinding(AgentResultBase):
    """A finding, risk, gap, or recommendation from an agent."""

    finding_id: Optional[str] = Field(
        default=None,
        max_length=200,
    )
    code: str = Field(..., min_length=1, max_length=100)
    category: FindingCategory = "other"
    severity: Severity = "medium"

    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(..., min_length=1, max_length=4000)

    # Distinguish observed facts, calculations, assumptions, and hypotheses.
    basis: Literal[
        "source_evidence",
        "deterministic_calculation",
        "assumption",
        "model_inference",
        "human_input",
    ] = "source_evidence"

    evidence_refs: list[EvidenceReference] = Field(
        default_factory=list,
        max_length=100,
    )

    recommended_action: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    requires_human_review: bool = False


class MissingInformation(AgentResultBase):
    """An input or evidence gap that limits an assessment."""

    field_name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)

    severity: Severity = "medium"

    # Why this information matters to the assessment.
    impact: Optional[str] = Field(default=None, max_length=2000)

    suggested_source: Optional[str] = Field(
        default=None,
        max_length=500,
    )


class AgentResult(AgentResultBase):
    """
    Generic result contract shared by specialist agents.

    The payload field accommodates agent-specific structured results.
    Use dedicated Pydantic models for payload validation where possible.
    """

    agent_name: str = Field(..., min_length=1, max_length=100)
    status: AgentStatus = "success"

    summary: str = Field(default="", max_length=4000)

    findings: list[AgentFinding] = Field(
        default_factory=list,
        max_length=500,
    )

    missing_information: list[MissingInformation] = Field(
        default_factory=list,
        max_length=200,
    )

    evidence_refs: list[EvidenceReference] = Field(
        default_factory=list,
        max_length=500,
    )

    payload: dict[str, Any] = Field(default_factory=dict)

    requires_human_review: bool = False
    error_code: Optional[str] = Field(default=None, max_length=100)
    error_message: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("payload")
    @classmethod
    def validate_payload_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Keep payloads structured and bounded at the API boundary.
        if len(value) > 200:
            raise ValueError("payload cannot contain more than 200 keys.")
        return value

    @field_validator("error_message")
    @classmethod
    def avoid_blank_error_message(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is not None and not value.strip():
            return None
        return value

    @field_validator("error_code")
    @classmethod
    def avoid_blank_error_code(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is not None and not value.strip():
            return None
        return value


class ProtocolAgentResult(AgentResult):
    """Structured result for protocol and eligibility analysis."""

    agent_name: Literal["protocol_agent"] = "protocol_agent"

    protocol_id: Optional[str] = None
    therapeutic_area: Optional[str] = None

    target_enrollment: Optional[int] = Field(default=None, ge=1)
    eligibility_criteria_count: Optional[int] = Field(default=None, ge=0)

    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)

    @field_validator(
        "target_enrollment",
        "eligibility_criteria_count",
        mode="before",
    )
    @classmethod
    def validate_protocol_counts(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("Protocol counts must be integers.")
        return value


class PopulationAgentResult(AgentResult):
    """Structured result for patient population feasibility."""

    agent_name: Literal["patient_population_agent"] = (
        "patient_population_agent"
    )

    geographic_population: Optional[int] = Field(default=None, ge=0)
    eligible_population_estimate: Optional[int] = Field(
        default=None, ge=0
    )
    recruitable_population_estimate: Optional[int] = Field(
        default=None, ge=0
    )

    population_sufficient: Optional[bool] = None


class SiteAgentResult(AgentResult):
    """Structured result for site feasibility scoring."""

    agent_name: Literal["site_feasibility_agent"] = (
        "site_feasibility_agent"
    )

    site_id: str = Field(..., min_length=1, max_length=100)
    site_name: Optional[str] = Field(default=None, max_length=200)

    score: Optional[float] = Field(default=None, ge=0, le=100)
    data_coverage_percent: Optional[float] = Field(
        default=None, ge=0, le=100
    )
    score_band: Optional[
        Literal["high", "moderate", "low", "insufficient_data"]
    ] = None

    capacity_per_month: Optional[float] = Field(default=None, ge=0)

    @field_validator(
        "score",
        "data_coverage_percent",
        "capacity_per_month",
        mode="before",
    )
    @classmethod
    def validate_site_numeric_fields(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Site numeric fields must be finite numbers.")
        return value


class RecruitmentAgentResult(AgentResult):
    """Structured result for recruitment rate and capacity analysis."""

    agent_name: Literal["recruitment_risk_agent"] = (
        "recruitment_risk_agent"
    )

    target_enrollment: Optional[int] = Field(default=None, ge=1)
    enrolled_to_date: Optional[int] = Field(default=None, ge=0)

    required_monthly_rate: Optional[float] = Field(
        default=None, ge=0
    )
    available_monthly_capacity: Optional[float] = Field(
        default=None, ge=0
    )

    projected_enrollment_gap: Optional[float] = Field(default=None, ge=0)

    capacity_status: Optional[str] = Field(
        default=None, max_length=100
    )


class TimelineCostAgentResult(AgentResult):
    """Structured result for timeline and cost estimation."""

    agent_name: Literal["cost_timeline_agent"] = "cost_timeline_agent"

    planned_months: Optional[int] = Field(default=None, ge=1)
    estimated_months_to_target: Optional[int] = Field(
        default=None, ge=0
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    estimated_total_cost: Optional[float] = Field(default=None, ge=0)
    remaining_cost_estimate: Optional[float] = Field(default=None, ge=0)

    @field_validator(
        "estimated_total_cost",
        "remaining_cost_estimate",
        mode="before",
    )
    @classmethod
    def validate_cost_values(cls, value: Any) -> Any:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Cost values must be finite numbers.")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        normalized = value.upper()
        if not normalized.isalpha():
            raise ValueError("currency must contain three letters.")
        return normalized


class CriticAgentResult(AgentResult):
    """Structured output for cross-agent validation."""

    agent_name: Literal["critic_agent"] = "critic_agent"

    consistency_checks_passed: int = Field(default=0, ge=0)
    consistency_checks_failed: int = Field(default=0, ge=0)

    contradictions: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    review_required: bool = True
