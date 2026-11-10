
"""
Pydantic request schemas for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Compatible with Pydantic v2.
"""

from __future__ import annotations

import math
from typing import Any, Optional,List
from typing import Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class StrictInputModel(BaseModel):
    """Base model that rejects unexpected request fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

class SiteFeasibilityInput(StrictInputModel):
    """Site capacity and feasibility scoring inputs."""

    site_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique site identifier.",
    )
    site_name: Optional[str] = Field(
        default=None,
        max_length=200,
    )

    # Expected enrollment capacity per month for this site.
    capacity_per_month: Optional[float] = Field(
        default=None,
        ge=0,
        description="Expected participant enrollment per month.",
    )

    # Optional 0-5 feasibility scoring criteria.
    patient_access_score: Optional[float] = Field(
        default=None, ge=0, le=5,
    )
    trial_experience_score: Optional[float] = Field(
        default=None, ge=0, le=5,
    )
    staffing_score: Optional[float] = Field(
        default=None, ge=0, le=5,
    )
    facility_score: Optional[float] = Field(
        default=None, ge=0, le=5,
    )
    regulatory_readiness_score: Optional[float] = Field(
        default=None, ge=0, le=5,
    )

    evidence_refs: list[str] = Field(
        default_factory=list,
        description="References supporting site assumptions.",
    )

    @field_validator("capacity_per_month", mode="before")
    @classmethod
    def validate_capacity(cls, value: Any) -> Any:
        if value is not None:
            if isinstance(value, bool):
                raise ValueError("capacity_per_month must be numeric.")
            if not isinstance(value, (int, float)):
                raise ValueError("capacity_per_month must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError("capacity_per_month must be finite.")
        return value

    @field_validator(
        "patient_access_score",
        "trial_experience_score",
        "staffing_score",
        "facility_score",
        "regulatory_readiness_score",
        mode="before",
    )
    @classmethod
    def validate_scores(cls, value: Any) -> Any:
        if value is not None:
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Site scores must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError("Site scores must be finite.")
        return value


class PopulationAssumptions(StrictInputModel):
    """Optional population and recruitability assumptions."""

    geographic_population: Optional[int] = Field(
        default=None, ge=0,
    )
    prevalence_rate: Optional[float] = Field(
        default=None, ge=0, le=1,
    )
    eligible_population_estimate: Optional[int] = Field(
        default=None, ge=0,
    )
    recruitable_population_estimate: Optional[int] = Field(
        default=None, ge=0,
    )
    recruitment_fraction: Optional[float] = Field(
        default=None, ge=0, le=1,
    )
    evidence_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_source: Optional[str] = Field(default=None, max_length=300)

    @field_validator(
        "geographic_population",
        "eligible_population_estimate",
        "recruitable_population_estimate",
        mode="before",
    )
    @classmethod
    def validate_population_counts(cls, value: Any) -> Any:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Population counts must be integers.")
        return value

    @field_validator(
        "prevalence_rate",
        "recruitment_fraction",
        mode="before",
    )
    @classmethod
    def validate_population_rates(cls, value: Any) -> Any:
        if value is not None:
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Population rates must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError("Population rates must be finite.")
        return value


class CostAssumptions(StrictInputModel):
    """Unit-cost assumptions for indicative budget forecasting."""

    site_startup_cost: Optional[float] = Field(
        default=None, ge=0,
    )
    cost_per_participant: Optional[float] = Field(
        default=None, ge=0,
    )
    monthly_operating_cost: Optional[float] = Field(
        default=None, ge=0,
    )
    contingency_rate: Optional[float] = Field(
        default=0.10, ge=0, le=1,
    )
    currency: str = Field(default="USD", min_length=3, max_length=3)

    @field_validator(
        "site_startup_cost",
        "cost_per_participant",
        "monthly_operating_cost",
        "contingency_rate",
        mode="before",
    )
    @classmethod
    def validate_cost_numbers(cls, value: Any) -> Any:
        if value is not None:
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Cost assumptions must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError("Cost assumptions must be finite.")
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        normalized = value.upper()
        if not normalized.isalpha():
            raise ValueError("currency must contain three letters.")
        return normalized


class FeasibilityRequest(StrictInputModel):
    """Validated input contract for a feasibility assessment."""

    study_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique clinical study identifier.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Clinical trial title.",
    )
    therapeutic_area: str = Field(
        ...,
        min_length=1,
        max_length=150,
    )

    protocol_text: Optional[str] = Field(
        default=None,
        max_length=200_000,
        description=(
            "Protocol text for a POC. Prefer document IDs or secure "
            "document storage for larger or sensitive documents."
        ),
    )
    protocol_document_id: Optional[str] = Field(
        default=None,
        max_length=200,
    )

    target_enrollment: int = Field(
        ...,
        gt=0,
        description="Target number of participants.",
    )
    planned_months: int = Field(
        ...,
        gt=0,
        le=240,
        description="Planned recruitment or enrollment duration in months.",
    )

    enrolled_to_date: int = Field(default=0, ge=0)
    elapsed_months: int = Field(default=0, ge=0)

    sites: list[SiteFeasibilityInput] = Field(
        default_factory=list,
        max_length=500,
    )

    population_data: Optional[PopulationAssumptions] = None

    historical_monthly_enrollment: Optional[list[float]] = Field(
        default=None,
        max_length=120,
        description=(
            "Observed enrollment counts per month, ordered oldest to newest."
        ),
    )

    capacity_utilization: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Planning utilization assumption for reported capacity.",
    )

    cost_data: Optional[CostAssumptions] = None

    competing_trials_count: Optional[int] = Field(
        default=None,
        ge=0,
    )

    evidence_refs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    @field_validator(
        "target_enrollment",
        "planned_months",
        "enrolled_to_date",
        "elapsed_months",
        "competing_trials_count",
        mode="before",
    )
    @classmethod
    def validate_integer_fields(cls, value: Any) -> Any:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("This field must be an integer.")
        return value

    @field_validator("capacity_utilization", mode="before")
    @classmethod
    def validate_capacity_utilization(cls, value: Any) -> Any:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("capacity_utilization must be finite numeric.")
        return value

    @field_validator("historical_monthly_enrollment", mode="before")
    @classmethod
    def validate_historical_enrollment(cls, value: Any) -> Any:
        if value is None:
            return value

        if not isinstance(value, list):
            raise ValueError(
                "historical_monthly_enrollment must be a list."
            )

        for index, count in enumerate(value):
            if (
                isinstance(count, bool)
                or not isinstance(count, (int, float))
                or not math.isfinite(float(count))
                or count < 0
            ):
                raise ValueError(
                    f"Historical enrollment at index {index} "
                    "must be finite and nonnegative."
                )

        return value

    @model_validator(mode="after")
    def validate_timeline(self) -> "FeasibilityRequest":
        if self.enrolled_to_date > self.target_enrollment:
            raise ValueError(
                "enrolled_to_date cannot exceed target_enrollment."
            )

        if self.elapsed_months > self.planned_months:
            raise ValueError(
                "elapsed_months cannot exceed planned_months."
            )

        site_ids = [site.site_id for site in self.sites]
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("Each site_id must be unique.")

        return self


class ReviewRequest(BaseModel):
    """Human review decision for a feasibility assessment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    reviewer: str = Field(..., min_length=1, max_length=150)

    decision: Literal[
        "approved",
        "rejected",
        "needs_revision",
    ]

    comments: str | None = Field(default=None, max_length=2000)

    @field_validator("reviewer")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        if not value:
            raise ValueError("Reviewer cannot be empty")
        return value