"""
Shared typed state for the Multi-Agent Clinical Trial Feasibility Assessment.

Field names match the uploaded specialist agents:
- patient_population_agent -> population_result
- site_feasibility_agent -> site_results, site_assessment_summary
- recruitment_risk_agent -> recruitment_result
- cost_timeline_agent -> timeline_cost_result
- critic_agent -> critic_result

Protocol and evidence outputs remain optional because corresponding agent
implementations were not included among the supplied files.
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class EvidenceReference(TypedDict, total=False):
    evidence_id: str
    document_id: str
    source: str
    section: str
    page: int
    page_number: int
    chunk_id: str
    excerpt: str
    relevance_score: float


class FeasibilityFinding(TypedDict, total=False):
    category: str
    severity: Literal["low", "medium", "high", "critical", "unknown"]
    finding: str
    rationale: str
    evidence_refs: list[Any]
    assumptions: list[str]
    recommended_action: str
    requires_human_review: bool


class ProtocolResult(TypedDict, total=False):
    summary: str
    eligibility_findings: list[FeasibilityFinding]
    design_complexity: str
    missing_information: list[str]
    evidence_refs: list[Any]


class PopulationResult(TypedDict, total=False):
    study_id: str
    target_enrollment: int | None
    geographic_population: float | None
    prevalence_rate: float | None
    disease_population_estimate: float | None
    eligible_population_estimate: float | None
    recruitable_population_estimate: float | None
    population_status: str
    findings: list[FeasibilityFinding]
    evidence_refs: list[Any]
    assumptions: list[str]
    missing_information: list[str]
    limitations: list[str]
    requires_human_review: bool


class SiteResult(TypedDict, total=False):
    site_id: str
    site_name: str
    score: float | None
    score_coverage_percent: float
    suitability: str
    capacity_per_month: float | None
    missing_criteria: list[str]
    invalid_criteria: list[str]
    missing_information: list[str]
    findings: list[FeasibilityFinding]
    evidence_refs: list[Any]
    assumptions: list[str]
    score_weights: dict[str, float]
    requires_human_review: bool


class SiteAssessmentSummary(TypedDict, total=False):
    study_id: str
    sites_evaluated: int
    sites_with_supported_scores: int
    sites_requiring_review: int
    missing_information: list[str]
    limitations: list[str]
    requires_human_review: bool


class RecruitmentResult(TypedDict, total=False):
    study_id: str
    target_enrollment: int
    planned_months: int
    required_monthly_enrollment: float
    available_monthly_capacity: float | None
    capacity_gap_per_month: float | None
    expected_capacity_over_period: float | None
    capacity_status: str
    population_status: str
    findings: list[FeasibilityFinding]
    assumptions: list[str]
    missing_information: list[str]
    limitations: list[str]
    requires_human_review: bool


class TimelineCostResult(TypedDict, total=False):
    study_id: str
    target_enrollment: int
    planned_months: int
    required_monthly_enrollment: float
    available_monthly_capacity: float | None
    estimated_enrollment_months: int | None
    timeline_status: str
    estimated_cost: float | None
    cost_breakdown: dict[str, Any] | None
    currency: str | None
    findings: list[FeasibilityFinding]
    assumptions: list[str]
    missing_information: list[str]
    limitations: list[str]
    requires_human_review: bool


class CriticResult(TypedDict, total=False):
    issues_found: bool
    contradictions: list[FeasibilityFinding]
    unsupported_claims: list[str]
    missing_information: list[str]
    review_required: bool
    summary: str
    checks_performed: list[str]
    limitations: list[str]


class FeasibilityState(TypedDict, total=False):
    """LangGraph state. Nodes return partial updates which LangGraph merges."""

    request: Any
    assessment_id: str
    study_id: str
    orchestration_engine: str
    workflow_status: str
    workflow_error: str | None
    execution_plan: list[str]
    reanalysis_attempts: int

    # Optional outputs from protocol/evidence modules not supplied here.
    protocol_result: ProtocolResult
    evidence: list[EvidenceReference]

    # Outputs from the attached agents.
    population_result: PopulationResult
    site_results: list[SiteResult]
    site_assessment_summary: SiteAssessmentSummary
    recruitment_result: RecruitmentResult
    timeline_cost_result: TimelineCostResult
    critic_result: CriticResult

    findings: list[FeasibilityFinding]
    contradictions: list[FeasibilityFinding]
    missing_information: list[str]
    limitations: list[str]

    review_status: Literal[
        "not_started",
        "pending_human_review",
        "approved",
        "rejected",
        "needs_revision",
    ]
    final_report: dict[str, Any]
