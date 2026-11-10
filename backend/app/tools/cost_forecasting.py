
"""
Cost forecasting utilities for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Features:
- Site startup cost estimation.
- Participant-based cost estimation.
- Monthly operating cost estimation.
- Contingency calculation.
- Baseline, low-cost, and high-cost scenarios.
- Explicit input validation and forecast limitations.

All amounts use the caller-supplied currency. This module does not
perform currency conversion or replace a sponsor-approved budget.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


DEFAULT_CONTINGENCY_RATE = 0.10


class CostForecastError(ValueError):
    """Raised when cost forecasting inputs are invalid."""


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative(name: str, value: Any) -> float:
    if not _is_finite_number(value) or float(value) < 0:
        raise CostForecastError(
            f"{name} must be a finite, nonnegative number."
        )
    return float(value)


def _positive_integer(name: str, value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise CostForecastError(
            f"{name} must be a positive integer."
        )
    return value


def _rate(name: str, value: Any) -> float:
    if (
        not _is_finite_number(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise CostForecastError(
            f"{name} must be between 0 and 1."
        )
    return float(value)


def forecast_study_cost(
    *,
    target_enrollment: int,
    site_count: int,
    planned_months: int,
    site_startup_cost: float,
    cost_per_participant: float,
    monthly_operating_cost: float,
    contingency_rate: float = DEFAULT_CONTINGENCY_RATE,
    currency: str = "USD",
    enrolled_to_date: int = 0,
    elapsed_months: int = 0,
    cost_per_additional_month: Optional[float] = None,
) -> dict[str, Any]:
    """
    Estimate total study costs using explicit unit-cost assumptions.

    Cost model:
        startup = site_count * site_startup_cost
        participant = target_enrollment * cost_per_participant
        operations = planned_months * monthly_operating_cost
        subtotal = startup + participant + operations
        contingency = subtotal * contingency_rate

    If enrolled_to_date and elapsed_months are supplied, the function
    also estimates remaining costs using remaining participants and
    remaining months. Past spend is not inferred from these inputs.

    cost_per_additional_month, when supplied, estimates the extra
    operating cost for each month beyond the original planned duration.
    """
    target = _positive_integer("target_enrollment", target_enrollment)
    sites = _positive_integer("site_count", site_count)
    months = _positive_integer("planned_months", planned_months)

    startup_unit = _nonnegative("site_startup_cost", site_startup_cost)
    participant_unit = _nonnegative(
        "cost_per_participant", cost_per_participant
    )
    monthly_unit = _nonnegative(
        "monthly_operating_cost", monthly_operating_cost
    )
    contingency = _rate("contingency_rate", contingency_rate)

    enrolled = _nonnegative("enrolled_to_date", enrolled_to_date)
    elapsed = _nonnegative("elapsed_months", elapsed_months)

    if not enrolled.is_integer():
        raise CostForecastError("enrolled_to_date must be an integer.")

    if not elapsed.is_integer():
        raise CostForecastError("elapsed_months must be an integer.")

    enrolled_int = int(enrolled)
    elapsed_int = int(elapsed)

    if enrolled_int > target:
        raise CostForecastError(
            "enrolled_to_date cannot exceed target_enrollment."
        )

    if elapsed_int > months:
        raise CostForecastError(
            "elapsed_months cannot exceed planned_months."
        )

    if not isinstance(currency, str) or not currency.strip():
        raise CostForecastError("currency must be a non-empty string.")

    currency = currency.strip().upper()

    if cost_per_additional_month is not None:
        extra_month_unit = _nonnegative(
            "cost_per_additional_month",
            cost_per_additional_month,
        )
    else:
        extra_month_unit = None

    startup_total = sites * startup_unit
    participant_total = target * participant_unit
    operating_total = months * monthly_unit

    base_subtotal = (
        startup_total + participant_total + operating_total
    )
    contingency_total = base_subtotal * contingency
    total_cost = base_subtotal + contingency_total

    remaining_participants = target - enrolled_int
    remaining_months = months - elapsed_int

    # This is a future-cost estimate, not an estimate of actual spend
    # already incurred.
    remaining_participant_cost = (
        remaining_participants * participant_unit
    )
    remaining_operating_cost = remaining_months * monthly_unit

    # Startup costs are excluded from the remaining estimate by default
    # because the current activation status of each site is unknown.
    remaining_subtotal = (
        remaining_participant_cost + remaining_operating_cost
    )
    remaining_contingency = remaining_subtotal * contingency
    remaining_estimate = remaining_subtotal + remaining_contingency

    return {
        "currency": currency,
        "cost_model_version": "study-cost-v1",
        "inputs": {
            "target_enrollment": target,
            "site_count": sites,
            "planned_months": months,
            "enrolled_to_date": enrolled_int,
            "elapsed_months": elapsed_int,
            "site_startup_cost_per_site": round(startup_unit, 2),
            "cost_per_participant": round(participant_unit, 2),
            "monthly_operating_cost": round(monthly_unit, 2),
            "contingency_rate": contingency,
        },
        "cost_breakdown": {
            "site_startup_total": round(startup_total, 2),
            "participant_cost_total": round(participant_total, 2),
            "operating_cost_total": round(operating_total, 2),
            "base_subtotal": round(base_subtotal, 2),
            "contingency_amount": round(contingency_total, 2),
            "total_estimated_cost": round(total_cost, 2),
        },
        "remaining_cost_estimate": {
            "remaining_participants": remaining_participants,
            "remaining_months": remaining_months,
            "remaining_participant_cost": round(
                remaining_participant_cost, 2
            ),
            "remaining_operating_cost": round(
                remaining_operating_cost, 2
            ),
            "remaining_subtotal": round(remaining_subtotal, 2),
            "remaining_contingency": round(
                remaining_contingency, 2
            ),
            "remaining_estimated_cost": round(
                remaining_estimate, 2
            ),
            "startup_costs_included": False,
            "note": (
                "Remaining estimate excludes startup costs because "
                "site activation and unpaid commitments are unknown."
            ),
        },
        "cost_per_target_participant": round(
            total_cost / target, 2
        ),
        "cost_per_site": round(total_cost / sites, 2),
        "additional_month_sensitivity": (
            {
                "cost_per_additional_month": round(
                    extra_month_unit, 2
                ),
                "contingency_adjusted_additional_month_cost": round(
                    extra_month_unit * (1.0 + contingency), 2
                ),
            }
            if extra_month_unit is not None
            else None
        ),
        "limitations": [
            "Unit costs are assumed to be complete and mutually exclusive.",
            "Taxes, currency conversion, inflation, and payment milestones "
            "are not modeled.",
            "Startup costs are applied to every supplied site.",
            "Operating expenses are assumed constant each month.",
            "Actual historical spend and outstanding commitments are not "
            "derived from enrollment and elapsed time.",
            "The contingency rate is a planning assumption.",
            "This estimate is not a sponsor-approved clinical trial budget.",
        ],
    }


def forecast_cost_scenarios(
    *,
    target_enrollment: int,
    site_count: int,
    planned_months: int,
    site_startup_cost: float,
    cost_per_participant: float,
    monthly_operating_cost: float,
    contingency_rate: float = DEFAULT_CONTINGENCY_RATE,
    currency: str = "USD",
    low_cost_factor: float = 0.90,
    high_cost_factor: float = 1.20,
) -> dict[str, Any]:
    """
    Compare low, baseline, and high unit-cost scenarios.

    Factors are applied to the three cost inputs. Contingency is applied
    separately to each resulting subtotal.
    """
    low_factor = _nonnegative("low_cost_factor", low_cost_factor)
    high_factor = _nonnegative("high_cost_factor", high_cost_factor)

    if low_factor > 1.0:
        raise CostForecastError(
            "low_cost_factor must be at most 1.0."
        )

    if high_factor < 1.0:
        raise CostForecastError(
            "high_cost_factor must be at least 1.0."
        )

    scenarios: dict[str, dict[str, Any]] = {}

    definitions = {
        "low": low_factor,
        "baseline": 1.0,
        "high": high_factor,
    }

    for name, factor in definitions.items():
        scenarios[name] = forecast_study_cost(
            target_enrollment=target_enrollment,
            site_count=site_count,
            planned_months=planned_months,
            site_startup_cost=site_startup_cost * factor,
            cost_per_participant=cost_per_participant * factor,
            monthly_operating_cost=monthly_operating_cost * factor,
            contingency_rate=contingency_rate,
            currency=currency,
        )

    return {
        "currency": currency.strip().upper(),
        "scenario_method": "unit_cost_sensitivity",
        "scenario_totals": {
            name: result["cost_breakdown"]["total_estimated_cost"]
            for name, result in scenarios.items()
        },
        "scenarios": scenarios,
        "limitations": [
            "Scenario factors are assumptions, not statistical confidence bounds.",
            "Low and high cases change unit costs but do not change enrollment, "
            "site count, or study duration.",
        ],
    }


def forecast_cost_with_overrun(
    baseline_cost: float,
    overrun_rate: float,
    currency: str = "USD",
) -> dict[str, Any]:
    """Calculate a simple cost-overrun sensitivity."""
    baseline = _nonnegative("baseline_cost", baseline_cost)
    rate = _nonnegative("overrun_rate", overrun_rate)

    if not isinstance(currency, str) or not currency.strip():
        raise CostForecastError("currency must be a non-empty string.")

    projected_overrun = baseline * rate

    return {
        "currency": currency.strip().upper(),
        "baseline_cost": round(baseline, 2),
        "overrun_rate": rate,
        "projected_overrun_amount": round(projected_overrun, 2),
        "projected_total_cost": round(baseline + projected_overrun, 2),
        "limitations": [
            "This is a sensitivity calculation, not a probability-based forecast."
        ],
    }
