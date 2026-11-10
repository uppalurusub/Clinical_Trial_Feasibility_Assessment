
"""
Enrollment forecasting utilities for the Multi-Agent Clinical Trial
Feasibility Assessment project.

Features:
- Required monthly enrollment calculation.
- Historical recruitment rate estimation.
- Forecast using a moving average.
- Forecast using simple exponential smoothing.
- Enrollment gap and timeline assessment.
- Scenario analysis using site recruitment capacity.
- Input validation and explicit forecast limitations.

All forecasts are planning estimates, not validated clinical-trial
predictions. Validate assumptions with clinical operations SMEs.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


DEFAULT_MOVING_AVERAGE_WINDOW = 3
DEFAULT_ALPHA = 0.30


class EnrollmentForecastError(ValueError):
    """Raised when enrollment forecast inputs are invalid."""


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_nonnegative_number(name: str, value: Any) -> float:
    if not _is_finite_number(value) or float(value) < 0:
        raise EnrollmentForecastError(
            f"{name} must be a finite, nonnegative number."
        )
    return float(value)


def _require_positive_integer(name: str, value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise EnrollmentForecastError(
            f"{name} must be a positive integer."
        )
    return value


def required_monthly_enrollment(
    target_enrollment: int,
    planned_months: int,
    enrolled_to_date: int = 0,
    elapsed_months: int = 0,
) -> dict[str, Any]:
    """
    Calculate the enrollment rate needed to meet the target.

    Returns both the original plan rate and the rate required from now.
    """
    target = _require_positive_integer("target_enrollment", target_enrollment)
    planned = _require_positive_integer("planned_months", planned_months)
    enrolled = _require_nonnegative_number("enrolled_to_date", enrolled_to_date)
    elapsed = _require_nonnegative_number("elapsed_months", elapsed_months)

    if not enrolled.is_integer():
        raise EnrollmentForecastError("enrolled_to_date must be an integer.")

    if not elapsed.is_integer():
        raise EnrollmentForecastError("elapsed_months must be an integer.")

    enrolled_int = int(enrolled)
    elapsed_int = int(elapsed)

    if enrolled_int > target:
        raise EnrollmentForecastError(
            "enrolled_to_date cannot exceed target_enrollment."
        )

    if elapsed_int > planned:
        raise EnrollmentForecastError(
            "elapsed_months cannot exceed planned_months."
        )

    remaining_months = planned - elapsed_int
    remaining_enrollment = target - enrolled_int

    required_from_start = target / planned

    if remaining_months > 0:
        required_from_now = remaining_enrollment / remaining_months
    elif remaining_enrollment == 0:
        required_from_now = 0.0
    else:
        required_from_now = None

    return {
        "target_enrollment": target,
        "planned_months": planned,
        "enrolled_to_date": enrolled_int,
        "elapsed_months": elapsed_int,
        "remaining_enrollment": remaining_enrollment,
        "remaining_months": remaining_months,
        "required_monthly_rate_from_start": round(required_from_start, 4),
        "required_monthly_rate_from_now": (
            round(required_from_now, 4)
            if required_from_now is not None
            else None
        ),
        "timeline_already_elapsed": elapsed_int == planned,
        "target_already_met": remaining_enrollment == 0,
    }


def estimate_historical_rate(
    monthly_enrollment: Sequence[int | float],
) -> dict[str, Any]:
    """Summarize observed monthly enrollment."""
    if isinstance(monthly_enrollment, (str, bytes)):
        raise EnrollmentForecastError(
            "monthly_enrollment must be a numeric sequence."
        )

    values = list(monthly_enrollment)

    if not values:
        raise EnrollmentForecastError(
            "At least one historical monthly value is required."
        )

    validated: list[float] = []

    for index, value in enumerate(values):
        number = _require_nonnegative_number(
            f"monthly_enrollment[{index}]",
            value,
        )
        validated.append(number)

    total = sum(validated)
    average = total / len(validated)

    return {
        "months_observed": len(validated),
        "total_enrolled": round(total, 4),
        "average_monthly_enrollment": round(average, 4),
        "minimum_monthly_enrollment": round(min(validated), 4),
        "maximum_monthly_enrollment": round(max(validated), 4),
        "zero_enrollment_months": sum(
            value == 0 for value in validated
        ),
        "monthly_values": validated,
    }


def moving_average_forecast(
    monthly_enrollment: Sequence[int | float],
    forecast_months: int,
    window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> dict[str, Any]:
    """
    Forecast monthly enrollment using the mean of the latest observations.

    The same rate is carried forward for each forecast month. This is a
    simple baseline and does not model seasonality or site activation.
    """
    history = estimate_historical_rate(monthly_enrollment)["monthly_values"]
    horizon = _require_positive_integer("forecast_months", forecast_months)
    window_size = _require_positive_integer("window", window)

    selected = history[-min(window_size, len(history)):]
    forecast_rate = sum(selected) / len(selected)

    monthly_forecast = [round(forecast_rate, 4) for _ in range(horizon)]

    return {
        "method": "moving_average",
        "window_requested": window_size,
        "observations_used": len(selected),
        "forecast_months": horizon,
        "forecast_monthly_rate": round(forecast_rate, 4),
        "monthly_forecast": monthly_forecast,
        "cumulative_forecast": round(sum(monthly_forecast), 4),
    }


def exponential_smoothing_forecast(
    monthly_enrollment: Sequence[int | float],
    forecast_months: int,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Apply simple exponential smoothing.

    The final smoothed level is used as the forecast for each future month.
    This method does not model trend or seasonality.
    """
    history = estimate_historical_rate(monthly_enrollment)["monthly_values"]
    horizon = _require_positive_integer("forecast_months", forecast_months)

    if (
        not _is_finite_number(alpha)
        or not 0.0 < float(alpha) <= 1.0
    ):
        raise EnrollmentForecastError("alpha must be in (0, 1].")

    smoothing = float(alpha)
    level = history[0]

    for observation in history[1:]:
        level = (
            smoothing * observation
            + (1.0 - smoothing) * level
        )

    monthly_forecast = [round(level, 4) for _ in range(horizon)]

    return {
        "method": "simple_exponential_smoothing",
        "alpha": smoothing,
        "observations_used": len(history),
        "forecast_months": horizon,
        "forecast_monthly_rate": round(level, 4),
        "monthly_forecast": monthly_forecast,
        "cumulative_forecast": round(sum(monthly_forecast), 4),
    }


def assess_enrollment_feasibility(
    target_enrollment: int,
    planned_months: int,
    monthly_site_capacities: Sequence[int | float],
    enrolled_to_date: int = 0,
    elapsed_months: int = 0,
    historical_monthly_enrollment: Optional[
        Sequence[int | float]
    ] = None,
    capacity_utilization: float = 1.0,
) -> dict[str, Any]:
    """
    Assess whether reported monthly site capacity may support the target.

    monthly_site_capacities should contain comparable monthly enrollment
    capacities for active sites. Do not include inactive sites unless their
    expected activation timing is explicitly modeled elsewhere.

    capacity_utilization is an explicit planning assumption from 0 to 1.
    """
    plan = required_monthly_enrollment(
        target_enrollment=target_enrollment,
        planned_months=planned_months,
        enrolled_to_date=enrolled_to_date,
        elapsed_months=elapsed_months,
    )

    if isinstance(monthly_site_capacities, (str, bytes)):
        raise EnrollmentForecastError(
            "monthly_site_capacities must be a numeric sequence."
        )

    capacities = [
        _require_nonnegative_number(
            f"monthly_site_capacities[{index}]",
            value,
        )
        for index, value in enumerate(monthly_site_capacities)
    ]

    if (
        not _is_finite_number(capacity_utilization)
        or not 0.0 <= float(capacity_utilization) <= 1.0
    ):
        raise EnrollmentForecastError(
            "capacity_utilization must be between 0 and 1."
        )

    raw_capacity = sum(capacities)
    effective_capacity = raw_capacity * float(capacity_utilization)
    required_rate = plan["required_monthly_rate_from_now"]

    if required_rate is None:
        capacity_status = (
            "target_met"
            if plan["target_already_met"]
            else "planned_timeline_elapsed"
        )
        estimated_months_to_target = (
            0 if plan["target_already_met"] else None
        )
    elif plan["remaining_enrollment"] == 0:
        capacity_status = "target_met"
        estimated_months_to_target = 0
    elif effective_capacity <= 0:
        capacity_status = "no_reported_capacity"
        estimated_months_to_target = None
    else:
        estimated_months_to_target = math.ceil(
            plan["remaining_enrollment"] / effective_capacity
        )

        if effective_capacity >= required_rate:
            capacity_status = "capacity_meets_required_rate"
        else:
            capacity_status = "potential_enrollment_gap"

    projected_enrollment_by_deadline = (
        plan["enrolled_to_date"]
        + effective_capacity * plan["remaining_months"]
    )

    projected_gap = max(
        0.0,
        plan["target_enrollment"] - projected_enrollment_by_deadline,
    )

    result: dict[str, Any] = {
        **plan,
        "site_count": len(capacities),
        "reported_monthly_site_capacity": round(raw_capacity, 4),
        "capacity_utilization_assumption": float(capacity_utilization),
        "effective_monthly_capacity": round(effective_capacity, 4),
        "capacity_status": capacity_status,
        "estimated_months_to_target_at_capacity": (
            estimated_months_to_target
        ),
        "projected_enrollment_by_deadline": round(
            projected_enrollment_by_deadline, 4
        ),
        "projected_enrollment_gap": round(projected_gap, 4),
        "historical_analysis": None,
        "forecast_comparison": None,
        "requires_human_review": True,
        "limitations": [
            "Site capacities are assumed to be achievable and additive.",
            "Site activation dates and ramp-up are not modeled.",
            "Screen failures, withdrawals, competing trials, and seasonality "
            "are not modeled unless reflected in the supplied assumptions.",
            "Capacity is not the same as a statistically validated forecast.",
            "Review assumptions with clinical operations before decisions.",
        ],
    }

    if historical_monthly_enrollment is not None:
        historical = estimate_historical_rate(
            historical_monthly_enrollment
        )
        result["historical_analysis"] = historical

        forecast_horizon = max(plan["remaining_months"], 1)

        moving = moving_average_forecast(
            historical["monthly_values"],
            forecast_months=forecast_horizon,
        )

        smoothing = exponential_smoothing_forecast(
            historical["monthly_values"],
            forecast_months=forecast_horizon,
        )

        result["forecast_comparison"] = {
            "moving_average": moving,
            "exponential_smoothing": smoothing,
        }

        if required_rate is not None:
            result["historical_rate_vs_required"] = {
                "historical_average_monthly_enrollment": historical[
                    "average_monthly_enrollment"
                ],
                "required_monthly_rate_from_now": required_rate,
                "historical_average_meets_required_rate": (
                    historical["average_monthly_enrollment"]
                    >= required_rate
                ),
            }

    return result
