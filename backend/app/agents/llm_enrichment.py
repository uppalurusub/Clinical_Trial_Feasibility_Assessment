"""Advisory LLM enrichment shared by deterministic feasibility agents.

The LLM is not used for arithmetic, scoring, capacity aggregation, cost math,
or approval decisions. Those remain deterministic and are preserved as returned
by each specialist. If the model/configuration fails, the workflow still returns
deterministic results and records the LLM enrichment as unavailable.
"""
import json
import logging
from typing import Any

from app.schemas.agent_outputs import AgentLLMInsight

from app.config.settings import settings
logger = settings.get_logger()

_REQUEST_FIELDS = (
    "title", "therapeutic_area", "target_enrollment", "planned_months",
    "enrolled_to_date", "elapsed_months", "competing_trials_count",
    "capacity_utilization", "population_data", "cost_data",
    "historical_monthly_enrollment",
)
_SITE_FIELDS = (
    "capacity_per_month", "patient_access_score", "trial_experience_score",
    "staffing_score", "facility_score", "regulatory_readiness_score",
)

def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump().get(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)

def _safe_context(state: dict) -> dict:
    request = state.get("request")
    context = {key: _get(request, key) for key in _REQUEST_FIELDS}
    sites = _get(request, "sites", []) or []
    if isinstance(sites, list):
        context["sites"] = [
            {field: _get(site, field) for field in _SITE_FIELDS}
            for site in sites
        ]
    else:
        context["sites"] = []
    # Do not send raw protocol text, document contents, reviewer identity,
    # patient identifiers, or evidence references to the LLM.
    return context

def _safe_result(value: Any) -> Any:
    """Redact identifiers/evidence and keep the LLM context bounded."""
    if isinstance(value, dict):
        blocked = {
            "study_id", "site_id", "site_name", "protocol_text",
            "protocol_document_id", "evidence_refs", "reviewer", "comments",
        }
        return {
            str(k): _safe_result(v)
            for k, v in value.items()
            if k not in blocked
        }
    if isinstance(value, list):
        return [_safe_result(item) for item in value[:100]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:1500] if isinstance(value, str) else value
    return str(value)[:500]

def enrich_result(
    *,
    state: dict,
    agent_name: str,
    output: dict,
    output_key: str,
) -> dict:
    """Add advisory insights through the shared LangSmith-traced OpenRouter helper.

    Deterministic outputs remain unchanged. The shared invoke_openrouter helper
    owns the model invocation and LangSmith token/cost/tag instrumentation.
    If the response is not valid schema JSON or the call fails, the original
    deterministic result is returned with a fallback insight.
    """
    payload = output.get(output_key)
    if not isinstance(payload, dict):
        logger.warning(
            "LLM enrichment skipped; result payload missing | agent=%s key=%s",
            agent_name, output_key,
        )
        return output

    try:
        # Lazy import keeps deterministic agents importable if tracing/LLM
        # dependencies are not installed in a test or offline environment.
        from app.observability.langsmith_usage import invoke_openrouter

        context = {
            "request_context": _safe_context(state),
            "deterministic_agent_result": _safe_result(payload),
        }
        serialized = json.dumps(context, ensure_ascii=False, default=str)[:14000]

        messages = [
            (
                "system",
                "You provide advisory narrative for a clinical-trial feasibility "
                "proof of concept. Use only facts present in the JSON. Do not invent "
                "numbers, evidence, source references, patient facts, or clinical claims. "
                "Do not recalculate arithmetic; do not change deterministic findings, "
                "scores, costs, timelines, status fields, or human-review requirements. "
                "If data is missing, say so. Keep recommendations as validation checks "
                "for qualified clinical operations reviewers, not medical advice. "
                "Return ONLY a valid JSON object with exactly these fields: "
                "summary (string), qualitative_risks (array of strings), "
                "recommended_checks (array of strings), uncertainty_notes (array of strings)."
            ),
            (
                "human",
                f"Agent: {agent_name}\\nAnalyze this supplied context only:\\n{serialized}",
            ),
        ]

        response = invoke_openrouter(
            messages,
            operation=agent_name,
            tags=[
                "workflow:clinical_trial_feasibility",
                f"agent:{agent_name}",
                "feature:advisory_enrichment",
            ],
        )
        answer = response.content
        if isinstance(answer, list):
            # Some providers return text as content blocks.
            answer = "".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("LLM returned empty structured insight content.")

        # Handle common JSON code-fence formatting without accepting arbitrary text.
        cleaned = answer.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        insight = AgentLLMInsight.model_validate_json(cleaned)
        payload["llm_insights"] = {
            **insight.model_dump(),
            "source": "openrouter_structured_output",
            "advisory_only": True,
        }
        logger.info(
            "LLM enrichment completed | agent=%s | langsmith_traced=true",
            agent_name,
        )
    except Exception as exc:
        logger.warning(
            "LLM enrichment unavailable; deterministic result preserved | "
            "agent=%s error_type=%s",
            agent_name, type(exc).__name__,
        )
        payload["llm_insights"] = {
            "summary": "LLM narrative enrichment was unavailable; use deterministic findings.",
            "qualitative_risks": [],
            "recommended_checks": [],
            "uncertainty_notes": [
                "LLM enrichment failed or returned invalid structured output; "
                "deterministic checks remain available."
            ],
            "source": "fallback",
            "advisory_only": True,
        }

    return output