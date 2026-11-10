"""OpenRouter invocation with LangSmith tags and token/cost metadata.

Configure LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT,
OPENROUTER_API_KEY, and OPENROUTER_MODEL in the environment or .env.

Only operational summaries are stored as trace inputs/outputs. Avoid PHI and
confidential site/study content in traceable function arguments.
"""
from __future__ import annotations


import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import get_current_run_tree, traceable, tracing_context

load_dotenv()
from app.config.settings import settings
logger = settings.get_logger()


def _tracing_enabled() -> bool:
    flag = (
        os.getenv("LANGSMITH_TRACING")
        or os.getenv("LANGCHAIN_TRACING_V2")
        or ""
    ).strip().lower()
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return flag in {"true", "1", "yes", "on"} and bool(key)


def _project_name() -> str:
    return (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "Clinical-Trial-Feasibility-Assessment"
    )


def _safe_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    messages = inputs.get("messages", [])
    return {
        "message_count": len(messages) if isinstance(messages, (list, tuple)) else 0,
        "model_provider": "openrouter",
        "model_name": str(inputs.get("model_name") or os.getenv("OPENROUTER_MODEL") or os.getenv("model") or "unknown"),
        "operation": str(inputs.get("operation", "llm_call")),
    }


def _safe_outputs(output: Any) -> dict[str, Any]:
    usage = getattr(output, "usage_metadata", None) or {}
    content = getattr(output, "content", None)
    return {
        "output_present": bool(content),
        "usage_available": bool(usage),
    }


def _as_int(value: Any) -> int | None:
    try:
        return max(0, int(value)) if value is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def _extract_usage(message: Any) -> dict[str, Any]:
    response_metadata = getattr(message, "response_metadata", {}) or {}
    usage_metadata = getattr(message, "usage_metadata", {}) or {}
    raw_usage = (
        response_metadata.get("usage")
        or response_metadata.get("token_usage")
        or response_metadata.get("usage_metadata")
        or {}
    )
    if not isinstance(raw_usage, dict):
        raw_usage = {}

    input_tokens = _as_int(usage_metadata.get("input_tokens"))
    output_tokens = _as_int(usage_metadata.get("output_tokens"))
    total_tokens = _as_int(usage_metadata.get("total_tokens"))

    if input_tokens is None:
        input_tokens = _as_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens")))
    if output_tokens is None:
        output_tokens = _as_int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens")))
    if total_tokens is None:
        total_tokens = _as_int(raw_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    usage: dict[str, Any] = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens

    cost = raw_usage.get("cost")
    cost_details = raw_usage.get("cost_details") or {}
    if cost is None and isinstance(cost_details, dict):
        cost = cost_details.get("upstream_inference_cost")
    if cost is not None:
        try:
            usage["total_cost"] = float(cost)
        except (TypeError, ValueError):
            pass

    # Do not fabricate input/output cost splits when provider only reports total cost.
    if isinstance(cost_details, dict):
        for source_key, target_key in (
            ("upstream_inference_prompt_cost", "input_cost"),
            ("upstream_inference_completions_cost", "output_cost"),
        ):
            value = cost_details.get(source_key)
            if value is not None:
                try:
                    usage[target_key] = float(value)
                except (TypeError, ValueError):
                    pass
    return usage


@traceable(
    name="openrouter_chat_completion",
    run_type="llm",
    project_name=_project_name(),
    process_inputs=_safe_inputs,
    process_outputs=_safe_outputs,
)
def invoke_openrouter(
    messages: list[Any],
    *,
    operation: str = "clinical_trial_feasibility",
    tags: list[str] | None = None,
) -> Any:
    """Call OpenRouter; trace sanitized input/output and attach usage to this LLM run."""
    model_name = (
        os.getenv("OPENROUTER_MODEL")
        or os.getenv("model")
        or "meta-llama/llama-3.3-70b-instruct"
    )
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("openrouter_api_key")
    base_url = (
        os.getenv("OPENROUTER_BASE_URL")
        or os.getenv("base_url")
        or "https://openrouter.ai/api/v1"
    )
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or openrouter_api_key).")

    safe_tags = list(dict.fromkeys([
        "workflow:clinical_trial_feasibility",
        "provider:openrouter",
        f"operation:{operation}",
        *(tags or []),
    ]))
    metadata = {
        "ls_provider": "openrouter",
        "ls_model_name": model_name,
        "operation": operation,
        "project": _project_name(),
    }

    # Explicitly enable/disable nested tracing using current and legacy env names.
    with tracing_context(
        enabled=_tracing_enabled(),
        project_name=_project_name(),
        tags=safe_tags,
        metadata=metadata,
    ):
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            # Helps usage be included for integrations/providers that support it.
            stream_usage=True,
        )
        response = llm.invoke(messages)

    usage = _extract_usage(response)
    run_tree = get_current_run_tree()
    if run_tree is not None:
        # The traceable wrapper is declared run_type="llm", so normalized
        # usage is attached to the run that should display in Tokens/Cost.
        if usage:
            try:
                run_tree.set(usage_metadata=usage)
            except Exception:
                logger.exception("Could not attach usage metadata to LangSmith run")
        try:
            run_tree.add_tags(safe_tags)
            run_tree.add_metadata(metadata)
        except Exception:
            logger.debug("Could not add supplemental LangSmith metadata", exc_info=True)

    logger.info(
        "OpenRouter usage captured | operation=%s | model=%s | "
        "input_tokens=%s | output_tokens=%s | total_tokens=%s | total_cost_usd=%s",
        operation,
        model_name,
        usage.get("input_tokens", "unavailable"),
        usage.get("output_tokens", "unavailable"),
        usage.get("total_tokens", "unavailable"),
        usage.get("total_cost", "unavailable"),
    )
    if not usage:
        logger.warning(
            "OpenRouter response exposed no token usage metadata; response metadata keys=%s",
            list((getattr(response, "response_metadata", {}) or {}).keys()),
        )
    return response
