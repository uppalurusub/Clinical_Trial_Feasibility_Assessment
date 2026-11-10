"""
LangGraph orchestration for Multi-Agent Clinical Trial Feasibility Assessment.

Uses the five attached specialist implementations:
- patient_population_agent.py
- site_feasibility_agent.py
- recruitment_risk_agent.py
- cost_timeline_agent.py
- critic_agent.py

Protocol and RAG/evidence agent implementations were not among the supplied
files, so this graph does not import or fabricate them. The critic will flag
the missing protocol result and require human review.
"""
from __future__ import annotations


from typing import Any

from app.agents.cost_timeline_agent import cost_timeline_agent
from app.agents.critic_agent import critic_agent
from app.agents.patient_population_agent import patient_population_agent
from app.agents.recruitment_risk_agent import recruitment_risk_agent
from app.agents.site_feasibility_agent import site_feasibility_agent
from app.orchestration.routing import route_after_critic
from app.orchestration.state import FeasibilityState
from app.orchestration.supervisor import (
    human_review_node,
    report_ready_node,
    supervisor_node,
)

from app.config.settings import settings
logger = settings.get_logger()


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _validate_initial_state(state: dict[str, Any]) -> None:
    if state.get("request") is None:
        raise ValueError("A feasibility request is required.")
    if not _get_value(state["request"], "study_id"):
        raise ValueError("request.study_id is required.")


def _merge_node_output(state: dict[str, Any], node_fn) -> None:
    """Execute a node and merge its partial state update."""
    output = node_fn(state)
    if not isinstance(output, dict):
        raise TypeError(f"{node_fn.__name__} must return a dictionary.")
    state.update(output)


def _run_sequential(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Run the same node sequence without requiring LangGraph."""
    state = dict(initial_state)
    nodes = (
        supervisor_node,
        patient_population_agent,
        site_feasibility_agent,
        recruitment_risk_agent,
        cost_timeline_agent,
        critic_agent,
    )
    for node_fn in nodes:
        _merge_node_output(state, node_fn)

    state["orchestration_engine"] = "sequential_fallback"
    destination = route_after_critic(state)
    if destination == "human_review":
        state.update(human_review_node(state))
    else:
        state.update(report_ready_node(state))
    return state


def build_graph():
    """Build and compile the LangGraph workflow."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is not installed. Install it with 'pip install langgraph'."
        ) from exc

    graph = StateGraph(FeasibilityState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("patient_population", patient_population_agent)
    graph.add_node("sites", site_feasibility_agent)
    graph.add_node("recruitment", recruitment_risk_agent)
    graph.add_node("timeline_cost", cost_timeline_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("human_review", human_review_node)
    graph.add_node("report_ready", report_ready_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "patient_population")
    graph.add_edge("patient_population", "sites")
    graph.add_edge("sites", "recruitment")
    graph.add_edge("recruitment", "timeline_cost")
    graph.add_edge("timeline_cost", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "human_review": "human_review",
            "report_ready": "report_ready",
        },
    )
    graph.add_edge("human_review", END)
    graph.add_edge("report_ready", END)
    return graph.compile()


def run_workflow(request: Any) -> dict[str, Any]:
    """Run the feasibility workflow with LangGraph or a dependency fallback."""
    initial_state: FeasibilityState = {"request": request}
    _validate_initial_state(initial_state)

    study_id = _get_value(request, "study_id", "unknown")
    try:
        application = build_graph()
        result = application.invoke(initial_state)
        result["orchestration_engine"] = "langgraph"
        logger.info(
            "Feasibility workflow completed | study_id=%s | engine=langgraph",
            study_id,
        )
        return result
    except RuntimeError as exc:
        # Only use fallback when LangGraph itself is unavailable.
        if "LangGraph is not installed" not in str(exc):
            raise
        logger.warning("LangGraph unavailable; using sequential fallback.")
        return _run_sequential(initial_state)
    except Exception:
        logger.exception("Feasibility workflow failed | study_id=%s", study_id)
        raise
