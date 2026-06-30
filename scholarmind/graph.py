r"""
Wires the full ScholarMind graph:

    START -> planner -> researcher -> critic --[refine]--> researcher (loop)
                                          \--[proceed]--> human_review -> synthesizer -> END

The human_review node calls LangGraph's interrupt() to pause execution and
hand the Critic's scored paper list to whatever is driving the graph (the
Streamlit app, or eval/run_eval.py for batch runs). Execution resumes when
the caller invokes the graph again with Command(resume={...}).

Requires a checkpointer for interrupt() to work — InMemorySaver is fine for
local dev / a single Streamlit session; swap for a SqliteSaver/PostgresSaver
if you need state to survive a process restart.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .agents.critic import critic_node, route_after_critic
from .agents.planner import planner_node
from .agents.researcher import researcher_node
from .agents.synthesizer import synthesizer_node
from .state import ScholarMindState


def human_review_node(state: dict) -> dict:
    """Human-in-the-loop checkpoint: pause and show the scored shortlist."""
    decision = interrupt(
        {
            "type": "review_papers",
            "papers": state["papers"],
        }
    )
    # `decision` is whatever the caller passes to Command(resume=...).
    # Expected shape: {"approved_paper_ids": ["1234", "5678", ...]}
    approved_ids = set(decision.get("approved_paper_ids", []))
    approved = [p for p in state["papers"] if p.get("paper_id") in approved_ids]
    return {"approved_papers": approved}


def build_graph():
    graph = StateGraph(ScholarMindState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"refine": "researcher", "proceed": "human_review"},
    )
    graph.add_edge("human_review", "synthesizer")
    graph.add_edge("synthesizer", END)

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
