"""
Planner agent — takes the user's topic + claim and decomposes it into 3-5
searchable sub-questions plus a list of search terms, as a structured
(Pydantic) output. This is a single LLM call.
"""
from langchain_core.prompts import ChatPromptTemplate

from ..config import get_llm
from ..state import SearchPlan

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Planner agent in a literature-review system. Given a "
            "topic and a claim, decompose the claim into 3-5 specific "
            "sub-questions that a literature search could answer, and propose "
            "concise search terms (keywords or short phrases, NOT full "
            "sentences) to query academic search APIs with.",
        ),
        ("human", "Topic: {topic}\nClaim to verify: {claim}"),
    ]
)


def planner_node(state: dict) -> dict:
    llm = get_llm().with_structured_output(SearchPlan)
    plan: SearchPlan = (PLANNER_PROMPT | llm).invoke(
        {"topic": state["topic"], "claim": state["claim"]}
    )
    return {
        "sub_questions": plan.sub_questions,
        "search_terms": plan.search_terms,
        "papers": [],
        "refine_count": 0,
    }
