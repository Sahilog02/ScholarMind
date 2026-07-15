"""
Critic agent + conditional edge — this is the piece that makes the graph a
state machine instead of a linear pipeline.

For every not-yet-scored paper, ask the LLM for a stance (-1/0/+1) and a
confidence. If too few "strong" papers (non-neutral, confidence >= 0.6) have
been found and we haven't hit max_refines yet, the Critic ALSO proposes new,
differently-angled search terms in the same node call (one extra LLM call,
only made when actually needed) and sets needs_refine=True. The conditional
edge function just reads that flag — all the actual decision logic lives in
one place (critic_node), so it can't drift out of sync with the routing.

max_refines bounds the loop so it can't run forever: once it's hit, the graph
proceeds to human review regardless of evidence strength, and the HITL step
makes that visible to the user (it's expected to see "contested" verdicts
when evidence stays thin even after refinement — that's correct behaviour,
not a bug).
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import get_llm
from ..state import CriticScore

CRITIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Critic agent. Read the abstract and decide whether "
            "this paper's findings support, contradict, or are neutral/"
            "irrelevant to the claim. -1 = contradicts the claim, 0 = neutral "
            "or not actually relevant, +1 = supports the claim. Be skeptical: "
            "only assign +1 or -1 when the abstract gives a clear directional "
            "result on the SPECIFIC claim, not just on the same general topic.",
        ),
        ("human", "Claim: {claim}\n\nPaper title: {title}\nAbstract: {abstract}"),
    ]
)


class RefinedSearch(BaseModel):
    search_terms: list[str] = Field(
        description=(
            "2-4 new search terms, more specific or from a different angle "
            "than the terms already tried, aimed at finding papers with a "
            "clearer directional result on the claim."
        )
    )


REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The literature search so far has not turned up enough strong "
            "evidence (clear support or contradiction) for the claim below. "
            "Propose new search terms that approach the claim from a "
            "different angle than what has already been tried.",
        ),
        (
            "human",
            "Claim: {claim}\nSub-questions: {sub_questions}\n"
            "Terms already tried: {tried_terms}\nPapers found so far: {papers_found}",
        ),
    ]
)


def critic_node(state: dict) -> dict:
    llm = get_llm().with_structured_output(CriticScore)

    scored = []
    for p in state["papers"]:
        if "stance" in p:  # already scored in a previous loop iteration
            scored.append(p)
            continue

        abstract = p.get("abstract") or ""
        if not abstract:
            scored.append({**p, "stance": 0, "confidence": 0.0, "rationale": "No abstract available."})
            continue

        result: CriticScore = (CRITIC_PROMPT | llm).invoke(
            {"claim": state["claim"], "title": p["title"], "abstract": abstract}
        )
        scored.append(
            {
                **p,
                "stance": result.stance,
                "confidence": result.confidence,
                "rationale": result.rationale,
            }
        )

    strong = [p for p in scored if p["stance"] != 0 and p.get("confidence", 0) >= 0.6]
    needs_refine = (
        len(strong) < state.get("min_strong_evidence", 2)
        and state.get("refine_count", 0) < state.get("max_refines", 1)
    )

    update: dict = {
        "papers": scored,
        "strong_evidence_count": len(strong),
        "needs_refine": needs_refine,
    }

    if needs_refine:
        refine_llm = get_llm().with_structured_output(RefinedSearch)
        refined: RefinedSearch = (REFINE_PROMPT | refine_llm).invoke(
            {
                "claim": state["claim"],
                "sub_questions": state["sub_questions"],
                "tried_terms": state.get("search_terms", []),
                "papers_found": len(scored),
            }
        )
        update["search_terms"] = refined.search_terms
        update["refine_count"] = state.get("refine_count", 0) + 1

    return update


def route_after_critic(state: dict) -> Literal["refine", "proceed"]:
    return "refine" if state.get("needs_refine") else "proceed"
