"""
Shared state schema for the ScholarMind LangGraph graph, plus the Pydantic
models used for structured LLM outputs (Planner's search plan, Critic's
stance score).

Design note: `papers` is NOT an auto-accumulating (Annotated/operator.add)
field. Each node that touches it receives the full current list and returns
the full updated, deduped list. This is slightly more verbose than relying on
a reducer, but it makes the dedup logic explicit and easy to reason about —
useful both for debugging and for explaining the design in an interview.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SearchPlan(BaseModel):
    """Structured output produced by the Planner agent."""

    sub_questions: list[str] = Field(
        description=(
            "3-5 specific, literature-searchable sub-questions that together "
            "would let you verify or refute the claim."
        )
    )
    search_terms: list[str] = Field(
        description=(
            "Concise keyword/short-phrase queries (not full sentences) to run "
            "against arXiv and Semantic Scholar."
        )
    )


class CriticScore(BaseModel):
    """Structured output produced by the Critic agent for a single paper."""

    stance: Literal[-1, 0, 1] = Field(
        description=(
            "-1 if the abstract's findings contradict the claim, 0 if neutral "
            "or not actually relevant, +1 if the findings support the claim."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this stance score.")
    rationale: str = Field(description="One-sentence justification for the score.")


def dedupe_papers(papers: list[dict]) -> list[dict]:
    """De-duplicate by paper_id, falling back to a normalised title."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in papers:
        key = p.get("paper_id") or (p.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


class ScholarMindState(TypedDict, total=False):
    # --- input ---
    topic: str
    claim: str

    # --- planner output ---
    sub_questions: list[str]
    search_terms: list[str]

    # --- researcher output (accumulated + deduped across refine loops) ---
    papers: list[dict]

    # --- critic output / loop control ---
    refine_count: int
    max_refines: int
    min_strong_evidence: int
    strong_evidence_count: int
    needs_refine: bool

    # --- human-in-the-loop output ---
    approved_papers: list[dict]

    # --- synthesiser output ---
    report: Optional[dict]
