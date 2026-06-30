"""
Synthesiser agent — receives the human-approved paper shortlist, retrieves
the most relevant abstracts per sub-question via ChromaDB (rather than
passing every raw abstract to the LLM), and writes a structured verdict.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..config import get_llm
from ..rag import build_collection, retrieve

SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the Synthesiser agent. Using ONLY the evidence provided "
            "below (retrieved via RAG over the approved papers' abstracts), "
            "write a structured verdict on the claim. 'supported' = the "
            "evidence clearly backs the claim, 'refuted' = the evidence "
            "clearly contradicts it, 'contested' = the evidence is mixed, "
            "thin, or inconclusive. For each sub-question, summarise what the "
            "evidence says in 1-2 sentences, citing paper titles inline.",
        ),
        ("human", "Claim: {claim}\n\nEvidence by sub-question:\n{evidence}"),
    ]
)


class SubQuestionEvidence(BaseModel):
    sub_question: str
    summary: str


class SynthesisReport(BaseModel):
    verdict: Literal["supported", "refuted", "contested"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: list[SubQuestionEvidence]
    overall_summary: str


def synthesizer_node(state: dict) -> dict:
    papers = state["approved_papers"]
    collection = build_collection(papers)
    llm = get_llm().with_structured_output(SynthesisReport)

    evidence_blocks = []
    for sq in state["sub_questions"]:
        hits = retrieve(collection, sq, k=4)
        lines = [f"Sub-question: {sq}"]
        for h in hits:
            meta = h["metadata"]
            lines.append(f"- ({meta['title']}, {meta.get('year', 'n/a')}): {h['text'][:400]}")
        evidence_blocks.append("\n".join(lines))

    report: SynthesisReport = (SYNTH_PROMPT | llm).invoke(
        {"claim": state["claim"], "evidence": "\n\n".join(evidence_blocks)}
    )

    citations = [
        {"title": p["title"], "year": p.get("year"), "url": p.get("url"), "source": p.get("source")}
        for p in papers
    ]
    report_dict = report.model_dump()
    report_dict["citations"] = citations
    return {"report": report_dict}
