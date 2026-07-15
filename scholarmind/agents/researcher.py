"""
Researcher agent — hits arXiv and Semantic Scholar for every current search
term, merges results into the existing paper list, and de-duplicates. This
node is re-entered by the Critic's conditional edge when evidence is weak,
using a refined set of search_terms the Critic proposed.
"""
from ..state import dedupe_papers
from ..tools.arxiv_tool import search_arxiv
from ..tools.semantic_scholar_tool import search_semantic_scholar


def researcher_node(state: dict) -> dict:
    existing = state.get("papers", [])
    fetched: list[dict] = []

    for term in state["search_terms"]:
        fetched.extend(search_arxiv(term, max_results=2))
        #fetched.extend(search_semantic_scholar(term, max_results=5))

    merged = dedupe_papers(existing + fetched)
    return {"papers": merged}
