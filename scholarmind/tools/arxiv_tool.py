"""
Thin wrapper around the arXiv API. No auth required.
Docs: https://info.arxiv.org/help/api/index.html
"""
import arxiv


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()

    papers = []
    for result in client.results(search):
        papers.append(
            {
                "paper_id": result.entry_id.rsplit("/", 1)[-1],
                "title": (result.title or "").strip(),
                "abstract": (result.summary or "").strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "year": result.published.year if result.published else None,
                "source": "arxiv",
                "url": result.entry_id,
            }
        )
    return papers
