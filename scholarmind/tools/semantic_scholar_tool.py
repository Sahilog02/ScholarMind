"""
Thin wrapper around the Semantic Scholar Academic Graph API.
Works without a key (shared, heavily-throttled pool) but a free key raises
your personal rate limit substantially — request one at
https://www.semanticscholar.org/product/api#api-key-form

Docs: https://api.semanticscholar.org/api-docs/
"""
import os
import time

import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,authors,year,url"


def search_semantic_scholar(query: str, max_results: int = 5) -> list[dict]:
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    params = {"query": query, "limit": max_results, "fields": FIELDS}

    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
    if resp.status_code == 429:
        # Unauthenticated pool is shared and gets throttled — back off once and retry.
        time.sleep(2)
        resp = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()

    data = resp.json().get("data", []) or []
    papers = []
    for item in data:
        papers.append(
            {
                "paper_id": item.get("paperId"),
                "title": (item.get("title") or "").strip(),
                "abstract": (item.get("abstract") or "").strip(),
                "authors": [a.get("name") for a in item.get("authors", []) if a.get("name")],
                "year": item.get("year"),
                "source": "semantic_scholar",
                "url": item.get("url"),
            }
        )
    return papers
