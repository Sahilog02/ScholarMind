"""
Offline unit tests — no network or API keys required. Run with:
    pytest tests/
or just:
    python tests/test_tools.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scholarmind.state import dedupe_papers  # noqa: E402


def test_dedupe_by_paper_id():
    papers = [
        {"paper_id": "a1", "title": "Paper A"},
        {"paper_id": "a1", "title": "Paper A duplicate"},
        {"paper_id": "b2", "title": "Paper B"},
    ]
    result = dedupe_papers(papers)
    assert len(result) == 2
    assert {p["paper_id"] for p in result} == {"a1", "b2"}


def test_dedupe_by_title_when_no_id():
    papers = [
        {"title": "Same Title"},
        {"title": "same title"},  # different case, should still dedupe
        {"title": "Different Title"},
    ]
    result = dedupe_papers(papers)
    assert len(result) == 2


def test_dedupe_keeps_order_of_first_occurrence():
    papers = [{"paper_id": "x"}, {"paper_id": "y"}, {"paper_id": "x"}]
    result = dedupe_papers(papers)
    assert [p["paper_id"] for p in result] == ["x", "y"]


if __name__ == "__main__":
    test_dedupe_by_paper_id()
    test_dedupe_by_title_when_no_id()
    test_dedupe_keeps_order_of_first_occurrence()
    print("All tests passed.")
