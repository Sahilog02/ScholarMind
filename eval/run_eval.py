"""
Batch-runs ScholarMind over eval/claims.json and reports RAGAS faithfulness,
plus a plain-English summary block you can copy straight back to Claude when
asking it to write CV bullets from your real results.

Usage:
    python eval/run_eval.py

Note on the human-in-the-loop step: a batch eval can't have a person click
through 20 approval screens, so this script auto-approves every paper the
Critic scored as non-neutral (stance != 0) and treats that as "approved".
That's a reasonable stand-in for offline evaluation; the interactive
Streamlit app is where a real human actually reviews the shortlist.
"""
import json
import os
import sys
import time
import uuid
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv  # noqa: E402
from langgraph.types import Command  # noqa: E402

from scholarmind.graph import build_graph  # noqa: E402

load_dotenv()


def run_claim(graph, topic: str, claim: str, min_evidence: int = 4) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "topic": topic,
        "claim": claim,
        "refine_count": 0,
        "max_refines": 2,
        "min_strong_evidence": min_evidence,
    }
    result = graph.invoke(initial_state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        approved_ids = [p["paper_id"] for p in payload["papers"] if p.get("stance") != 0]
        result = graph.invoke(Command(resume={"approved_paper_ids": approved_ids}), config=config)

    return result


def main():
    claims_path = os.path.join(os.path.dirname(__file__), "claims.json")
    with open(claims_path) as f:
        claims = json.load(f)

    graph = build_graph()
    rows: list[dict] = []
    verdicts: Counter = Counter()
    refine_counts: list[int] = []
    failures = 0
    start = time.time()

    for item in claims:
        print(f"[{item['id']}/{len(claims)}] {item['claim'][:70]}")
        try:
            result = run_claim(graph, item["topic"], item["claim"])
            report = result.get("report", {})
            contexts = [
                p.get("abstract", "") for p in result.get("approved_papers", []) if p.get("abstract")
            ]
            verdict = report.get("verdict", "unknown")
            verdicts[verdict] += 1
            refine_counts.append(result.get("refine_count", 0))
            rows.append(
                {
                    "user_input": item["claim"],
                    "response": report.get("overall_summary", ""),
                    "retrieved_contexts": contexts or ["No evidence retrieved."],
                    "verdict": verdict,
                    "expected_verdict": item.get("expected_verdict"),
                    "refine_count": result.get("refine_count", 0),
                    "papers_considered": len(result.get("papers", [])),
                }
            )
        except Exception as e:
            failures += 1
            print(f"  FAILED: {e}")

    elapsed = time.time() - start

    out_path = os.path.join(os.path.dirname(__file__), "eval_raw_results.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n--- Summary (copy this block back to Claude for CV bullets) ---")
    print(f"Claims attempted:       {len(claims)}")
    print(f"Completed successfully: {len(rows)}")
    print(f"Failed:                 {failures}")
    print(f"Verdict distribution:   {dict(verdicts)}")
    triggered = sum(1 for r in refine_counts if r > 0)
    print(f"Claims that triggered the Critic->Researcher refine loop: {triggered}/{len(refine_counts)}")
    print(f"Total wall-clock time:  {elapsed:.1f}s  ({elapsed / max(len(claims), 1):.1f}s/claim avg)")

    if rows:
        score_with_ragas(rows)


def score_with_ragas(rows: list[dict]):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness
    except ImportError:
        print(
            "\nragas/datasets not installed — skipping faithfulness scoring.\n"
            "pip install ragas datasets and re-run."
        )
        return

    ds = Dataset.from_list(
        [
            {
                "user_input": r["user_input"],
                "response": r["response"],
                "retrieved_contexts": r["retrieved_contexts"],
            }
            for r in rows
            if r["response"]
        ]
    )

    try:
        result = evaluate(ds, metrics=[Faithfulness()])
    except Exception as e:
        print(
            f"\nRAGAS evaluate() failed ({e}).\n"
            "RAGAS's expected column names have changed across versions "
            "(question/answer/contexts vs. user_input/response/retrieved_contexts). "
            "Run `pip show ragas`, check https://docs.ragas.io for your installed "
            "version, and adjust the dict keys above if needed — this is a "
            "self-contained fix, the rest of the pipeline doesn't depend on it."
        )
        return

    df = result.to_pandas()
    csv_path = os.path.join(os.path.dirname(__file__), "ragas_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMean RAGAS faithfulness: {df['faithfulness'].mean():.3f}")
    print(f"Per-claim scores saved to {csv_path}")


if __name__ == "__main__":
    main()
