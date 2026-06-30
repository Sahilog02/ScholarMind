# ScholarMind

A multi-agent literature-review system. Give it a topic and a claim — *"do
transformers outperform RNNs on long sequences?"* — and four agents work
together to verify it: a **Planner** breaks the claim into sub-questions, a
**Researcher** pulls real papers from arXiv and Semantic Scholar, a **Critic**
scores each paper's stance and sends the Researcher back for another pass if
the evidence is weak, and a **Synthesiser** writes a structured, cited verdict
once a human has reviewed the shortlist.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) on a real
cyclic graph (not a linear chain), with a human-in-the-loop checkpoint and a
RAGAS-evaluated, LangSmith-traced output.

## Why this isn't just a pipeline

Most "agent" demos are a DAG: step 1 → step 2 → step 3, done. The interesting
part of ScholarMind is the **Critic → Researcher conditional edge**: if the
Critic doesn't find enough strong (non-neutral, confident) evidence, it
proposes new, differently-angled search terms and the graph loops back to the
Researcher — up to a capped number of times, so it can't run forever. That's
a genuine state machine with a stopping condition, not a one-way pipe.

The second piece is the **human-in-the-loop checkpoint**: after the Critic
scores papers, a `LangGraph interrupt()` pauses execution and hands the
scored list to a person (via the Streamlit app) to prune before the
Synthesiser runs. Execution resumes exactly where it paused via
`Command(resume=...)`.

## The agent graph

```
START
  │
  ▼
Planner            — 1 LLM call, structured output (Pydantic SearchPlan):
  │                   topic + claim → 3-5 sub-questions + search terms
  ▼
Researcher  ◄───┐  — hits arXiv (no auth) + Semantic Scholar (free tier)
  │             │     for every search term, merges + dedupes papers
  ▼             │
Critic ─────────┘  — scores each new paper's stance (-1/0/+1) + confidence.
  │  [refine]        If strong-evidence papers < threshold AND refine budget
  │                  remains: proposes new search terms, loops back.
  │ [proceed]
  ▼
Human review        — interrupt() pauses the graph; Streamlit shows the
  │                    scored list; user unchecks irrelevant papers, approves
  ▼
Synthesiser         — RAGs over approved abstracts via ChromaDB (per
  │                    sub-question, not the whole abstract dump), writes a
  │                    structured verdict: supported / refuted / contested
  ▼
END
```

## Stack

LangGraph 1.2 · LangSmith (observability) · ChromaDB + sentence-transformers
(RAG) · RAGAS (faithfulness eval) · arXiv API · Semantic Scholar API ·
Pydantic (structured outputs) · Groq (`openai/gpt-oss-120b`, free tier) or
OpenAI · Streamlit

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd scholarmind
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
- Get a free Groq key at https://console.groq.com/keys (default provider —
  `GROQ_MODEL` defaults to `openai/gpt-oss-120b`; Groq deprecates model names
  fairly often, check https://console.groq.com/docs/models if it errors).
- Optionally get a free Semantic Scholar key at
  https://www.semanticscholar.org/product/api#api-key-form — works without
  one, but you'll hit the shared rate limit faster.
- Optionally get a free LangSmith key at https://smith.langchain.com and set
  `LANGSMITH_TRACING=true` to get full execution traces (this is what the
  spec means by "observability" — no code changes needed, just env vars).

Sanity-check the wiring before running anything that costs API calls:

```bash
python3 -c "from scholarmind.graph import build_graph; build_graph(); print('graph compiles OK')"
```

## Running it

**Interactive app** (the actual human-in-the-loop experience):
```bash
streamlit run app.py
```

**Batch evaluation** (runs all 20 claims in `eval/claims.json`, auto-approves
at the HITL step since there's no person to click through 20 screens, scores
RAGAS faithfulness):
```bash
python eval/run_eval.py
```
This writes `eval/eval_raw_results.json` (full output per claim) and
`eval/ragas_results.csv` (per-claim faithfulness scores), and prints a
summary block — claims completed/failed, verdict distribution, how many
claims actually triggered the refine loop, mean faithfulness, total runtime.
**Copy that summary block** if you want help turning it into CV bullets.

**Unit tests** (no network or API keys needed):
```bash
pytest tests/
```

**Optional — visual graph debugger** (LangGraph Studio):
```bash
pip install "langgraph-cli[inmem]"
langgraph dev
```

## Project structure

```
scholarmind/
├── app.py                          Streamlit UI (HITL pause/resume)
├── langgraph.json                  optional LangGraph Studio config
├── scholarmind/
│   ├── state.py                    shared graph state + Pydantic schemas
│   ├── config.py                   LLM provider switch (Groq/OpenAI)
│   ├── graph.py                    graph wiring, conditional edge, interrupt
│   ├── rag.py                      ChromaDB embed/retrieve helpers
│   ├── tools/
│   │   ├── arxiv_tool.py
│   │   └── semantic_scholar_tool.py
│   └── agents/
│       ├── planner.py
│       ├── researcher.py
│       ├── critic.py               the conditional-loop logic lives here
│       └── synthesizer.py
├── eval/
│   ├── claims.json                 20 test claims spanning ML/AI topics
│   └── run_eval.py                 batch run + RAGAS faithfulness scoring
└── tests/
    └── test_tools.py
```

## Design notes (useful if you get asked about this in an interview)

- **Why the refine loop is capped:** `max_refines` bounds the Critic↔Researcher
  cycle so a stubborn claim with genuinely thin literature can't loop forever.
  Once the cap is hit, the graph proceeds to human review regardless of
  evidence strength — you'll correctly see "contested" verdicts for claims
  where the literature really is mixed or sparse.
- **Why `papers` isn't an auto-accumulating reducer field:** dedup logic
  (`dedupe_papers` in `state.py`) is explicit and node-local rather than
  hidden behind a LangGraph reducer, which makes it easier to reason about
  and to explain.
- **Why RAG instead of dumping raw abstracts:** the Synthesiser retrieves the
  top-k most relevant abstracts *per sub-question* from ChromaDB rather than
  stuffing every approved abstract into one prompt — keeps the prompt
  grounded and scales to larger shortlists.
- **Where the eval deviates from a literal graph node:** RAGAS evaluation is
  implemented as a standalone batch script (`eval/run_eval.py`) rather than a
  graph node, since faithfulness scoring is naturally a post-hoc batch
  operation over many runs, not a per-conversation step.

## Resume bullet

Fill this in with your *actual* numbers from `eval/run_eval.py` — see the
setup notes above:

> Built a multi-agent literature-review system (LangGraph, Groq/Llama,
> arXiv + Semantic Scholar APIs) with Planner, Researcher, Critic, and
> Synthesiser agents; Critic-to-Researcher conditional loop auto-refines
> search terms on weak evidence; human-in-the-loop checkpoint via LangGraph
> interrupt; evaluated on 20 claims achieving RAGAS faithfulness **[your
> score]**; LangSmith observability integrated.

## License

MIT — see [LICENSE](LICENSE).
