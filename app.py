"""
Streamlit UI for ScholarMind.

Run with: streamlit run app.py

Flow: submit a topic+claim -> graph runs Planner -> Researcher -> Critic
(looping internally if evidence is weak) -> pauses at the human_review
interrupt -> this UI renders the scored shortlist as checkboxes -> on
"Approve and synthesize" we resume the graph with Command(resume=...) ->
Synthesiser runs -> final report is rendered.
"""
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from scholarmind.graph import build_graph

load_dotenv()

st.set_page_config(page_title="ScholarMind", layout="wide")
st.title("ScholarMind")
st.caption("Multi-agent literature review — Planner → Researcher ⇄ Critic → Human review → Synthesiser")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "result" not in st.session_state:
    st.session_state.result = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

with st.form("claim_form"):
    topic = st.text_input("Topic", placeholder="e.g. Sequence modeling")
    claim = st.text_area("Claim to verify", placeholder="e.g. Transformers outperform RNNs on long sequences")
    min_evidence = st.slider("Minimum strong-evidence papers before stopping", 2, 8, 4)
    submitted = st.form_submit_button("Run literature review")

if submitted and topic and claim:
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "topic": topic,
        "claim": claim,
        "refine_count": 0,
        "max_refines": 2,
        "min_strong_evidence": min_evidence,
    }
    with st.spinner("Planning sub-questions, fetching papers, scoring evidence..."):
        st.session_state.result = st.session_state.graph.invoke(initial_state, config=config)

result = st.session_state.result

if result and "__interrupt__" in result:
    st.subheader("Review the shortlist before synthesis")
    st.caption("The Critic scored each paper. Untick anything irrelevant, then approve.")

    payload = result["__interrupt__"][0].value
    papers = payload["papers"]
    approved_ids = []

    for p in sorted(papers, key=lambda x: -abs(x.get("stance", 0))):
        stance = p.get("stance", 0)
        conf = p.get("confidence", 0.0)
        label = f"[{stance:+d}, conf {conf:.2f}] {p['title']} ({p.get('year', 'n/a')}, {p.get('source')})"
        default = stance != 0
        checked = st.checkbox(label, value=default, key=p.get("paper_id", p["title"]))
        with st.expander("abstract / rationale"):
            st.write(p.get("abstract", "(no abstract)"))
            st.caption(p.get("rationale", ""))
        if checked:
            approved_ids.append(p.get("paper_id"))

    if st.button("Approve and synthesize", type="primary"):
        with st.spinner("Synthesising report..."):
            st.session_state.result = st.session_state.graph.invoke(
                Command(resume={"approved_paper_ids": approved_ids}), config=config
            )
        st.rerun()

elif result and result.get("report"):
    report = result["report"]
    st.subheader(f"Verdict: {report['verdict'].upper()}  (confidence {report['confidence']:.2f})")
    st.write(report["overall_summary"])

    st.markdown("**Evidence by sub-question**")
    for ev in report["evidence_summary"]:
        st.markdown(f"- **{ev['sub_question']}** — {ev['summary']}")

    st.markdown("**Citations**")
    for c in report["citations"]:
        st.markdown(f"- {c['title']} ({c.get('year', 'n/a')}) — {c.get('url', '')}")

    if st.session_state.get("graph"):
        refine_count = result.get("refine_count", 0)
        st.caption(
            f"Refine loop fired {refine_count} time(s) · "
            f"{len(result.get('papers', []))} papers considered · "
            f"{len(result.get('approved_papers', []))} approved for synthesis"
        )

if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    st.warning("No GROQ_API_KEY or OPENAI_API_KEY found. Copy .env.example to .env and add a key.")
