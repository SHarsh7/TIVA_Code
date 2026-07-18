"""
EV Bus Intelligence — Streamlit showcase for the RAG pipeline.

Two views:
  - Dashboard: live market/evidence-base analytics (Plotly)
  - Ask the Corpus: chatbot over the tuned hybrid retrieval + generation stack

Run:  streamlit run app.py
"""
import os
import re
import sys
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
RAG_DIR = os.path.join(BASE, "data", "rag")

# ---------------------------------------------------------------- palette
BLUE, GREEN, AQUA, ORANGE, RED, YELLOW, VIOLET = (
    "#2a78d6", "#008300", "#1baf7a", "#eb6834", "#e34948", "#eda100", "#4a3aa7")
INK, INK2, MUTED, GRID, SURFACE, PLANE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#f9f9f7")

st.set_page_config(page_title="EV Bus Intelligence — Strategic AI Taskforce",
                    page_icon="🚌", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
.stat-card {{ background:{PLANE}; border:1px solid {GRID}; border-radius:10px;
  padding:14px 16px; text-align:left; }}
.stat-card .num {{ font-size:26px; font-weight:800; color:{BLUE}; line-height:1.1; }}
.stat-card .lab {{ font-size:12px; color:{INK2}; margin-top:4px; }}
.hero {{ background: linear-gradient(120deg,#0d2440,#1a5a8a 60%,#1baf7a);
  border-radius: 14px; padding: 22px 28px; color: white; margin-bottom: 1.2rem; }}
.hero h1 {{ font-size: 26px; margin: 0 0 4px 0; }}
.hero p {{ margin:0; opacity:.92; font-size:14px; }}
.src-chip {{ display:inline-block; background:{PLANE}; border:1px solid {GRID};
  border-radius:14px; padding:3px 10px; margin:2px 4px 2px 0; font-size:12px; color:{INK2}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚌 EV Bus Intelligence — Strategic AI Taskforce</h1>
  <p>Multi-modal RAG system over 2,852 evidence documents · India's school &amp; employee EV-bus transport market · TIVA AI 2026</p>
</div>
""", unsafe_allow_html=True)

view = st.sidebar.radio("View", ["📊 Dashboard", "💬 Ask the Corpus"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.caption(
    "Hybrid dense (MiniLM/FAISS) + sparse (BM25) retrieval, Reciprocal Rank Fusion, "
    "cross-encoder reranking, and a citation-locked, refusal-gated generation contract."
)
st.sidebar.caption("Set `GROQ_API_KEY` (free tier, model `llama-3.3-70b-versatile`) or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` for live LLM answers — otherwise the app shows the retrieved evidence pack.")

_llm_badge = "🔴 Evidence-pack mode (no LLM key set)"
if os.environ.get("GROQ_API_KEY"):
    _llm_badge = f"🟢 Live generation via Groq ({os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')})"
elif os.environ.get("ANTHROPIC_API_KEY"):
    _llm_badge = "🟢 Live generation via Anthropic"
elif os.environ.get("OPENAI_API_KEY"):
    _llm_badge = "🟢 Live generation via OpenAI"
elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
    _llm_badge = "🟢 Live generation via Gemini"
st.sidebar.caption(_llm_badge)

# ==================================================================
# DATA LOADERS (cached)
# ==================================================================

@st.cache_data(show_spinner=False)
def load_text_df():
    return pd.read_csv(os.path.join(BASE, "EV_Bus_Text_Data.csv"), encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def load_media_df():
    return pd.read_csv(os.path.join(BASE, "EV_Bus_Media_Data.csv"), encoding="utf-8-sig")

@st.cache_data(show_spinner=False)
def load_tuning():
    import json
    with open(os.path.join(RAG_DIR, "tuning_results.json"), encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def compute_sentiment_split(text_df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    c = text_df[text_df.doc_type == "comment"].copy()
    safety_kw = re.compile(r"safe|accident|pothole|crash|toppl|fire|brake|rash|rogue|injur|risk", re.I)
    c["is_safety"] = c.text.astype(str).str.contains(safety_kw)
    c["vader"] = c.text.astype(str).map(lambda t: an.polarity_scores(t)["compound"])
    c["bucket"] = pd.cut(c.vader, [-1.01, -0.05, 0.05, 1.01], labels=["neg", "neu", "pos"])
    out = {}
    for grp, sub in [("Safety-related", c[c.is_safety]), ("All other", c[~c.is_safety])]:
        n = len(sub)
        pct = (sub.bucket.value_counts(normalize=True) * 100).round(0)
        out[grp] = {"n": n, "neg": pct.get("neg", 0), "pos": pct.get("pos", 0)}
    return out

@st.cache_data(show_spinner=False)
def compute_salience(text_df):
    LEX = {
        "Safety": r"\bsaf(e|ety)\b|accident|crash|brake|toppl|fire|injur",
        "Economics": r"\bcost|price|₹|rupee|subsid|fame|per.?km|tco|fare",
    }
    voice_map = {"comment": "Riders & parents", "article": "Media", "report_pdf": "Institutions"}
    d = text_df[text_df.doc_type.isin(voice_map)].copy()
    d["voice"] = d.doc_type.map(voice_map)
    rows = []
    for v, sub in d.groupby("voice"):
        words = sub.word_count.sum()
        row = {"voice": v}
        for k, pat in LEX.items():
            hits = sub.text.astype(str).str.count(pat, flags=re.I).sum()
            row[k] = round(100 * hits / max(words, 1), 2)
        rows.append(row)
    return pd.DataFrame(rows)

# ==================================================================
# RAG STACK LOADER (cached resource — heavy: FAISS + models)
# ==================================================================

@st.cache_resource(show_spinner="Loading retrieval stack (FAISS index, embedding model, reranker)…")
def get_stack():
    from rag_query import load_stack
    return load_stack()

def run_query(query, k=5):
    from rag_query import retrieve, build_context_pack, llm_generate, SYSTEM_PROMPT, REFUSAL_GATE, REFUSAL_TEXT
    index, chunks_df, bm25, model, reranker = get_stack()
    hits = retrieve(query, index, chunks_df, bm25, model, reranker, k=k)
    dense_max = float(hits.dense_max.iloc[0])
    if dense_max < REFUSAL_GATE:
        return {"refused": True, "dense_max": dense_max, "hits": hits, "answer": REFUSAL_TEXT}
    ctx = build_context_pack(hits)
    prompt = f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"
    answer = llm_generate(SYSTEM_PROMPT, prompt)
    live = answer is not None
    if not live:
        answer = ("_No LLM API key configured — showing the retrieved, reranked evidence pack "
                   "that would be handed to the model:_\n\n" + ctx[:2200] +
                   ("…" if len(ctx) > 2200 else ""))
    return {"refused": False, "dense_max": dense_max, "hits": hits, "answer": answer, "live": live}

# ==================================================================
# DASHBOARD
# ==================================================================
if view == "📊 Dashboard":
    text_df = load_text_df()
    media_df = load_media_df()
    tuning = load_tuning()
    best = tuning["best"]

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{len(text_df) + len(media_df[media_df.media_type=='video']):,}", "Evidence documents (text + video)"),
        (f"{best['hit_rate_at5']:.3f}", "Hit@5 retrieval accuracy (tuned)"),
        (f"{best['precision_at5']:.3f}", "Precision@5 (tuned)"),
        (f"{best['mean_latency_ms']:.1f} ms", "Mean query latency"),
    ]
    for col, (num, lab) in zip([c1, c2, c3, c4], stats):
        col.markdown(f'<div class="stat-card"><div class="num">{num}</div><div class="lab">{lab}</div></div>',
                      unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)

    with left:
        st.subheader("Corpus composition")
        comp = text_df.groupby("doc_type").agg(docs=("doc_id", "size"), words=("word_count", "sum")).reset_index()
        fig = go.Figure()
        fig.add_bar(y=comp.doc_type, x=comp.docs, orientation="h", marker_color=BLUE, name="documents")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
                           xaxis_title="documents", showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with right:
        st.subheader("Safety vs. economics salience by voice")
        sal = compute_salience(text_df)
        fig = go.Figure()
        fig.add_bar(x=sal.voice, y=sal.Safety, name="Safety", marker_color=YELLOW)
        fig.add_bar(x=sal.voice, y=sal.Economics, name="Economics", marker_color=BLUE)
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                           plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
                           yaxis_title="mentions / 100 words", legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width='stretch')

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Sentiment: safety threads vs. all other")
        with st.spinner("Scoring sentiment (VADER)…"):
            split = compute_sentiment_split(text_df)
        cats = list(split.keys())
        fig = go.Figure()
        fig.add_bar(y=cats, x=[-split[c]["neg"] for c in cats], orientation="h", marker_color=RED, name="Negative %")
        fig.add_bar(y=cats, x=[split[c]["pos"] for c in cats], orientation="h", marker_color=BLUE, name="Positive %")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), barmode="overlay",
                           plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK, size=12),
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Safety-related n={int(split['Safety-related']['n'])} · All other n={int(split['All other']['n'])}")

    with right2:
        st.subheader("Retrieval accuracy — baseline vs. tuned")
        grid = pd.DataFrame(tuning["grid"])
        base_row = grid[grid.config.str.contains("500tok/15%", na=False)]
        base = base_row.iloc[0] if len(base_row) else grid.iloc[0]
        metrics = ["hit_rate_at5", "mrr_at10", "precision_at5"]
        labels = ["Hit@5", "MRR@10", "Precision@5"]
        fig = go.Figure()
        fig.add_bar(x=labels, y=[base[m] for m in metrics], name="Baseline", marker_color=MUTED)
        fig.add_bar(x=labels, y=[best[m] for m in metrics], name="Tuned & promoted", marker_color=BLUE)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), barmode="group",
                           plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK, size=12), yaxis_range=[0, 1.1],
                           legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig, width='stretch')

    st.subheader("Channel share of voice (video corpus)")
    vids = media_df[media_df.media_type == "video"].copy()
    ch = (vids.groupby("organization")
          .agg(n=("media_id", "size"), views=("view_count", "sum"))
          .query("n >= 2").sort_values("views", ascending=False).head(8).reset_index())
    fig = go.Figure()
    fig.add_bar(x=ch.organization, y=ch.views, marker_color=AQUA)
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=dict(color=INK, size=12), yaxis_title="cumulative views")
    st.plotly_chart(fig, width='stretch')

    st.info("💡 Full write-up, TCO analysis, and strategic recommendations are in **Strategic_Business_Report.docx**.")

# ==================================================================
# CHATBOT
# ==================================================================
else:
    st.subheader("💬 Ask the Corpus")
    st.caption("Queries run against the tuned hybrid retrieval index (512 tok / 10% overlap, cosine, MRR@10 = 1.000). "
               "Out-of-domain questions are refused before any generation happens.")

    examples = [
        "What concerns do parents raise about school bus safety?",
        "What is the total cost of ownership of electric vs diesel buses in India?",
        "How do employees experience Cityflo and Shuttl shuttle services?",
        "What financing models exist for scaling electric bus fleets?",
        "How do OEMs like Tata and JBM market safety features?",
    ]
    cols = st.columns(len(examples))
    picked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, width='stretch'):
            picked = ex

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(" ".join(f'<span class="src-chip">[{s}]</span>' for s in msg["sources"]),
                            unsafe_allow_html=True)

    query = st.chat_input("Ask about EV bus adoption, safety, TCO, financing, OEM positioning…") or picked

    if query:
        st.session_state.chat.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating…"):
                result = run_query(query)
            if result["refused"]:
                st.warning(f"🚫 **Insufficient evidence in corpus.** (max dense score {result['dense_max']:.3f} < 0.35 refusal gate)")
                st.session_state.chat.append({"role": "assistant", "content":
                    f"🚫 Insufficient evidence in corpus. (max dense score {result['dense_max']:.3f} < 0.35)"})
            else:
                st.markdown(result["answer"])
                hits = result["hits"]
                src_list = [f"{h.chunk_id} · {h.source_name}" for _, h in hits.iterrows()]
                with st.expander(f"📎 Top {len(hits)} evidence sources (reranked)"):
                    for _, h in hits.iterrows():
                        st.markdown(f"**[{h.chunk_id}]** {h.source_name} — *{h.voice}*")
                        st.caption(h.text[:280] + ("…" if len(h.text) > 280 else ""))
                st.session_state.chat.append({"role": "assistant", "content": result["answer"], "sources": src_list})
