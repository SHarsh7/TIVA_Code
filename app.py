"""
EV Bus Intelligence — Streamlit showcase for the RAG pipeline.

Two views:
  - Dashboard: interactive market/evidence analytics (Plotly, click-to-drill)
  - Ask the Corpus: chatbot over the tuned hybrid retrieval + generation stack

Run:  streamlit run app.py
"""
import os
import re
import sys
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
RAG_DIR = os.path.join(BASE, "data", "rag")

# ---------------------------------------------------------------- design tokens
# Validated categorical palette (dataviz reference instance, light mode)
BLUE, GREEN, AQUA, ORANGE, RED, YELLOW, VIOLET = (
    "#2a78d6", "#008300", "#1baf7a", "#eb6834", "#e34948", "#eda100", "#4a3aa7")
BLUE_SOFT = "#b7d3f6"          # sequential step 150 — de-emphasised bars
INK, INK2, MUTED, GRID, SURFACE, PLANE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#f9f9f7")
BASELINE = "#c3c2b7"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
PLOT_CFG = {"displayModeBar": False}

VOICE_META = {
    "public_rider_parent": ("Riders & parents", ORANGE),
    "institutional":       ("Institutions", VIOLET),
    "creator_or_oem":      ("Creators & OEMs", AQUA),
    "media":               ("News media", BLUE),
    "oem_marketing":       ("OEM marketing", GREEN),
}

VIEW_DASH = ":material/monitoring: Dashboard"
VIEW_CHAT = ":material/forum: Ask the Corpus"

# ---------------------------------------------------------------- inline icons
_ICON_PATHS = {
    "bus": '<path d="M8 6v6"/><path d="M15 6v6"/><path d="M2 12h19.6"/>'
           '<path d="M18 18h3s.5-1.7.8-2.8c.1-.4.2-.8.2-1.2 0-.4-.1-.8-.2-1.2l-1.4-5'
           'C20.1 6.8 19.1 6 18 6H4a2 2 0 0 0-2 2v10h3"/>'
           '<circle cx="7" cy="18" r="2"/><path d="M9 18h5"/><circle cx="16" cy="18" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/>'
                '<path d="M3 12a9 3 0 0 0 18 0"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
              '<circle cx="12" cy="12" r="2"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
             '<polyline points="22 4 12 14.01 9 11.01"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>'
           '<circle cx="12" cy="10" r="3"/>',
    "chat": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>'
           '<path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "clip": '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84'
            'l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>',
    "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6'
              'a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1 1 0 0 1 1.52 0C14.51 3.81 17 5 19 5'
              'a1 1 0 0 1 1 1z"/>',
}

def icon(name, size=16, color="currentColor", sw=2):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="vertical-align:-2px">{_ICON_PATHS[name]}</svg>')


st.set_page_config(page_title="EV Bus Intelligence — Strategic AI Taskforce",
                   page_icon=":material/directions_bus:", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1180px; }}
h3 {{ font-size: 1.05rem !important; letter-spacing: -0.01em; }}

/* ---------- hero ---------- */
.hero {{
  position: relative; overflow: hidden;
  background: linear-gradient(115deg, #0a1f38 0%, #123a63 45%, #14567e 75%, #0f7a5c 100%);
  border-radius: 16px; padding: 26px 30px 22px; color: #fff; margin-bottom: 1.3rem;
}}
.hero::after {{
  content: ""; position: absolute; right: -80px; top: -120px; width: 340px; height: 340px;
  background: radial-gradient(circle, rgba(255,255,255,.14) 0%, rgba(255,255,255,0) 70%);
}}
.hero .eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; opacity: .75; margin-bottom: 6px; }}
.hero h1 {{ font-size: 26px; font-weight: 800; margin: 0 0 6px; letter-spacing: -0.015em; }}
.hero p  {{ margin: 0 0 12px; opacity: .88; font-size: 14px; max-width: 740px; line-height: 1.55; }}
.hero.compact {{ padding: 18px 24px 15px; margin-bottom: 1rem; }}
.hero.compact h1 {{ font-size: 19px; margin-bottom: 3px; }}
.hero.compact p {{ margin: 0; font-size: 13px; }}
.hero-chips span {{
  display: inline-block; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px; padding: 3px 12px; margin: 2px 6px 0 0; font-size: 12px; font-weight: 600;
}}

/* ---------- KPI stat tiles ---------- */
.stat-card {{
  background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px;
  padding: 16px 18px 15px; height: 100%; box-shadow: 0 1px 2px rgba(11,11,11,.04);
  transition: box-shadow .15s ease, transform .15s ease;
}}
.stat-card:hover {{ box-shadow: 0 6px 18px rgba(11,11,11,.08); transform: translateY(-1px); }}
.stat-top {{ display: flex; justify-content: space-between; align-items: flex-start; }}
.icon-chip {{ width: 32px; height: 32px; border-radius: 9px; display: flex;
  align-items: center; justify-content: center; flex: none; }}
.stat-card .lab {{
  font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  color: {MUTED}; margin: 2px 0 10px;
}}
.stat-card .num {{ font-size: 30px; font-weight: 800; color: {INK}; line-height: 1.05; }}
.stat-card .num small {{ font-size: 15px; font-weight: 700; color: {INK2}; }}
.stat-card .sub {{ font-size: 12px; color: {INK2}; margin-top: 6px; line-height: 1.4; }}

.mini-tile {{ background: {PLANE}; border: 1px solid {GRID}; border-radius: 10px;
  padding: 10px 13px; margin-bottom: 8px; }}
.mini-tile .k {{ font-size: 10.5px; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; color: {MUTED}; }}
.mini-tile .v {{ font-size: 21px; font-weight: 800; color: {INK}; line-height: 1.2; }}
.mini-tile .v small {{ font-size: 12px; font-weight: 600; color: {MUTED}; }}

/* ---------- tabs (BI pages) ---------- */
div[data-testid="stTabs"] button[data-baseweb="tab"] {{
  font-weight: 600; letter-spacing: .01em; color: {MUTED}; padding: 8px 4px;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {BLUE}; }}
div[data-baseweb="tab-highlight"] {{ background-color: {BLUE}; height: 2.5px; }}
div[data-baseweb="tab-border"] {{ background-color: {GRID}; }}

/* ---------- chart cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 14px !important; border-color: {GRID} !important; background: {SURFACE};
  box-shadow: 0 1px 2px rgba(11,11,11,.04);
}}
.chart-title {{ font-size: 14.5px; font-weight: 700; color: {INK}; margin: 2px 0 0; }}
.chart-sub   {{ font-size: 12px; color: {MUTED}; margin: 1px 0 4px; line-height: 1.4; }}
.chart-note  {{ font-size: 12.5px; color: {INK2}; margin-top: 2px; line-height: 1.55;
               border-top: 1px dashed {GRID}; padding-top: 9px; }}
.note-tag {{ display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: .09em;
            text-transform: uppercase; color: {BLUE}; margin-right: 7px; }}
.drill-hint {{ font-size: 11.5px; color: {MUTED}; margin: -2px 0 4px; }}
.panel-head {{ font-size: 13px; font-weight: 700; color: {INK}; margin: 2px 0 8px;
              display: flex; align-items: center; gap: 7px; }}

/* ---------- section eyebrows ---------- */
.sec-head {{
  font-size: 12px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
  color: {MUTED}; margin: 1.5rem 0 .5rem; display: flex; align-items: center; gap: 10px;
}}
.sec-head::after {{ content: ""; flex: 1; height: 1px; background: {GRID}; }}

/* ---------- spec sheet ---------- */
.spec-row {{ display: flex; justify-content: space-between; gap: 12px;
            padding: 7px 2px; border-bottom: 1px dashed {GRID}; font-size: 13px; }}
.spec-row:last-child {{ border-bottom: none; }}
.spec-row .k {{ color: {MUTED}; }}
.spec-row .v {{ color: {INK}; font-weight: 600; text-align: right; }}

/* ---------- comment cards ---------- */
.cmt-card {{ background: {PLANE}; border: 1px solid {GRID}; border-radius: 10px;
  padding: 10px 14px; margin: 6px 0; }}
.cmt-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
.cmt-city {{ border-radius: 999px; padding: 1px 9px; font-size: 10.5px; font-weight: 700;
  background: {BLUE}1f; color: {BLUE}; }}
.cmt-score {{ font-size: 11px; color: {MUTED}; font-weight: 600; display: flex;
  align-items: center; gap: 3px; }}
.cmt-mood {{ width: 8px; height: 8px; border-radius: 50%; flex: none; }}
.cmt-text {{ font-size: 12.5px; color: {INK2}; line-height: 1.5; }}

/* ---------- chat ---------- */
.stChatMessage {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px; }}
div[data-testid="stChatInput"] {{ border-radius: 14px; }}

.welcome {{
  background: {SURFACE}; border: 1px solid {GRID}; border-radius: 14px;
  padding: 24px 28px; margin-bottom: .6rem; box-shadow: 0 1px 2px rgba(11,11,11,.04);
}}
.welcome h4 {{ margin: 0 0 6px; font-size: 17px; color: {INK}; }}
.welcome p  {{ margin: 0; font-size: 13.5px; color: {INK2}; line-height: 1.55; max-width: 660px; }}

/* example-question chips (all main-area buttons render as pills) */
div[data-testid="stMain"] div[data-testid="stButton"] > button {{
  border-radius: 999px; border: 1px solid {GRID}; background: {SURFACE};
  color: {INK2}; font-size: 13px; font-weight: 500; padding: 6px 14px; width: 100%;
  transition: all .12s ease;
}}
div[data-testid="stMain"] div[data-testid="stButton"] > button:hover {{
  border-color: {BLUE}; color: {BLUE}; background: #f2f7fd;
}}

/* answer meta strip */
.ans-meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
.conf-pill {{
  display: inline-flex; align-items: center; gap: 6px; border-radius: 999px;
  padding: 2px 11px; font-size: 11.5px; font-weight: 600; border: 1px solid {GRID};
  background: {PLANE}; color: {INK2};
}}
.conf-pill .dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}

/* source cards */
.src-card {{
  background: {PLANE}; border: 1px solid {GRID}; border-left: 3px solid {BLUE};
  border-radius: 10px; padding: 10px 14px; margin: 6px 0;
}}
.src-top {{ display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }}
.src-name {{ font-size: 13px; font-weight: 700; color: {INK}; }}
.voice-pill {{ border-radius: 999px; padding: 1px 9px; font-size: 10.5px; font-weight: 700;
              white-space: nowrap; }}
.src-snippet {{ font-size: 12.5px; color: {INK2}; margin: 5px 0 4px; line-height: 1.5; }}
.src-meta {{ font-size: 11.5px; color: {MUTED}; }}
.src-meta a {{ color: {BLUE}; text-decoration: none; font-weight: 600; }}

/* ---------- sidebar ---------- */
.side-brand {{ display: flex; align-items: center; gap: 9px; }}
.side-logo {{ width: 34px; height: 34px; border-radius: 10px; flex: none;
  background: linear-gradient(135deg, #123a63, #0f7a5c); display: flex;
  align-items: center; justify-content: center; }}
.side-title {{ font-size: 15px; font-weight: 800; color: {INK}; letter-spacing: -0.01em;
  line-height: 1.15; }}
.side-sub {{ font-size: 11px; color: {MUTED}; }}
.engine-badge {{
  border-radius: 10px; padding: 9px 12px; font-size: 12px; font-weight: 600;
  line-height: 1.45; border: 1px solid; margin: 4px 0 2px;
  display: flex; gap: 8px; align-items: flex-start;
}}
.engine-badge .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex: none; }}
.engine-badge.on  {{ background: #eef7ee; border-color: #bfe0bf; color: #135c13; }}
.engine-badge.on .dot {{ background: #0ca30c; }}
.engine-badge.off {{ background: #fdf3e7; border-color: #f0d9b8; color: #7a5210; }}
.engine-badge.off .dot {{ background: #eda100; }}
.side-stat {{ display: flex; justify-content: space-between; font-size: 12.5px;
             color: {INK2}; padding: 3px 0; }}
.side-stat b {{ color: {INK}; }}
</style>
""", unsafe_allow_html=True)

# ==================================================================
# SIDEBAR
# ==================================================================
with st.sidebar:
    st.markdown(
        f'<div class="side-brand"><div class="side-logo">{icon("bus", 18, "#ffffff")}</div>'
        f'<div><div class="side-title">EV Bus Intelligence</div>'
        f'<div class="side-sub">Strategic AI Taskforce · TIVA AI 2026</div></div></div>',
        unsafe_allow_html=True)
    st.write("")
    view = st.radio("View", [VIEW_DASH, VIEW_CHAT], label_visibility="collapsed")

    st.markdown("---")
    if os.environ.get("GROQ_API_KEY"):
        engine = ("on", f"Live answers on<br><span style='font-weight:400'>Groq · "
                        f"{os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')}</span>")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        engine = ("on", "Live answers on<br><span style='font-weight:400'>Anthropic Claude</span>")
    elif os.environ.get("OPENAI_API_KEY"):
        engine = ("on", "Live answers on<br><span style='font-weight:400'>OpenAI</span>")
    elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        engine = ("on", "Live answers on<br><span style='font-weight:400'>Google Gemini</span>")
    else:
        engine = ("off", "Evidence-only mode<br><span style='font-weight:400'>"
                         "No LLM key set — the bot shows retrieved evidence instead of a written answer.</span>")
    st.markdown(f'<div class="engine-badge {engine[0]}"><span class="dot"></span>'
                f'<span>{engine[1]}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="side-stat"><span>Evidence documents</span><b>2,852</b></div>'
                '<div class="side-stat"><span>Index chunks</span><b>3,113</b></div>'
                '<div class="side-stat"><span>Stakeholder voices</span><b>5</b></div>'
                '<div class="side-stat"><span>Retrieval Hit@5</span><b>100%</b></div>',
                unsafe_allow_html=True)

    with st.expander("How answers are built", icon=":material/route:"):
        st.markdown(
            "1. **Search** — your question is matched against 3,113 chunks two ways: "
            "by meaning (MiniLM + FAISS) and by keywords (BM25).\n"
            "2. **Fuse & rerank** — both result lists are merged (Reciprocal Rank Fusion), "
            "then a cross-encoder re-scores the top pool.\n"
            "3. **Gate** — off-topic questions are refused before any generation.\n"
            "4. **Write** — the LLM answers **only** from the retrieved evidence, "
            "with every source listed below the answer."
        )

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
def load_chunk_voices():
    df = pd.read_parquet(os.path.join(RAG_DIR, "chunk_metadata.parquet"), columns=["voice"])
    return df.voice.value_counts()

@st.cache_data(show_spinner=False)
def load_tuning():
    import json
    with open(os.path.join(RAG_DIR, "tuning_results.json"), encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_state_deployment():
    """State-wise e-bus counts parsed from the PIB / MoRTH press release in the corpus
    (Vahan 4 records as on 19 July 2024)."""
    df = pd.read_parquet(os.path.join(RAG_DIR, "chunk_metadata.parquet"),
                         columns=["doc_id", "text"])
    txt = df[df.doc_id == "pib_ebus_programme"].text.iloc[0]
    rows = re.findall(r"([A-Za-z][A-Za-z &.]+?) \| (\d+|-) \| (\d+|-) \|", txt)
    out = pd.DataFrame(rows, columns=["state", "pure", "hybrid"])
    for c in ("pure", "hybrid"):
        out[c] = out[c].replace("-", "0").astype(int)
    out["total"] = out.pure + out.hybrid
    return out.sort_values("total", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def comments_enriched():
    """All Reddit comments with city, VADER sentiment, and safety flag."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    c = load_text_df().query("doc_type == 'comment'").copy()
    c["city"] = c.source_name.str.extract(r"r/(\w+)")[0].str.capitalize()
    # score arrives as "99 points" strings for some scrape methods — keep the number
    c["score"] = (pd.to_numeric(c.score.astype(str).str.extract(r"(-?\d+)")[0],
                                errors="coerce").fillna(0).astype(int))
    safety_kw = re.compile(r"safe|accident|pothole|crash|toppl|fire|brake|rash|rogue|injur|risk", re.I)
    c["is_safety"] = c.text.astype(str).str.contains(safety_kw)
    c["vader"] = c.text.astype(str).map(lambda t: an.polarity_scores(t)["compound"])
    c["bucket"] = pd.cut(c.vader, [-1.01, -0.05, 0.05, 1.01], labels=["neg", "neu", "pos"])
    return c[["city", "text", "score", "vader", "is_safety", "bucket", "word_count"]]

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


def sentiment_split(c):
    out = {}
    for grp, sub in [("Safety threads", c[c.is_safety]), ("All other threads", c[~c.is_safety])]:
        pct = (sub.bucket.value_counts(normalize=True) * 100)
        out[grp] = {"n": len(sub),
                    "neg": float(pct.get("neg", 0) or 0),
                    "pos": float(pct.get("pos", 0) or 0)}
    return out

# ==================================================================
# CHART HELPERS (dataviz spec: hairline grid, muted ink, thin rounded bars)
# ==================================================================

def style_fig(fig, height=290, legend=False, x_grid=False, y_grid=True):
    fig.update_layout(
        height=height, margin=dict(l=4, r=8, t=6, b=4),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(family=FONT, color=INK2, size=12.5),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(family=FONT, color=INK, size=12.5)),
        showlegend=legend, barcornerradius=4, bargap=0.42, dragmode=False,
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=1.14, x=0, font=dict(size=12)))
    fig.update_xaxes(showgrid=x_grid, gridcolor=GRID, gridwidth=1,
                     zeroline=False, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11.5), title=None)
    fig.update_yaxes(showgrid=y_grid, gridcolor=GRID, gridwidth=1,
                     zeroline=False, linecolor=BASELINE,
                     tickfont=dict(color=MUTED, size=11.5), title=None)
    return fig


def chart_card(title, sub, fig, note=None, key=None, drill_hint=None):
    """Bordered card: title, subtitle, chart, plain-English explanation.
    Pass key= to make the chart click-selectable (returns the selection event)."""
    event = None
    with st.container(border=True):
        st.markdown(f'<p class="chart-title">{title}</p><p class="chart-sub">{sub}</p>',
                    unsafe_allow_html=True)
        if drill_hint:
            st.markdown(f'<p class="drill-hint">{icon("pin", 12, MUTED)} {drill_hint}</p>',
                        unsafe_allow_html=True)
        if key:
            event = st.plotly_chart(fig, width='stretch', config=PLOT_CFG,
                                    key=key, on_select="rerun", selection_mode="points")
        else:
            st.plotly_chart(fig, width='stretch', config=PLOT_CFG)
        if note:
            st.markdown(f'<p class="chart-note"><span class="note-tag">What this shows</span>'
                        f'{note}</p>', unsafe_allow_html=True)
    return event


def selected_value(key):
    """Read the clicked bar's customdata (or category) from a chart's stored selection."""
    state = st.session_state.get(key)
    if not state:
        return None
    try:
        pts = state["selection"]["points"]
    except (KeyError, TypeError):
        return None
    if not pts:
        return None
    p = pts[0]
    cd = p.get("customdata")
    if cd:
        return cd[0] if isinstance(cd, (list, tuple)) else cd
    return p.get("y") if p.get("y") is not None else p.get("x")


def fmt_count(n):
    n = float(n)
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.0f}K"
    return f"{n:.0f}"


def mini_tile(label, value, suffix=""):
    sfx = f" <small>{suffix}</small>" if suffix else ""
    return (f'<div class="mini-tile"><div class="k">{label}</div>'
            f'<div class="v">{value}{sfx}</div></div>')

# ==================================================================
# RAG STACK + QUERY PIPELINE
# ==================================================================

@st.cache_resource(show_spinner="Warming up the retrieval stack (FAISS index, embedder, reranker)…")
def get_stack():
    from rag_query import load_stack
    return load_stack()


FRIENDLY_REFUSAL = ("I couldn't find enough information about that in our data. "
                    "Try asking about EV bus safety, cost, charging, or policy in India.")

SYSTEM_PROMPT = """You are the analyst behind "EV Bus Intelligence" — a research assistant that
answers questions about India's electric school-bus and employee-shuttle market
using ONLY the evidence supplied in CONTEXT.

HOW TO ANSWER
- Open with a direct answer to the question in the first sentence, then back it up.
- Keep it genuinely easy to read: short sentences, everyday words, no business jargon,
  no filler like "Based on the context provided".
- Use markdown well: **bold** the few numbers and phrases that carry the answer; use
  "-" bullets when comparing or listing three or more things. Never use headings.
- Attribute facts naturally by describing the source — "parents on Reddit said…",
  "a government press release lists…", "one industry report projects…". NEVER show
  chunk IDs, URLs, bracketed tags, or file names in the answer.
- Match length to the question: a simple factual question needs 2-4 sentences; a
  comparison or "explain" question can take two short paragraphs or a bulleted
  list. Never pad.
- When sources disagree or tell different stories (worried parents vs. glossy OEM
  marketing, for example), point out the tension — that contrast is the insight.
- If this is a follow-up, use CONVERSATION SO FAR to resolve what "it" or "they"
  refers to, and don't repeat what you already told the visitor.

HARD RULES
1. Every fact must come from CONTEXT. Add nothing from outside knowledge, however
   confident you are.
2. Never invent numbers, names, prices, or sources.
3. If CONTEXT does not contain enough to answer the question, reply exactly:
""" + FRIENDLY_REFUSAL


def confidence_of(dense_max):
    if dense_max >= 0.60:
        return "High confidence", GREEN
    if dense_max >= 0.45:
        return "Good confidence", BLUE
    return "Partial match", YELLOW


def retrieval_query_for(query, history):
    """Short follow-ups ("what about diesel?", "why?") retrieve poorly on their
    own — fold in the previous user question for context."""
    prior = [m["content"] for m in history if m["role"] == "user"]
    if not prior:
        return query
    short = len(query.split()) <= 5
    anaphoric = re.search(r"\b(it|they|them|those|these|that|this|why|how so|what about)\b",
                          query, re.I) and len(query.split()) <= 9
    if short or anaphoric:
        return f"{prior[-1]} {query}"
    return query


def run_retrieval(query, history, k=8):
    from rag_query import retrieve, REFUSAL_GATE
    index, chunks_df, bm25, model, reranker = get_stack()
    rq = retrieval_query_for(query, history)
    hits = retrieve(rq, index, chunks_df, bm25, model, reranker, k=k, pool=24)
    dense_max = float(hits.dense_max.iloc[0])
    return hits, dense_max, dense_max < REFUSAL_GATE


def build_user_prompt(query, hits, history):
    from rag_query import build_context_pack
    ctx = build_context_pack(hits)
    hist = ""
    turns = history[-6:]
    if turns:
        lines = []
        for m in turns:
            who = "Visitor" if m["role"] == "user" else "You"
            lines.append(f"{who}: {m['content'][:400]}")
        hist = "CONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"
    return f"{hist}CONTEXT:\n{ctx}\n\nQUESTION: {query}"


def sources_from_hits(hits):
    """Dedupe chunks to one card per document, preserving rerank order."""
    out, seen = [], set()
    for _, h in hits.iterrows():
        if h.source_name in seen:
            continue
        seen.add(h.source_name)
        snippet = str(h.text).strip().replace("\n", " ")
        out.append({
            "name": str(h.source_name),
            "org": str(h.organization) if pd.notna(h.organization) else "",
            "voice": str(h.voice),
            "url": str(h.url) if pd.notna(h.url) else "",
            "snippet": snippet[:220] + ("…" if len(snippet) > 220 else ""),
        })
    return out


def source_cards_html(sources):
    cards = []
    for s in sources:
        label, color = VOICE_META.get(s["voice"], (s["voice"].replace("_", " ").title(), MUTED))
        link = (f'<a href="{escape(s["url"], quote=True)}" target="_blank" '
                f'rel="noopener">Open source ↗</a>') if s["url"].startswith("http") else ""
        org = escape(s["org"])
        sep = " · " if (org and link) else ""
        cards.append(
            f'<div class="src-card" style="border-left-color:{color}">'
            f'<div class="src-top"><span class="src-name">{escape(s["name"])}</span>'
            f'<span class="voice-pill" style="background:{color}1f;color:{color}">{label}</span></div>'
            f'<div class="src-snippet">{escape(s["snippet"])}</div>'
            f'<div class="src-meta">{org}{sep}{link}</div></div>')
    return "".join(cards)


def render_answer_meta(msg):
    """Confidence pill + source count + expandable source cards."""
    if not msg.get("sources"):
        return
    label, color = msg.get("confidence", ("", MUTED))
    pills = (f'<span class="conf-pill"><span class="dot" style="background:{color}"></span>'
             f'{label}</span>'
             f'<span class="conf-pill">{icon("clip", 12)} {len(msg["sources"])} sources</span>')
    st.markdown(f'<div class="ans-meta">{pills}</div>', unsafe_allow_html=True)
    with st.expander("See the evidence behind this answer"):
        st.markdown(source_cards_html(msg["sources"]), unsafe_allow_html=True)

# ==================================================================
# DASHBOARD
# ==================================================================
if view == VIEW_DASH:
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">Strategic AI Taskforce · TIVA AI 2026</div>
      <h1>EV Bus Intelligence</h1>
      <p>What 2,852 pieces of real-world evidence — parent forums, news reports, government data,
         and OEM channels — say about electrifying India's school &amp; employee bus transport.</p>
      <div class="hero-chips"><span>Hybrid RAG pipeline</span><span>5 stakeholder voices</span>
      <span>Citation-locked answers</span><span>Evidence window 2023–2026</span></div>
    </div>
    """, unsafe_allow_html=True)

    text_df = load_text_df()
    media_df = load_media_df()
    tuning = load_tuning()
    best = tuning["best"]
    n_docs = len(text_df) + len(media_df[media_df.media_type == "video"])

    tiles = [
        ("database", BLUE,   f"{n_docs:,}", "Evidence base",
         "documents scraped &amp; verified — text + video"),
        ("target",   GREEN,  f"{best['hit_rate_at5']*100:.0f}<small>%</small>", "Retrieval Hit@5",
         "queries whose top-5 results contain the right evidence"),
        ("check",    VIOLET, f"{best['precision_at5']*100:.1f}<small>%</small>", "Precision@5",
         "of the top-5 retrieved chunks are on-target"),
        ("zap",      ORANGE, f"{best['mean_latency_ms']:.0f}<small> ms</small>", "Query latency",
         "mean end-to-end retrieval time"),
    ]
    for col, (ic, accent, num, lab, sub) in zip(st.columns(4), tiles):
        col.markdown(
            f'<div class="stat-card"><div class="stat-top"><div class="lab">{lab}</div>'
            f'<div class="icon-chip" style="background:{accent}14">{icon(ic, 16, accent)}</div></div>'
            f'<div class="num">{num}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True)

    st.write("")
    tab_market, tab_pulse, tab_media, tab_engine = st.tabs([
        ":material/map: Market snapshot",
        ":material/groups: Community pulse",
        ":material/play_circle: Media & reach",
        ":material/memory: Retrieval engine",
    ])

    # ---------------- TAB 1 · Market snapshot (state drilldown) ----------------
    with tab_market:
        states = load_state_deployment()
        sel_state = selected_value("state_chart")

        c_chart, c_panel = st.columns([3, 1.15])
        with c_chart:
            top = states.head(15).iloc[::-1]
            colors = [BLUE if (sel_state is None or s == sel_state) else BLUE_SOFT
                      for s in top.state]
            fig = go.Figure(go.Bar(
                y=top.state, x=top.total, orientation="h",
                marker=dict(color=colors),
                customdata=list(zip(top.state, top.pure, top.hybrid)),
                text=[f"{v:,}" for v in top.total], textposition="outside",
                textfont=dict(color=INK2, size=11.5),
                hovertemplate="<b>%{customdata[0]}</b><br>%{x:,} e-buses "
                              "(%{customdata[1]:,} pure · %{customdata[2]:,} hybrid)<extra></extra>"))
            fig.update_traces(selected=dict(marker=dict(opacity=1)),
                              unselected=dict(marker=dict(opacity=0.85)))
            style_fig(fig, height=430, y_grid=False)
            fig.update_xaxes(visible=False, range=[0, top.total.max() * 1.16])
            chart_card(
                "Electric buses on the road, by state",
                "Registered e-buses (pure + strong hybrid) · Vahan 4 records, July 2024",
                fig, key="state_chart",
                drill_hint="Click a state bar to drill down · click empty space to reset",
                note="Deployment is heavily concentrated: Maharashtra, Delhi and Karnataka alone "
                     "run nearly 60% of India's registered e-buses, while most states are still "
                     "below a few hundred. In plain terms — the market today is a handful of city "
                     "clusters, not a nationwide rollout, so early school- and shuttle-fleet plays "
                     "should target those clusters first.")

        with c_panel:
            with st.container(border=True):
                if sel_state and sel_state in set(states.state):
                    row = states[states.state == sel_state].iloc[0]
                    rank = int(states.index[states.state == sel_state][0]) + 1
                    share = 100 * row.total / states.total.sum()
                    st.markdown(f'<div class="panel-head">{icon("pin", 14, BLUE)} '
                                f'{escape(str(sel_state))}</div>', unsafe_allow_html=True)
                    st.markdown(
                        mini_tile("Total e-buses", f"{row.total:,}") +
                        mini_tile("Pure electric", f"{row.pure:,}") +
                        mini_tile("Strong hybrid", f"{row.hybrid:,}") +
                        mini_tile("National share", f"{share:.1f}", "%") +
                        mini_tile("Rank", f"#{rank}", f"of {len(states)}"),
                        unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="panel-head">{icon("pin", 14, BLUE)} '
                                f'All India</div>', unsafe_allow_html=True)
                    st.markdown(
                        mini_tile("Total e-buses", f"{states.total.sum():,}") +
                        mini_tile("Pure electric", f"{states.pure.sum():,}") +
                        mini_tile("Strong hybrid", f"{states.hybrid.sum():,}") +
                        mini_tile("States & UTs reporting", f"{len(states)}") +
                        mini_tile("Top state", escape(states.state.iloc[0])),
                        unsafe_allow_html=True)
        st.caption("Source: National Electric Bus Programme press release (PIB / MoRTH), "
                   "retrieved from the evidence corpus. Excludes Telangana & Lakshadweep "
                   "(not on centralized Vahan 4).")

    # ---------------- TAB 2 · Community pulse (city drilldown) ----------------
    with tab_pulse:
        cmt = comments_enriched()
        sel_city = selected_value("city_chart")
        cities = cmt.groupby("city").agg(
            n=("text", "size"), neg_share=("bucket", lambda b: (b == "neg").mean() * 100)
        ).sort_values("n").reset_index()

        c_left, c_right = st.columns(2)
        with c_left:
            colors = [BLUE if (sel_city is None or c == sel_city) else BLUE_SOFT
                      for c in cities.city]
            fig = go.Figure(go.Bar(
                y=cities.city, x=cities.n, orientation="h",
                marker=dict(color=colors),
                customdata=list(zip(cities.city, cities.neg_share)),
                text=[f"{v:,}" for v in cities.n], textposition="outside",
                textfont=dict(color=INK2, size=12),
                hovertemplate="<b>%{customdata[0]}</b><br>%{x:,} comments · "
                              "%{customdata[1]:.0f}% negative<extra></extra>"))
            style_fig(fig, height=280, y_grid=False)
            fig.update_xaxes(visible=False, range=[0, cities.n.max() * 1.18])
            chart_card(
                "Where the conversation happens",
                "Reddit comments in the corpus, by city subreddit",
                fig, key="city_chart",
                drill_hint="Click a city to filter this whole tab · click empty space to reset",
                note="Bengaluru dominates the online conversation about commutes and school buses "
                     "— it has more comments than the other three metros combined. Whatever "
                     "Bengaluru parents and riders decide about EV buses will set the tone "
                     "nationally.")

        scope = cmt if not sel_city else cmt[cmt.city == sel_city]
        scope_label = sel_city or "all four cities"

        with c_right:
            split = sentiment_split(scope)
            cats = list(split.keys())[::-1]
            fig = go.Figure()
            fig.add_bar(y=cats, x=[-split[c]["neg"] for c in cats], orientation="h",
                        marker_color=RED, name="Negative",
                        text=[f"{split[c]['neg']:.0f}%" for c in cats], textposition="inside",
                        insidetextanchor="middle", textfont=dict(color="#ffffff", size=12),
                        customdata=[split[c]["neg"] for c in cats],
                        hovertemplate="%{y} · negative: %{customdata:.0f}%<extra></extra>")
            fig.add_bar(y=cats, x=[split[c]["pos"] for c in cats], orientation="h",
                        marker_color=BLUE, name="Positive",
                        text=[f"{split[c]['pos']:.0f}%" for c in cats], textposition="inside",
                        insidetextanchor="middle", textfont=dict(color="#ffffff", size=12),
                        hovertemplate="%{y} · positive: %{x:.0f}%<extra></extra>")
            style_fig(fig, height=280, legend=True, y_grid=False)
            fig.update_layout(barmode="overlay", bargap=0.5)
            fig.update_xaxes(visible=False)
            fig.add_vline(x=0, line_color=BASELINE, line_width=1)
            n_saf = split["Safety threads"]["n"]
            n_oth = split["All other threads"]["n"]
            chart_card(
                f"How threads feel — {scope_label}",
                f"Share of comments scoring negative vs. positive (VADER) · "
                f"safety n={n_saf:,}, other n={n_oth:,}",
                fig,
                note="Comments that mention safety (accidents, rash driving, potholes, fires) "
                     "lean clearly more negative than the general commute chatter. Safety anxiety "
                     "— not price — is the emotional blocker an EV school-bus pitch has to answer "
                     "first.")

        c_left2, c_right2 = st.columns(2)
        with c_left2:
            sal = compute_salience(text_df)
            fig = go.Figure()
            fig.add_bar(x=sal.voice, y=sal.Safety, name="Safety", marker_color=BLUE,
                        text=[f"{v:.1f}" for v in sal.Safety], textposition="outside",
                        textfont=dict(color=INK2, size=11.5),
                        hovertemplate="%{x} · Safety: %{y:.2f} per 100 words<extra></extra>")
            fig.add_bar(x=sal.voice, y=sal.Economics, name="Economics", marker_color=GREEN,
                        text=[f"{v:.1f}" for v in sal.Economics], textposition="outside",
                        textfont=dict(color=INK2, size=11.5),
                        hovertemplate="%{x} · Economics: %{y:.2f} per 100 words<extra></extra>")
            style_fig(fig, height=300, legend=True)
            fig.update_layout(barmode="group")
            fig.update_yaxes(title=dict(text="mentions / 100 words",
                                        font=dict(size=11.5, color=MUTED)))
            chart_card(
                "What each voice talks about",
                "Safety vs. economics mentions per 100 words, across all stakeholder voices",
                fig,
                note="Parents and riders talk about safety far more often than money; "
                     "institutions flip that ratio and frame the market in costs and subsidies. "
                     "The two sides are having different conversations — OEM messaging has to "
                     "translate between them.")

        with c_right2:
            with st.container(border=True):
                st.markdown(f'<p class="chart-title">Loudest voices — {escape(scope_label)}</p>'
                            f'<p class="chart-sub">Most-upvoted comments in scope</p>',
                            unsafe_allow_html=True)
                mood_color = {"neg": RED, "pos": BLUE, "neu": MUTED}
                top_c = scope.sort_values("score", ascending=False).head(4)
                cards = []
                for _, r in top_c.iterrows():
                    txt = escape(str(r.text).strip().replace("\n", " ")[:230])
                    cards.append(
                        f'<div class="cmt-card"><div class="cmt-top">'
                        f'<span class="cmt-mood" style="background:'
                        f'{mood_color.get(str(r.bucket), MUTED)}"></span>'
                        f'<span class="cmt-city">{escape(str(r.city))}</span>'
                        f'<span class="cmt-score">{icon("arrow-up", 11)} '
                        f'{int(r.score)} upvotes</span></div>'
                        f'<div class="cmt-text">{txt}…</div></div>')
                st.markdown("".join(cards), unsafe_allow_html=True)
                st.markdown('<p class="chart-note"><span class="note-tag">What this shows</span>'
                            "The community posts real people actually agreed with — the dot marks "
                            "each comment's sentiment. These are the verbatim concerns an EV bus "
                            "operator would need to win over.</p>", unsafe_allow_html=True)

    # ---------------- TAB 3 · Media & reach (channel drilldown) ----------------
    with tab_media:
        vids = media_df[media_df.media_type == "video"].copy()
        ch = (vids.groupby("organization")
              .agg(n=("media_id", "size"), views=("view_count", "sum"))
              .query("n >= 2").sort_values("views").tail(8).reset_index())
        sel_org = selected_value("channel_chart")

        c_left3, c_right3 = st.columns([1.2, 1])
        with c_left3:
            names = [o if len(o) <= 22 else o[:20] + "…" for o in ch.organization]
            colors = [BLUE if (sel_org is None or o == sel_org) else BLUE_SOFT
                      for o in ch.organization]
            fig = go.Figure(go.Bar(
                y=names, x=ch.views, orientation="h", marker=dict(color=colors),
                text=[fmt_count(v) for v in ch.views], textposition="outside",
                textfont=dict(color=INK2, size=12),
                customdata=list(zip(ch.organization, ch.n)),
                hovertemplate="<b>%{customdata[0]}</b><br>%{x:,.0f} views · "
                              "%{customdata[1]} videos<extra></extra>"))
            style_fig(fig, height=320, y_grid=False)
            fig.update_xaxes(visible=False, range=[0, ch.views.max() * 1.2])
            chart_card(
                "Where the video audience is",
                "Cumulative views by channel (video corpus, ≥2 videos)",
                fig, key="channel_chart",
                drill_hint="Click a channel to see its videos on the right",
                note="A handful of channels concentrate almost all of the audience for EV bus "
                     "content. For an OEM, this is the shortlist for reviews, partnerships, or "
                     "sponsored explainers — everyone else is reach rounding error.")

        with c_right3:
            with st.container(border=True):
                panel_org = sel_org if sel_org in set(vids.organization) else None
                title = panel_org or "Top videos overall"
                st.markdown(f'<p class="chart-title">{escape(str(title))}</p>'
                            f'<p class="chart-sub">Most-viewed videos '
                            f'{"on this channel" if panel_org else "across the corpus"}</p>',
                            unsafe_allow_html=True)
                pool = vids[vids.organization == panel_org] if panel_org else vids
                tbl = (pool.sort_values("view_count", ascending=False)
                       .head(6)[["title", "view_count", "like_count"]]
                       .rename(columns={"title": "Video", "view_count": "Views",
                                        "like_count": "Likes"}))
                st.dataframe(
                    tbl, hide_index=True, width='stretch',
                    column_config={
                        "Video": st.column_config.TextColumn(width="large"),
                        "Views": st.column_config.NumberColumn(format="localized"),
                        "Likes": st.column_config.NumberColumn(format="localized"),
                    })
                st.markdown('<p class="chart-note"><span class="note-tag">What this shows</span>'
                            "The actual videos driving that reach. Titles reveal what makes EV bus "
                            "content travel — launches, reviews, and safety incidents, not spec "
                            "sheets.</p>", unsafe_allow_html=True)

    # ---------------- TAB 4 · Retrieval engine ----------------
    with tab_engine:
        c_left4, c_right4 = st.columns([3, 2])
        with c_left4:
            grid = pd.DataFrame(tuning["grid"])
            base_row = grid[grid.config.str.contains("500tok/15%", na=False)]
            base = base_row.iloc[0] if len(base_row) else grid.iloc[0]
            metrics = ["hit_rate_at5", "mrr_at10", "precision_at5"]
            labels = ["Hit@5", "MRR@10", "Precision@5"]
            fig = go.Figure()
            fig.add_bar(x=labels, y=[base[m] for m in metrics], name="Baseline",
                        marker_color=MUTED,
                        text=[f"{base[m]:.2f}" for m in metrics], textposition="outside",
                        textfont=dict(color=MUTED, size=11.5),
                        hovertemplate="Baseline · %{x}: %{y:.3f}<extra></extra>")
            fig.add_bar(x=labels, y=[best[m] for m in metrics], name="Tuned & promoted",
                        marker_color=BLUE,
                        text=[f"{best[m]:.2f}" for m in metrics], textposition="outside",
                        textfont=dict(color=INK, size=11.5),
                        hovertemplate="Tuned · %{x}: %{y:.3f}<extra></extra>")
            style_fig(fig, height=300, legend=True)
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 1.12])
            chart_card(
                "Retrieval accuracy — baseline vs. tuned",
                "Chunking / overlap grid search, evaluated on a held-out query set",
                fig,
                note=f"The gray bars are the first configuration tried; the blue bars are the "
                     f"promoted one ({best['config']}, cosine). Tuning chunk size and overlap "
                     f"lifted every metric — the engine now finds the right evidence in its top "
                     f"five results for every test query, at {best['mean_latency_ms']:.0f} ms "
                     f"per query.")

        with c_right4:
            with st.container(border=True):
                st.markdown(f'<p class="chart-title">Promoted pipeline spec</p>'
                            f'<p class="chart-sub">What answers every chat query</p>',
                            unsafe_allow_html=True)
                spec = [
                    ("Chunking", f"{best['config']} · {best['n_chunks']:,} chunks"),
                    ("Embeddings", "MiniLM-L6-v2 · cosine / FAISS"),
                    ("Sparse channel", "BM25 keyword search"),
                    ("Fusion", "Reciprocal Rank Fusion"),
                    ("Reranker", "ms-marco cross-encoder"),
                    ("Refusal gate", "cosine < 0.35 → refuse"),
                    ("Generation", "citation-locked, corpus-only"),
                ]
                st.markdown("".join(
                    f'<div class="spec-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
                    for k, v in spec), unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<p class="chart-title">Full tuning grid</p>'
                        '<p class="chart-sub">Every configuration evaluated during the '
                        'chunking grid search</p>', unsafe_allow_html=True)
            show = grid[["config", "hit_rate_at5", "mrr_at10", "precision_at5",
                         "mean_latency_ms"]].copy()
            show["promoted"] = show.config == best["config"]
            st.dataframe(
                show.rename(columns={
                    "config": "Configuration", "hit_rate_at5": "Hit@5",
                    "mrr_at10": "MRR@10", "precision_at5": "Precision@5",
                    "mean_latency_ms": "Latency (ms)", "promoted": "Promoted"}),
                hide_index=True, width='stretch',
                column_config={
                    "Hit@5": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                    "MRR@10": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                    "Precision@5": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                    "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
                    "Promoted": st.column_config.CheckboxColumn(),
                })
            st.markdown('<p class="chart-note"><span class="note-tag">What this shows</span>'
                        "Each row is one way of slicing the corpus into chunks. Longer chunks with "
                        "modest overlap won: they keep enough surrounding context for the embedding "
                        "to understand what a passage is about, without diluting it.</p>",
                        unsafe_allow_html=True)

    st.caption("Full write-up, TCO analysis, and strategic recommendations: "
               "**Strategic_Business_Report.docx**")

# ==================================================================
# CHATBOT
# ==================================================================
else:
    st.markdown("""
    <div class="hero compact">
      <h1>Ask the Corpus</h1>
      <p>Every answer is written only from the 2,852-document evidence base and shows its sources.
         Off-topic questions are refused, not improvised.</p>
    </div>
    """, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    EXAMPLES = [
        "What worries parents about school buses?",
        "Are electric buses cheaper to run than diesel?",
        "How do riders rate Cityflo and Shuttl shuttles?",
        "How are electric bus fleets being financed?",
        "How do Tata and JBM market bus safety?",
        "What is the government doing about e-buses?",
    ]

    picked = None
    if not st.session_state.chat:
        st.markdown(f"""
        <div class="welcome">
          <h4>{icon("chat", 17, BLUE)}&nbsp; What would you like to know?</h4>
          <p>Ask about safety, running costs, charging, financing, policy, or how parents and
             commuters actually feel about electric buses in India. Answers stream in live and
             every claim is backed by the evidence shown beneath it.</p>
        </div>
        """, unsafe_allow_html=True)
        for row in range(2):
            cols = st.columns(3)
            for col, ex in zip(cols, EXAMPLES[row * 3:row * 3 + 3]):
                if col.button(ex, key=f"ex_{ex}"):
                    picked = ex
    else:
        _, btn_col = st.columns([4, 1])
        if btn_col.button("Clear chat", icon=":material/delete:",
                          help="Start a fresh conversation"):
            st.session_state.chat = []
            st.rerun()

    AVATAR = {"user": ":material/person:", "assistant": ":material/directions_bus:"}

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar=AVATAR[msg["role"]]):
            st.markdown(msg["content"])
            render_answer_meta(msg)

    query = st.chat_input("Ask about EV bus safety, costs, charging, financing, policy…") or picked

    if query:
        history = st.session_state.chat.copy()
        st.session_state.chat.append({"role": "user", "content": query})
        with st.chat_message("user", avatar=AVATAR["user"]):
            st.markdown(query)

        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            with st.spinner("Searching 3,113 evidence chunks…"):
                hits, dense_max, refused = run_retrieval(query, history)

            if refused:
                st.warning(FRIENDLY_REFUSAL, icon=":material/search_off:")
                st.session_state.chat.append({"role": "assistant", "content": FRIENDLY_REFUSAL})
            else:
                sources = sources_from_hits(hits)
                confidence = confidence_of(dense_max)
                prompt = build_user_prompt(query, hits, history)

                from rag_query import llm_generate_stream
                answer, live = None, True
                try:
                    stream = llm_generate_stream(SYSTEM_PROMPT, prompt)
                    if stream is not None:
                        answer = st.write_stream(stream)
                    else:
                        live = False
                except Exception as e:
                    st.error(f"The language model call failed ({type(e).__name__}). "
                             "Showing the retrieved evidence instead.")
                    live = False

                if not live or not answer:
                    answer = ("**No live answer available** — but here is exactly the evidence "
                              "the model would have answered from. Skim the highlighted sources "
                              "below, or set an LLM API key for written answers.")
                    st.markdown(answer)
                    st.markdown(source_cards_html(sources), unsafe_allow_html=True)

                msg = {"role": "assistant", "content": answer,
                       "sources": sources, "confidence": confidence}
                if live and answer:
                    render_answer_meta(msg)
                st.session_state.chat.append(msg)

        if picked:
            st.rerun()
