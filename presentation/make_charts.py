"""Presentation chart suite — EV Bus Adoption in India (TIVA AI 2026).

Computes the Phase-4 style analytics (salience indexes, VADER sentiment, QRD,
engagement) directly from EV_Bus_Unified_Data.csv and renders the story charts
into presentation/charts/. Every title is a claim; every chart carries n + source.

Run:  python presentation/make_charts.py
"""
import os
import re

import matplotlib.pyplot as plt
import matplotlib.font_manager  # noqa: F401
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

# ---- palette (validated defaults, light mode) -------------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
BLUE, GREEN, MAGENTA, AMBER, RED = "#2a78d6", "#008300", "#e87ba4", "#eda100", "#e34948"
NEUTRAL_MID = "#f0efec"
# semantic colours (blueprint §10): safety=amber, economics=blue, green/ESG=green,
# reliability=red — held constant across every chart in the deck
SEM = {"Safety": AMBER, "Economics": BLUE, "Green/ESG": GREEN, "Reliability": RED}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "font.family": "Segoe UI", "font.size": 10.5,
    "axes.edgecolor": BASELINE, "axes.linewidth": 1.0,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
})


def style_ax(ax, xgrid=False, ygrid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def headline(fig, claim, sub):
    fig.text(0.02, 0.965, claim, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=10, color=INK2, va="top")


def footer(fig, note):
    fig.text(0.02, 0.015, note, fontsize=8.5, color=MUTED, va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches=None)
    plt.close(fig)
    print("wrote", name)


# ---- load & tag -------------------------------------------------------------
df = pd.read_csv(os.path.join(BASE, "EV_Bus_Unified_Data.csv"), encoding="utf-8-sig")
df = df[df.status != "exclude"].copy()
VOICE = {"comment": "Riders & parents", "article": "Media", "report_pdf": "Institutions",
         "video": "Creators & OEMs", "image_source": "Creators & OEMs"}
df["voice"] = df.doc_type.map(VOICE)

LEX = {
    "Safety":      r"\b(safety|safe|unsafe|accident|crash|cctv|gps|panic\s*button|speed\s*governor|seat\s*belt|seatbelt|overspeed|rash|reckless|pothole|brake|fire|danger|dangerous|injur\w+)\b",
    "Economics":   r"\b(cost|price|fare|tco|capex|subsid\w+|per[-\s]?km|payback|fuel|savings?|lease|financ\w+|budget|lakh|crore|cheap\w*|expensive|afford\w*)\b|₹",
    "Green/ESG":   r"\b(emission\w*|pollut\w+|carbon|esg|sustainab\w+|green|clean\s+energy|climate)\b",
    "Reliability": r"\b(range|charg\w+|breakdown|broke\s+down|stranded|battery|batteries|mid[-\s]?route|delay\w*|late|frequency|waiting|unreliable|reliab\w+)\b",
}
QRD_PAT = (r"\b\d[\d,.]*\s*(km|kwh|kw|%|seats?|lakh|crore|rs\.?|rupees|minutes?|mins?"
           r"|hours?|hrs?|years?|yrs?|buses|kms)\b|₹\s*\d|\b\d{2,}\b")


def per100(texts, pat):
    t = " ".join(map(str, texts)).lower()
    return 100 * len(re.findall(pat, t)) / max(len(t.split()), 1)


# =============================================================================
# 1 · Corpus composition — documents vs words by voice
# =============================================================================
comp = df.groupby("voice").agg(docs=("doc_id", "size"), words=("word_count", "sum"))
order = ["Riders & parents", "Institutions", "Media", "Creators & OEMs"]
comp = comp.loc[order]

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
fig.subplots_adjust(top=0.80, bottom=0.10, left=0.17, right=0.97, wspace=0.42)
for ax, col, ttl in [(axes[0], "docs", "Documents"), (axes[1], "words", "Words")]:
    vals = comp[col][::-1]
    ax.barh(vals.index, vals.values, color=BLUE, height=0.55)
    for i, v in enumerate(vals.values):
        ax.text(v + max(vals.values) * 0.015, i, f"{v:,.0f}", va="center",
                fontsize=9.5, color=INK2)
    ax.set_title(ttl, fontsize=10.5, color=INK2, loc="left")
    ax.set_xlim(0, max(vals.values) * 1.18)
    ax.set_xticks([])
    style_ax(ax, ygrid=False)
headline(fig, "Parents speak in fragments; institutions write books",
         "2,747 rider/parent comments carry 78k words — six institutional reports alone carry 66k")
footer(fig, "n = 2,849 indexable assets · source: EV_Bus_Unified_Data.csv (Reddit, ITDP/SIAM/CESL/UITP/UNCRD PDFs, news, YouTube, OEM pages)")
save(fig, "01_corpus_voices.png")

# =============================================================================
# 2 · Salience indexes by voice — the H1 divergence
# =============================================================================
voices = ["Riders & parents", "Media", "Institutions"]
sal = {v: {k: per100(df[df.voice == v].text, pat) for k, pat in LEX.items()} for v in voices}

fig, ax = plt.subplots(figsize=(9.6, 4.8))
fig.subplots_adjust(top=0.76, bottom=0.14, left=0.07, right=0.97)
idx = np.arange(len(voices))
w = 0.19
for j, (k, colr) in enumerate(SEM.items()):
    vals = [sal[v][k] for v in voices]
    bars = ax.bar(idx + (j - 1.5) * w, vals, width=w * 0.92, color=colr, label=k)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}",
                ha="center", fontsize=8.8, color=INK2)
ax.set_xticks(idx, voices, fontsize=11, color=INK)
ax.set_ylabel("mentions per 100 words")
ax.legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0, 1.08), fontsize=9.5)
style_ax(ax)
headline(fig, "Institutions talk money; the safety conversation lives on the street",
         "Economics terms: 2.26/100 words in institutional reports vs 0.04 for safety — riders & parents weigh safety ~25× heavier relative to economics")
footer(fig, "Salience = lexicon hits per 100 words (SSI/ESI/GSI/RAI, blueprint §4 L7) · n = 78k words (riders), 66k (institutions), 8k (media)")
save(fig, "02_salience_divergence.png")

# =============================================================================
# 3 · Quantitative Reasoning Density — who argues in numbers
# =============================================================================
qrd = {v: per100(df[df.voice == v].text, QRD_PAT) for v in voices}
fig, ax = plt.subplots(figsize=(9.6, 3.9))
fig.subplots_adjust(top=0.74, bottom=0.10, left=0.17, right=0.95)
vals = pd.Series(qrd).sort_values()
ax.barh(vals.index, vals.values, color=BLUE, height=0.5)
for i, v in enumerate(vals.values):
    ax.text(v + 0.06, i, f"{v:.1f}", va="center", fontsize=10.5, color=INK2, fontweight="bold")
ax.set_xlim(0, vals.max() * 1.15)
ax.set_xticks([])
ax.set_yticklabels(vals.index, fontsize=11)
style_ax(ax, ygrid=False)
headline(fig, "Institutions argue in numbers; parents argue in stories",
         "Quantitative Reasoning Density (numerals + units per 100 words): 3.7 institutional vs 1.1 rider/parent — a 3.3× gap")
footer(fig, "QRD = numerals with units (km, kWh, Rs, lakh, %, seats…) per 100 words · creator/OEM video metadata excluded (auto-generated counts would inflate it)")
save(fig, "03_qrd_gap.png")

# =============================================================================
# 4 · Sentiment collapses when safety comes up
# =============================================================================
an = SentimentIntensityAnalyzer()
c = df[df.doc_type == "comment"].copy()
c["compound"] = c.text.astype(str).map(lambda t: an.polarity_scores(t)["compound"])
c["bucket"] = pd.cut(c.compound, [-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"])
c["mentions_safety"] = c.text.astype(str).str.lower().str.contains(LEX["Safety"], regex=True)

groups = [("Safety-related comments", c[c.mentions_safety]), ("All other comments", c[~c.mentions_safety])]
fig, ax = plt.subplots(figsize=(9.6, 3.7))
fig.subplots_adjust(top=0.72, bottom=0.16, left=0.05, right=0.95)
for i, (label, g) in enumerate(groups[::-1]):
    shares = g.bucket.value_counts(normalize=True)
    neg, neu, pos = shares.get("Negative", 0) * 100, shares.get("Neutral", 0) * 100, shares.get("Positive", 0) * 100
    ax.barh(i, -neg, color=RED, height=0.5)
    ax.barh(i, neu, left=0, color=NEUTRAL_MID, height=0.5,
            edgecolor="#0b0b0b", linewidth=0.3, alpha=0.9)
    ax.barh(i, pos, left=neu, color=BLUE, height=0.5)
    ax.text(-neg - 2, i, f"{neg:.0f}%", ha="right", va="center", color=RED, fontweight="bold")
    ax.text(neu + pos + 2, i, f"{pos:.0f}%", ha="left", va="center", color=BLUE, fontweight="bold")
    ax.text(-88, i, f"{label}\n(n = {len(g):,})", ha="left", va="center", fontsize=10.5, color=INK)
ax.axvline(0, color=BASELINE, linewidth=1)
ax.set_xlim(-90, 75)
ax.set_yticks([])
ax.set_xticks([])
style_ax(ax, ygrid=False)
fig.text(0.30, 0.155, "← negative share", fontsize=8.5, color=RED)
fig.text(0.78, 0.155, "positive share →", fontsize=8.5, color=BLUE)
headline(fig, "When safety comes up, positivity vanishes",
         "47% of safety-related comments are negative vs 28% of everything else — VADER compound on 2,747 Reddit comments")
footer(fig, "VADER sentiment, thresholds ±0.05 · safety = SSI lexicon match · source: r/bangalore, r/delhi, r/mumbai, r/hyderabad, Jul 2024–Jul 2026")
save(fig, "04_safety_sentiment.png")

# =============================================================================
# 5 · Sentiment by city — the operating-experience gradient
# =============================================================================
c["city"] = c.organization.str.extract(r"r/(\w+)")[0].str.capitalize()
city_order = (c.groupby("city").bucket
               .value_counts(normalize=True).unstack()["Negative"].sort_values(ascending=False).index)
fig, ax = plt.subplots(figsize=(9.6, 4.3))
fig.subplots_adjust(top=0.76, bottom=0.12, left=0.20, right=0.95)
for i, city in enumerate(list(city_order)[::-1]):
    g = c[c.city == city]
    shares = g.bucket.value_counts(normalize=True)
    neg, neu, pos = shares.get("Negative", 0) * 100, shares.get("Neutral", 0) * 100, shares.get("Positive", 0) * 100
    ax.barh(i, -neg, color=RED, height=0.52)
    ax.barh(i, neu, left=0, color=NEUTRAL_MID, height=0.52,
            edgecolor="#0b0b0b", linewidth=0.3, alpha=0.9)
    ax.barh(i, pos, left=neu, color=BLUE, height=0.52)
    ax.text(-neg - 2, i, f"{neg:.0f}%", ha="right", va="center", color=RED, fontweight="bold", fontsize=9.5)
    ax.text(neu + pos + 2, i, f"{pos:.0f}%", ha="left", va="center", color=BLUE, fontweight="bold", fontsize=9.5)
    ax.text(-56, i, f"{city}\n(n = {len(g):,})", ha="left", va="center", fontsize=10.5, color=INK)
ax.axvline(0, color=BASELINE, linewidth=1)
ax.set_xlim(-58, 92)
ax.set_yticks([])
ax.set_xticks([])
style_ax(ax, ygrid=False)
headline(fig, "Commuter mood is not uniform: Delhi grumbles, Hyderabad cheers",
         "Negative share ranges from 19% (Hyderabad) to 33% (Delhi) — rollout experience differs city by city")
footer(fig, "VADER on 2,747 comments from city subreddits · negative left, positive right, neutral at centre · Jul 2024–Jul 2026")
save(fig, "05_city_sentiment.png")

# =============================================================================
# 6 · Video engagement — reach does not buy engagement
# =============================================================================
v = df[df.doc_type == "video"].copy()
v["eng_rate"] = 100 * (v.like_count.fillna(0) + v.comment_count.fillna(0)) / v.view_count
seg_map = {"Employee Transport": ("Employee transport", BLUE),
           "General": ("Independent EV creators", GREEN),
           "General/OEM": ("OEM channels", MAGENTA)}
rho = v[["view_count", "eng_rate"]].corr(method="spearman").iloc[0, 1]

fig, ax = plt.subplots(figsize=(9.6, 5.0))
fig.subplots_adjust(top=0.78, bottom=0.13, left=0.09, right=0.97)
for seg, (label, colr) in seg_map.items():
    g = v[v.segment == seg]
    ax.scatter(g.view_count, g.eng_rate, s=46, color=colr, alpha=0.85,
               edgecolors=SURFACE, linewidths=1.2, label=f"{label} (n={len(g)})")
med = v.groupby(v.segment.map(lambda s: seg_map[s][0])).eng_rate.median()
ax.set_xscale("log")
ax.set_xlabel("views (log scale)")
ax.set_ylabel("engagement rate  (likes + comments) / views, %")
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
style_ax(ax)
headline(fig, f"Reach doesn't buy engagement (ρ = {rho:.2f}) — and OEM channels engage least",
         f"Median engagement: independent EV creators {med['Independent EV creators']:.1f}% · employee transport {med['Employee transport']:.1f}% · OEM channels {med['OEM channels']:.1f}%")
footer(fig, "n = 77 YouTube videos (3 contamination rows excluded) · Spearman rank correlation · engagement metadata from source sheet snapshot")
save(fig, "06_engagement_scatter.png")

# =============================================================================
# 7 · Discourse timeline — bursty and incident-driven
# =============================================================================
c["month"] = pd.to_datetime(c.published_date, errors="coerce", utc=True,
                            format="mixed").dt.tz_localize(None).dt.to_period("M")
months = pd.period_range(c.month.min(), c.month.max(), freq="M")
mv = (c.groupby(["month", "city"]).size().unstack(fill_value=0)
        .reindex(months, fill_value=0))
city_colors = {"Bangalore": BLUE, "Delhi": GREEN, "Mumbai": MAGENTA, "Hyderabad": AMBER}

fig, ax = plt.subplots(figsize=(9.6, 4.6))
fig.subplots_adjust(top=0.78, bottom=0.16, left=0.08, right=0.97)
bottom = np.zeros(len(mv))
for city, colr in city_colors.items():
    ax.bar(range(len(mv)), mv[city].values, bottom=bottom, color=colr,
           width=0.7, label=city, edgecolor=SURFACE, linewidth=0.6)
    bottom += mv[city].values
ticks = [i for i, m in enumerate(mv.index) if m.month in (1, 7)]
ax.set_xticks(ticks, [str(mv.index[i]) for i in ticks], fontsize=9)
ax.set_ylabel("comments per month")
ax.legend(frameon=False, ncols=4, loc="upper left", bbox_to_anchor=(0, 1.10), fontsize=9.5)
style_ax(ax)
peak = mv.sum(axis=1).idxmax()
ax.annotate("school-bus incident &\nBMTC threads spike",
            xy=(list(mv.index).index(peak), mv.sum(axis=1).max()),
            xytext=(list(mv.index).index(peak) - 7.5, mv.sum(axis=1).max() * 0.92),
            fontsize=9, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
headline(fig, "The conversation arrives in bursts — incidents and launches set the agenda",
         "Monthly Reddit comment volume across four metros; spikes follow accident threads and fleet-launch news, not a smooth trend")
footer(fig, "n = 2,747 comments, Jul 2024–Jul 2026 · thread-level scraping means volume reflects when large threads occurred (see bias register)")
save(fig, "07_discourse_timeline.png")

# =============================================================================
# 8 · RAG tuning — retrieval quality saturates
# =============================================================================
import json
with open(os.path.join(BASE, "data", "rag", "tuning_results.json"), encoding="utf-8") as f:
    tune = json.load(f)
grid = pd.DataFrame(tune["grid"])
grid = grid[grid.metric == "cosine"]

fig, ax = plt.subplots(figsize=(9.6, 4.2))
fig.subplots_adjust(top=0.76, bottom=0.14, left=0.24, right=0.90)
labels = grid.config.tolist()
vals = grid.mrr_at10.tolist()
lat = grid.mean_latency_ms.tolist()
bars = ax.barh(range(len(vals))[::-1], vals, color=BLUE, height=0.5)
for i, (v_, l_) in enumerate(zip(vals, lat)):
    y = len(vals) - 1 - i
    ax.text(v_ - 0.015, y, f"MRR {v_:.2f}", ha="right", va="center",
            color=SURFACE, fontsize=9.5, fontweight="bold")
    ax.text(1.015, y, f"{l_:.0f} ms", ha="left", va="center", color=INK2, fontsize=9.5)
ax.set_yticks(range(len(vals))[::-1], labels, fontsize=10)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
style_ax(ax, ygrid=False, xgrid=True)
headline(fig, "Retrieval is not the bottleneck: 4 of 5 configs score a perfect MRR",
         "Chunking grid over 12 gold queries — every config hits Hit@5 = 1.0; latency stays under 17 ms per query")
footer(fig, "MRR@10 on 12 hand-authored gold queries · FAISS IndexFlatIP + MiniLM embeddings · right column: mean query latency")
save(fig, "08_rag_tuning.png")

print("\nAll charts written to", OUT)
