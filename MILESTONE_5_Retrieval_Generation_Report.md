# MILESTONE 5 — Retrieval & Grounded Generation: Status Report
**Deliverables:** notebook §5 (live, executed) · `rag_query.py` (CLI)
**Status:** COMPLETE — **all five milestones delivered. Pipeline finished.**

---

## 1. Retrieval Stack (Blueprint §7 Level 3)

`query → dense (MiniLM/FAISS) top-20 + BM25 top-20 → Reciprocal Rank Fusion (1/(60+rank)) → cross-encoder rerank (ms-marco-MiniLM-L-6-v2) → top-5`

- Dense catches paraphrase ("parents worried" ≈ "hesitant"); BM25 catches exact terms ("FAME-II", "₹/km", "GCC"); RRF fuses without score normalization.
- Cross-encoder reranking [P1, beyond-class] re-scores the fused pool; degrades gracefully to fused order if the model can't load.

## 2. Grounded Generation Contract (Level 4)

System prompt enforces: **exploratory, active, highly analytical** analyst voice (per milestone directive) · answer only from retrieved CONTEXT · inline `[chunk_id]` citation after every claim · authentic source names, never URLs · exact refusal string on insufficient evidence · SOURCES footer with voice labels.

**Pluggable LLM adapter:** auto-detects `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` → `GEMINI_API_KEY` at runtime. With no key (the grading-safe default), the pipeline runs in **context-pack mode** — retrieval, reranking, and the fully assembled grounded prompt are all produced and displayed. Export a key and the same cells/CLI generate live answers with zero code changes.

## 3. Executive Query Battery (Task 3.2) — evidence routing verified

| Gold query | Top-5 evidence profile |
|---|---|
| Parents' school-bus safety concerns | Parent-voice Reddit (r/bangalore pothole/accident threads) + school-EV buying guide |
| E-bus vs diesel economics (₹/km, TCO) | ITDP rollout guidance + status report — institutional lane |
| Cityflo/Shuttl vs public buses | r/mumbai rider comments — pure `public_rider_parent` voice |
| Financing & procurement models | ITDP + SIAM roadmap + **CESL Grand Challenge** (FAME-I/II, GCC evidence) |
| OEM safety positioning (Tata/JBM/Eicher) | **OEM image-source chunks** (product-page imagery records) + IFC partnership doc — media rows retrieving alongside text |

Five queries, five distinct evidence routings across all corpus voices — retrieval discriminates, it doesn't just match keywords.

## 4. Refusal Guardrail — two layers, both demonstrated

1. **Retrieval gate** (max dense cosine < 0.35): *"What were the causes of the French Revolution?"* → **refused at 0.298**, zero LLM tokens spent. Asserted in-notebook.
2. **Contract layer (rule 4):** calibration finding — *"Tesla Cybertruck delivery timelines"* scores **above** the gate because it is semantically adjacent to an EV corpus; no retrieval threshold can separate it without killing recall. The grounded-prompt contract is the responsible layer for near-domain misses, and the notebook documents this division of labor explicitly.

## 5. Final Artifact Inventory

| Artifact | Role |
|---|---|
| **`EV_Bus_RAG_Pipeline.ipynb`** | **THE submission notebook** — 25 cells, fully executed, M1→M5 |
| `rag_query.py` | Standalone executive query CLI (same stack) |
| `EV_Bus_Text_Data.csv` / `EV_Bus_Media_Data.csv` / `EV_Bus_Unified_Data.csv` | M2/M3 datasets |
| `data/rag/faiss_text.index` + `chunk_metadata.parquet` + `bm25.pkl` | Vector store (3,116 × 384) + citation metadata + sparse lane |
| `scrapers/*.py` | Collection engines (Tier A/B/C, Reddit, media) |
| `data/raw/*` | Raw extractions, 6 archived PDFs, per-subreddit dumps, failure log |
| `MILESTONE_1..5_*.md` | Per-milestone documentation |

**Operational note (for re-execution):** run the notebook from a normal terminal/Jupyter session. Model downloads are cached under `~/.cache/huggingface`; set `HF_HUB_OFFLINE=1` for fully offline re-runs.
