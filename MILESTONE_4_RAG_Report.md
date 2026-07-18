# MILESTONE 4 — RAG Pipeline Construction: Status Report
**Deliverables:** `EV_Bus_RAG_Pipeline.ipynb` (single submission notebook, executed end-to-end) · `data/rag/` vector store
**Status:** COMPLETE. Awaiting authorization for Milestone 5.

---

## 1. The Submission Notebook (per user directive)

All pipeline code is now consolidated into **one executed Jupyter notebook — `EV_Bus_RAG_Pipeline.ipynb`** (18 cells): M1 triage summary → M2 extraction (with the Reddit adversarial log) → M3 media metadata → M4 chunking/embedding/indexing live → M5 placeholder → Bias & Limitations register.

- Network-heavy scraping cells are guarded by `RUN_SCRAPERS = False` — the notebook executes top-to-bottom in ~2 minutes against archived artifacts; flip the flag to re-run the live campaign.
- `REBUILD_INDEX = False` reloads the saved FAISS index on re-execution; the committed notebook contains the outputs of a full build.
- Standalone `scrapers/*.py` remain as artifacts and are invoked by the guarded cells.

## 2. Chunking (Blueprint §7 Level 1)

Paragraph-aware recursive chunker, **~500-token target (375 words), 15% overlap**; comments prepended with their thread title for context.

| source_type | chunks | avg words/chunk |
|---|---|---|
| comment | 2,750 | 41 |
| report_pdf | 253 | 272 |
| article | 30 | 303 |
| video (metadata summaries) | 77 | 29 |
| image_source | 6 | 39 |
| **Total** | **3,116** | from 2,849 docs |

The 3 `exclude`-tagged contamination rows never enter the index; the 2 `review` rows are indexed carrying their flag.

**Chunk metadata (citation contract):** `chunk_id, doc_id, source_name, organization, source_type, platform, modality, voice, url, thumbnail_url, title, date, segment, hypothesis, status, authenticity_score, chunk_index, n_chunks` — retrieval cites **authentic source names** ("Guidance for Electric Bus Rollout in Indian Cities — ITDP"), not raw URLs. Coarse `voice` tags (public_rider_parent / media / institutional / creator_or_oem / oem_marketing) seed the blueprint's H1 analysis lane.

## 3. Embedding & Vector Store (Blueprint §7 Level 2)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, local (no API key), L2-normalized → 384-dim.
- **Index:** FAISS **`IndexFlatIP`** — exact cosine search; approximate structures (IVF/HNSW) are unjustifiable at 3,116 vectors.
- **Hybrid lane [P1]:** BM25 (`rank_bm25`) built over the same chunks for exact-term recall (model names, "₹/km"); fused with dense scores via RRF in Milestone 5.
- **Artifacts:** `faiss_text.index` (4.7 MB, 3,116 × 384) · `chunk_metadata.parquet` · `bm25.pkl`. Index/metadata row-alignment asserted in-notebook.
- **Documented limitation:** MiniLM truncates at 256 word-pieces, so max-size chunk tails under-weight the dense vector; the BM25 full-text lane covers exact-term recall in the tails. CLIP visual lane (cross-modal) noted as the blueprint's extension over the 80 verified thumbnails + 70 OEM images.

## 4. Retrieval Smoke Test (in-notebook, dense lane)

| Query | Top-5 profile |
|---|---|
| "Why are parents hesitant about electric school buses?" | School-EV usecase article + buying guide + parent-voice Reddit comments (0.62–0.65) |
| "electric bus charging infrastructure problems in India" | ITDP rollout guidance ×3, SIAM roadmap, ITDP status report (0.72–0.80) — institutional lane |
| "employee shuttle service experience Cityflo Shuttl" | r/mumbai rider comments + **The Cityflo Experience video metadata** (0.48–0.60) |

Three different queries, three different evidence lanes, and video metadata proves retrievable alongside text — the unified corpus works as designed.

## 5. Artifacts

| File | Role |
|---|---|
| `EV_Bus_RAG_Pipeline.ipynb` | **The submission notebook** (executed, outputs embedded) |
| `data/rag/faiss_text.index` | Dense vector store (IndexFlatIP) |
| `data/rag/chunk_metadata.parquet` | Chunk-level citation metadata |
| `data/rag/bm25.pkl` | Sparse hybrid lane |
