"""
retrieval_tuning.py - Automated hyperparameter grid search for the RAG stack.

Grid:  chunk size {192w (~256 tok), 375w (~500 tok)} x overlap {10%, 20%}
       + the Milestone-4 baseline (375w, 15%)  ->  5 chunkings
       x distance metric {cosine, dot}         -> 10 configurations
Each chunking is encoded ONCE; both metrics are evaluated from the same raw
embeddings (normalized vs raw IP), halving compute honestly.

Evaluation: 12 gold queries hand-labeled against corpus doc_ids/titles.
  Hit@5 - any top-5 chunk from a gold document
  MRR@10 - 1/rank of the first gold chunk
  latency - mean per-query encode+search wall time

The winning configuration (MRR, tie-break Hit@5) is promoted: production
FAISS index, chunk metadata parquet, and BM25 lane are rebuilt under it.

Run:  python -X utf8 -m rag_modules.retrieval_tuning
"""

import json
import os
import pickle
import re
import time

import numpy as np

from .data_ingestion import BASE, build_chunks, load_corpus
from .embedding_engine import EmbeddingEngine

RAG_DIR = os.path.join(BASE, "data", "rag")
RESULTS_JSON = os.path.join(RAG_DIR, "tuning_results.json")

# ---- gold evaluation set (12 queries, labeled on doc_id and/or title) ------
GOLD_QUERIES = [
    ("cost per km of electric buses versus diesel in India",
     r"^(cesl_grand_challenge|itdp_|siam_roadmap|uitp_performance)", None),
    ("FAME scheme subsidy for electric bus procurement",
     r"^(itdp_|cesl_grand_challenge|siam_roadmap|pib_ebus)", None),
    ("parents concerns about school bus safety and accidents",
     r"^$", r"school bus"),
    ("BMTC electric bus ride quality complaints",
     r"^dh_bmtc_", r"\bBMTC\b"),
    ("employee shuttle services like Cityflo and Shuttl for office commute",
     r"^(uber_corporate|moveinsync|dh_moveinsync)", r"cityflo|shuttl"),
    ("GD Goenka private electric school bus pilot in Delhi",
     r"^(tribune_gd_goenka|tribune_delhi_flagoff|image_004)", r"GD Goenka"),
    ("JBM electric school bus safety features",
     r"^(image_002|ifc_jbm_greencell)", r"\bJBM\b"),
    ("corporate adoption of electric vehicles for employee transport",
     r"^(yourstory_corporate|moveinsync|dh_moveinsync|uber_corporate)", None),
    ("electric vehicle charging stations sitting idle in India",
     r"^(itdp_|siam_roadmap)", r"charging"),
    ("school bus fitness inspections and regulations",
     r"^telangana_today_fitness", r"fitness|school bus"),
    ("electric bus catching fire battery safety incident",
     r"^$", r"\bfire\b"),
    ("Tata Starbus and Ultra electric bus features",
     r"^image_000", r"tata (ultra|starbus|intercity)|starbus"),
]

GRID = [
    # (label,            words, overlap)
    ("256tok/10%",        192,  0.10),
    ("256tok/20%",        192,  0.20),
    ("512tok/10%",        375,  0.10),
    ("512tok/20%",        375,  0.20),
    ("M4 baseline 500tok/15%", 375, 0.15),
]


def is_gold(meta_row, doc_pat, title_pat) -> bool:
    if re.search(doc_pat, str(meta_row.doc_id)):
        return True
    if title_pat and re.search(title_pat, str(meta_row.title), re.IGNORECASE):
        return True
    return False


def evaluate(index, chunks_df, engine, metric, k=5, mrr_k=10):
    """Hit@5 saturates on this evidence-rich corpus (every gold query has many
    relevant docs), so Precision@5 - the fraction of the top-5 that is gold -
    carries the discriminative load alongside MRR@10."""
    hits, rrs, precs, lat = 0, [], [], []
    for query, doc_pat, title_pat in GOLD_QUERIES:
        t0 = time.perf_counter()
        qv = engine.query_vector(query, metric)
        _, idx = index.search(qv, mrr_k)
        lat.append(time.perf_counter() - t0)
        gold_flags = [is_gold(chunks_df.iloc[i], doc_pat, title_pat) for i in idx[0]]
        ranks = [r for r, g in enumerate(gold_flags, 1) if g]
        if ranks and ranks[0] <= k:
            hits += 1
        rrs.append(1.0 / ranks[0] if ranks else 0.0)
        precs.append(sum(gold_flags[:k]) / k)
    n = len(GOLD_QUERIES)
    return dict(hit_rate_at5=round(hits / n, 3), mrr_at10=round(float(np.mean(rrs)), 3),
                precision_at5=round(float(np.mean(precs)), 3),
                mean_latency_ms=round(1000 * float(np.mean(lat)), 1))


def promote(best, chunks_df, embeddings, engine):
    """Rebuild the production vector store under the winning configuration."""
    from rank_bm25 import BM25Okapi
    index = engine.build_index(embeddings, best["metric"])
    engine.save_index(index, os.path.join(RAG_DIR, "faiss_text.index"))
    chunks_df.to_parquet(os.path.join(RAG_DIR, "chunk_metadata.parquet"), index=False)
    tokenized = [t.lower().split() for t in chunks_df.text]
    with open(os.path.join(RAG_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump({"bm25": BM25Okapi(tokenized), "n": len(tokenized)}, f)
    print(f"PROMOTED to production: {best['config']} / {best['metric']} "
          f"({index.ntotal} vectors)")


def run_grid():
    corpus = load_corpus()
    engine = EmbeddingEngine()
    results, artifacts = [], {}
    for label, words, overlap in GRID:
        t0 = time.perf_counter()
        chunks_df = build_chunks(corpus, words, overlap)
        emb = engine.encode(chunks_df.text.tolist(), show_progress=False)
        build_s = time.perf_counter() - t0
        for metric in ("cosine", "dot"):
            index = engine.build_index(emb, metric)
            m = evaluate(index, chunks_df, engine, metric)
            rec = dict(config=label, chunk_words=words, overlap=overlap,
                       metric=metric, n_chunks=len(chunks_df),
                       index_build_s=round(build_s, 1), **m)
            results.append(rec)
            print(f"{label:26s} {metric:6s} chunks={len(chunks_df):5d} "
                  f"Hit@5={m['hit_rate_at5']:.3f} MRR@10={m['mrr_at10']:.3f} "
                  f"P@5={m['precision_at5']:.3f} lat={m['mean_latency_ms']}ms",
                  flush=True)
        artifacts[label] = (chunks_df, emb)

    best = max(results, key=lambda r: (r["mrr_at10"], r["precision_at5"], r["hit_rate_at5"]))
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({"grid": results, "best": best,
                   "gold_queries": [q for q, _, _ in GOLD_QUERIES]}, f, indent=2)
    print(f"\nBEST: {best['config']} / {best['metric']} "
          f"(MRR@10={best['mrr_at10']}, Hit@5={best['hit_rate_at5']}) -> {RESULTS_JSON}")

    chunks_df, emb = artifacts[best["config"]]
    promote(best, chunks_df, emb, engine)
    return results, best


if __name__ == "__main__":
    run_grid()
