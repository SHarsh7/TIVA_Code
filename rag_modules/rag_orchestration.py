"""
rag_orchestration.py - Hybrid retrieval + reranking + grounded generation.

Pipeline: query -> dense (MiniLM/FAISS) top-20 + sparse (BM25) top-20
       -> Reciprocal Rank Fusion -> cross-encoder rerank -> refusal gate
       -> grounded prompt with inline [chunk_id] citations -> pluggable LLM.

Explicit exception handling: missing artifacts raise OrchestrationError with
remediation hints; the reranker degrades gracefully to fused order; LLM calls
are wrapped with timeout/rate-limit retry.
"""

import os
import pickle
import time

import numpy as np
import pandas as pd

from .data_ingestion import BASE
from .embedding_engine import EmbeddingEngine, EmbeddingEngineError

RAG_DIR = os.path.join(BASE, "data", "rag")
REFUSAL_GATE = 0.35
REFUSAL_TEXT = "Insufficient evidence in corpus."

SYSTEM_PROMPT = """You are the strategy analyst for an automotive OEM's Strategic AI Taskforce
evaluating EV bus adoption in India's school and employee transport segments.

Be exploratory, active, and highly analytical: probe tensions between sources,
contrast stakeholder voices (parents/riders vs institutions vs OEM marketing),
and surface implications for the 3-year strategy - not just summaries.

HARD RULES
1. Use ONLY the evidence in CONTEXT. No outside knowledge, however confident.
2. Cite the chunk_id in square brackets immediately after every claim it supports.
3. Name sources by their authentic names (e.g. "SIAM e-bus roadmap"), never URLs.
4. If the context cannot support an answer, reply exactly: """ + REFUSAL_TEXT + """
5. End with SOURCES: a deduplicated list of source_name (voice) cited."""


class OrchestrationError(Exception):
    """Raised when the retrieval stack cannot be assembled."""


class RAGOrchestrator:
    def __init__(self, use_reranker: bool = True):
        try:
            self.engine = EmbeddingEngine()
        except EmbeddingEngineError as e:
            raise OrchestrationError(str(e)) from e
        try:
            self.index = self.engine.load_index(os.path.join(RAG_DIR, "faiss_text.index"))
            self.chunks = pd.read_parquet(os.path.join(RAG_DIR, "chunk_metadata.parquet"))
            with open(os.path.join(RAG_DIR, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)["bm25"]
        except (EmbeddingEngineError, FileNotFoundError, KeyError) as e:
            raise OrchestrationError(
                f"Vector store incomplete ({e}); run rag_modules.retrieval_tuning "
                "or notebook section 4 to build it.") from e
        if self.index.ntotal != len(self.chunks):
            raise OrchestrationError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs "
                f"{len(self.chunks)} chunk records - rebuild the store.")
        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"[warn] reranker unavailable ({type(e).__name__}) - fused order")

    # ---- Level 3: hybrid retrieval -------------------------------------
    def retrieve(self, query: str, k: int = 5, pool: int = 20) -> pd.DataFrame:
        qv = self.engine.query_vector(query, "cosine")
        d_scores, d_idx = self.index.search(qv, pool)
        b_idx = np.argsort(self.bm25.get_scores(query.lower().split()))[::-1][:pool]
        rrf = {}
        for rank, i in enumerate(d_idx[0]):
            rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
        for rank, i in enumerate(b_idx):
            rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
        fused = sorted(rrf, key=rrf.get, reverse=True)[:pool]
        if self.reranker is not None:
            ce = self.reranker.predict([(query, self.chunks.iloc[i].text) for i in fused])
            fused = [fused[j] for j in np.argsort(ce)[::-1]]
        hits = self.chunks.iloc[fused[:k]].copy()
        hits["dense_max"] = float(d_scores[0].max())
        return hits

    # ---- Level 4: grounded generation ----------------------------------
    @staticmethod
    def build_context_pack(hits: pd.DataFrame) -> str:
        blocks = [f"[{h.chunk_id}] source: {h.source_name} | org: {h.organization}"
                  f" | voice: {h.voice} | type: {h.source_type}\n{h.text}"
                  for _, h in hits.iterrows()]
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def llm_generate(system: str, user: str, retries: int = 3):
        """Provider auto-detect with timeout/rate-limit retry; None if no key."""
        for attempt in range(1, retries + 1):
            try:
                if os.environ.get("ANTHROPIC_API_KEY"):
                    import anthropic
                    r = anthropic.Anthropic().messages.create(
                        model="claude-sonnet-5", max_tokens=900, system=system,
                        messages=[{"role": "user", "content": user}])
                    return r.content[0].text
                if os.environ.get("OPENAI_API_KEY"):
                    import openai
                    r = openai.OpenAI().chat.completions.create(
                        model="gpt-4o-mini", max_tokens=900,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}])
                    return r.choices[0].message.content
                if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                    import google.generativeai as genai
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY")
                                    or os.environ.get("GOOGLE_API_KEY"))
                    return genai.GenerativeModel(
                        "gemini-1.5-flash", system_instruction=system
                    ).generate_content(user).text
                return None            # no key configured -> context-pack mode
            except Exception as e:     # API timeout / rate limit / transient
                if attempt == retries:
                    raise OrchestrationError(
                        f"LLM call failed after {retries} attempts: "
                        f"{type(e).__name__}: {e}") from e
                time.sleep(5 * attempt)

    def answer(self, query: str, k: int = 5) -> dict:
        hits = self.retrieve(query, k=k)
        if hits.dense_max.iloc[0] < REFUSAL_GATE:
            return {"query": query, "answer": REFUSAL_TEXT, "mode": "refusal-gate",
                    "citations": [], "sources": []}
        prompt = f"CONTEXT:\n{self.build_context_pack(hits)}\n\nQUESTION: {query}"
        answer = self.llm_generate(SYSTEM_PROMPT, prompt)
        return {"query": query, "answer": answer,
                "mode": "llm" if answer else "context-pack (no API key)",
                "citations": hits.chunk_id.tolist(),
                "sources": sorted(set(hits.source_name))}
