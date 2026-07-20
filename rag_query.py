"""
MILESTONE 5 - Final RAG query interface (CLI).

Same pipeline as notebook §5: dense (MiniLM/FAISS) + BM25 -> Reciprocal Rank
Fusion -> cross-encoder rerank -> refusal gate -> grounded generation with
inline [chunk_id] citations and authentic source names.

With no LLM API key set, runs in context-pack mode: prints the reranked
evidence and the fully assembled grounded prompt. Set ANTHROPIC_API_KEY /
OPENAI_API_KEY / GEMINI_API_KEY to enable live generation, no code changes.

Usage:
  python rag_query.py "How do parents feel about electric school buses?" [--k 5]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE, "data", "rag")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
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


def load_stack():
    import faiss
    from sentence_transformers import SentenceTransformer
    index = faiss.read_index(os.path.join(RAG_DIR, "faiss_text.index"))
    chunks_df = pd.read_parquet(os.path.join(RAG_DIR, "chunk_metadata.parquet"))
    with open(os.path.join(RAG_DIR, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)["bm25"]
    model = SentenceTransformer(EMB_MODEL)
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        reranker = None
    return index, chunks_df, bm25, model, reranker


def retrieve(query, index, chunks_df, bm25, model, reranker, k=5, pool=20):
    qv = model.encode([query], normalize_embeddings=True).astype(np.float32)
    d_scores, d_idx = index.search(qv, pool)
    b_idx = np.argsort(bm25.get_scores(query.lower().split()))[::-1][:pool]
    rrf = {}
    for rank, i in enumerate(d_idx[0]):
        rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
    for rank, i in enumerate(b_idx):
        rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + rank)
    fused = sorted(rrf, key=rrf.get, reverse=True)[:pool]
    if reranker is not None:
        ce = reranker.predict([(query, chunks_df.iloc[i].text) for i in fused])
        fused = [fused[j] for j in np.argsort(ce)[::-1]]
    hits = chunks_df.iloc[fused[:k]].copy()
    hits["dense_max"] = float(d_scores[0].max())
    return hits


def build_context_pack(hits):
    blocks = [f"[{h.chunk_id}] source: {h.source_name} | org: {h.organization}"
              f" | voice: {h.voice} | type: {h.source_type}\n{h.text}"
              for _, h in hits.iterrows()]
    return "\n\n---\n\n".join(blocks)


def llm_generate(system, user, max_tokens=1024, temperature=0.3):
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        r = Groq().chat.completions.create(
            model=model_name, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5", max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}])
        return r.content[0].text
    if os.environ.get("OPENAI_API_KEY"):
        import openai
        r = openai.OpenAI().chat.completions.create(
            model="gpt-4o-mini", max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-1.5-flash",
                                     system_instruction=system).generate_content(user).text
    return None


def llm_generate_stream(system, user, max_tokens=1024, temperature=0.3):
    """Like llm_generate, but returns a generator that yields the answer
    incrementally (for st.write_stream). Returns None if no API key is set."""
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        stream = Groq().chat.completions.create(
            model=model_name, max_tokens=max_tokens, temperature=temperature, stream=True,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        def gen():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return gen()
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        def gen():
            with client.messages.stream(
                    model="claude-sonnet-5", max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}]) as s:
                for text in s.text_stream:
                    yield text
        return gen()
    if os.environ.get("OPENAI_API_KEY"):
        import openai
        stream = openai.OpenAI().chat.completions.create(
            model="gpt-4o-mini", max_tokens=max_tokens, stream=True,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        def gen():
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        return gen()
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        resp = genai.GenerativeModel("gemini-1.5-flash",
                                     system_instruction=system).generate_content(user, stream=True)
        def gen():
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        return gen()
    return None


def main():
    ap = argparse.ArgumentParser(description="EV-bus corpus RAG query")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    index, chunks_df, bm25, model, reranker = load_stack()
    hits = retrieve(args.query, index, chunks_df, bm25, model, reranker, k=args.k)

    if hits.dense_max.iloc[0] < REFUSAL_GATE:
        print(f"{REFUSAL_TEXT} (max dense score {hits.dense_max.iloc[0]:.3f} < {REFUSAL_GATE})")
        sys.exit(0)

    print("=" * 90)
    print("TOP EVIDENCE (hybrid + reranked)")
    for _, h in hits.iterrows():
        print(f"  [{h.chunk_id}] {h.source_name} ({h.voice})")
    print("=" * 90)

    prompt = f"CONTEXT:\n{build_context_pack(hits)}\n\nQUESTION: {args.query}"
    answer = llm_generate(SYSTEM_PROMPT, prompt)
    if answer:
        print(answer)
    else:
        print("[context-pack mode - no LLM API key set]\n")
        print(prompt[:3000])
        print("\n... [prompt truncated for display; set an API key for live generation]")


if __name__ == "__main__":
    main()
