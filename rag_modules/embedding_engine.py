"""
embedding_engine.py - Dense embedding + FAISS index construction.

Wraps sentence-transformers/all-MiniLM-L6-v2 with explicit exception handling
(model-load timeouts, empty payloads) and supports both distance metrics under
test in the hyperparameter grid:
  - cosine:      L2-normalized vectors -> IndexFlatIP (bounded scores [-1, 1])
  - dot product: raw vectors          -> IndexFlatIP (unbounded, length-biased)
"""

import os

import numpy as np

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngineError(Exception):
    """Raised on model-load failure or empty/invalid embedding payloads."""


class EmbeddingEngine:
    def __init__(self, model_name: str = EMB_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception as e:  # network timeout, missing cache, bad env
            raise EmbeddingEngineError(
                f"Could not load embedding model '{model_name}': "
                f"{type(e).__name__}: {e}. If offline, warm the HF cache first."
            ) from e

    def encode(self, texts, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """Encode WITHOUT normalization - metric choice is applied at index time."""
        if not len(texts):
            raise EmbeddingEngineError("encode() received an empty text payload")
        emb = self.model.encode(list(texts), batch_size=batch_size,
                                show_progress_bar=show_progress,
                                normalize_embeddings=False)
        return np.asarray(emb, dtype=np.float32)

    @staticmethod
    def build_index(embeddings: np.ndarray, metric: str):
        """metric: 'cosine' (normalize -> IP) or 'dot' (raw IP)."""
        import faiss
        if metric not in ("cosine", "dot"):
            raise EmbeddingEngineError(f"Unknown metric '{metric}'")
        vecs = embeddings.copy()
        if metric == "cosine":
            faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return index

    def query_vector(self, query: str, metric: str) -> np.ndarray:
        qv = self.encode([query])
        if metric == "cosine":
            import faiss
            faiss.normalize_L2(qv)
        return qv

    @staticmethod
    def save_index(index, path: str):
        import faiss
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(index, path)

    @staticmethod
    def load_index(path: str):
        import faiss
        if not os.path.exists(path):
            raise EmbeddingEngineError(f"Index not found: {path}")
        return faiss.read_index(path)
