"""
data_ingestion.py - Corpus loading, validation, and configurable chunking.

Upgrades over the classroom baseline (Refrence Notebooks/EA26 Analysis.ipynb
processes media ad hoc, no retrieval corpus): a validated unified schema in
which every multimedia element is bound to its parental text segment
(video -> thumbnail_url + engagement metadata; image -> harvested OEM URLs),
plus a recursive, paragraph-aware chunker with strict size/overlap control
used by the hyperparameter grid in retrieval_tuning.py.
"""

import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_CSV = os.path.join(BASE, "EV_Bus_Unified_Data.csv")

REQUIRED_COLUMNS = {
    "doc_id", "source_name", "organization", "doc_type", "platform", "url",
    "title", "text", "status", "thumbnail_url", "segment", "hypothesis",
    "authenticity_score", "published_date",
}

VOICE_MAP = {"comment": "public_rider_parent", "article": "media",
             "report_pdf": "institutional", "dataset": "institutional",
             "video": "creator_or_oem", "image_source": "oem_marketing"}


class PayloadValidationError(Exception):
    """Raised when the corpus payload is missing mandatory metadata."""


def load_corpus(path: str = UNIFIED_CSV, drop_excluded: bool = True) -> pd.DataFrame:
    """Load the unified corpus with explicit payload validation."""
    if not os.path.exists(path):
        raise PayloadValidationError(f"Unified corpus not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise PayloadValidationError(f"Corpus missing mandatory columns: {sorted(missing)}")
    null_text = df["text"].isna().sum()
    if null_text:
        raise PayloadValidationError(f"{null_text} rows have null text payloads")
    if drop_excluded:
        df = df[df["status"] != "exclude"].copy()
    return df


def chunk_words(text: str, size: int, overlap_ratio: float) -> list:
    """Recursive character/paragraph splitter with strict overlap control.

    Packs whole paragraphs up to `size` words; oversize paragraphs fall back
    to sliding word windows with stride = size * (1 - overlap_ratio).
    """
    stride = max(1, int(size * (1 - overlap_ratio)))
    paras = [p.strip() for p in re.split(r"\n\s*\n", str(text)) if p.strip()]
    pieces, buf, n = [], [], 0
    for p in paras:
        w = p.split()
        if len(w) > size:
            if buf:
                pieces.append(" ".join(buf)); buf, n = [], 0
            for s in range(0, len(w), stride):
                pieces.append(" ".join(w[s:s + size]))
                if s + size >= len(w):
                    break
        elif n + len(w) > size:
            pieces.append(" ".join(buf)); buf, n = w, len(w)
        else:
            buf += w; n += len(w)
    if buf:
        pieces.append(" ".join(buf))
    return pieces or [str(text)]


def build_chunks(df: pd.DataFrame, size: int, overlap_ratio: float) -> pd.DataFrame:
    """Chunk the corpus under one (size, overlap) configuration.

    Every chunk inherits the unified metadata schema, binding multimedia
    elements (thumbnail_url, engagement counts) to their parental segment.
    """
    rows = []
    for _, r in df.iterrows():
        body = str(r["text"])
        if r["doc_type"] == "comment" and str(r["title"]) not in ("", "nan"):
            body = f'[Thread: {r["title"]}] {body}'
        parts = chunk_words(body, size, overlap_ratio)
        for ci, piece in enumerate(parts):
            rows.append(dict(
                chunk_id=f'{r["doc_id"]}__c{ci:02d}', doc_id=r["doc_id"],
                source_name=r["source_name"], organization=r["organization"],
                source_type=r["doc_type"], platform=r["platform"],
                modality=("video_metadata" if r["doc_type"] == "video"
                          else "image_metadata" if r["doc_type"] == "image_source"
                          else "text"),
                voice=VOICE_MAP.get(r["doc_type"], "unknown"),
                url=r["url"], thumbnail_url=r.get("thumbnail_url", ""),
                title=r["title"], date=r.get("published_date", ""),
                segment=r.get("segment", ""), hypothesis=r.get("hypothesis", ""),
                status=r["status"], authenticity_score=r["authenticity_score"],
                chunk_index=ci, n_chunks=len(parts), text=piece,
            ))
    return pd.DataFrame(rows)
