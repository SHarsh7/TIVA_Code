"""
MILESTONE 2 - Clean, normalize, consolidate -> EV_Bus_Text_Data.csv

Inputs:
  data/raw/web_scrape_results.jsonl        (Tier A/B/C articles, PDFs, datasets)
  data/raw/reddit/reddit_comments_*.csv    (per-subreddit comment dumps)

Cleaning:
  - unicode normalization (NFKC), whitespace collapse, control-char strip
  - boilerplate line removal for articles (cookie/subscribe/share prompts)
  - dedupe: exact text-hash within source type
  - Reddit: drop bot/automod comments, apply 2-year window on timestamps
    (cutoff 2024-07-18; untimestamped comments kept, flagged in `in_2y_window`)

Output: EV_Bus_Text_Data.csv - one row per document (article/PDF/dataset = 1 doc,
        each Reddit comment = 1 doc), blueprint §3.3-aligned columns.
"""

import csv
import glob
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
OUT_CSV = os.path.join(BASE, "EV_Bus_Text_Data.csv")

CUTOFF_2Y = datetime(2024, 7, 18, tzinfo=timezone.utc)

BOILERPLATE_PATTERNS = [
    r"(?i)^(subscribe|sign up|log ?in|register|share this|follow us|advertisement|adver?t\b).*",
    r"(?i)^(cookie|privacy) (policy|notice|settings).*",
    r"(?i)^(click here|read more|also read|related articles?|trending now|download (the )?app).*",
    r"(?i)^(published|updated) ?[:\-] ?\d.*",
    r"(?i)^copyright ©.*",
]

BOT_AUTHORS = {"automoderator", "automod", "[deleted]"}


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boilerplate(text):
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if any(re.match(p, line) for p in BOILERPLATE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def sha1(text):
    return hashlib.sha1(text.lower().encode("utf-8")).hexdigest()


def load_web_sources():
    rows = []
    path = os.path.join(RAW, "web_scrape_results.jsonl")
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok" or not rec.get("text"):
                continue
            text = normalize_text(rec["text"])
            if rec["scrape_method"] in ("html", "selenium"):
                text = strip_boilerplate(text)
            rows.append(dict(
                doc_id=rec["doc_id"],
                source_name=rec["source_name"],
                organization=rec.get("organization", ""),
                category=rec.get("category", ""),
                platform="web",
                doc_type={"pdf": "report_pdf", "uitp": "report_pdf",
                          "datagov-api": "dataset", "ckan-api": "dataset",
                          }.get(rec["scrape_method"], "article"),
                url=rec.get("final_url") or rec["url"],
                title=normalize_text(rec.get("title") or rec["source_name"]),
                text=text,
                author="",
                score="",
                published_date=rec.get("published_date") or "",
                scraped_at=rec.get("scraped_at", ""),
                scrape_method=rec["scrape_method"],
                relevance_score=rec.get("relevance_score", ""),
                reliability_score=rec.get("reliability_score", ""),
                authenticity_score=rec.get("authenticity_score", ""),
                in_2y_window="",  # n/a for reference docs
            ))
    return rows


def load_reddit():
    rows, dropped_old, dropped_bot = [], 0, 0
    for path in sorted(glob.glob(os.path.join(RAW, "reddit", "reddit_comments_*.csv"))):
        sub = os.path.basename(path).replace("reddit_comments_", "").replace(".csv", "")
        with open(path, encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                text = normalize_text(row.get("comment_text", ""))
                if len(text) < 3:
                    continue
                author = (row.get("comment_author") or "").strip().lower()
                if author in BOT_AUTHORS or "i am a bot" in text.lower():
                    dropped_bot += 1
                    continue
                ts_raw = (row.get("timestamp") or "").strip()
                in_window = "unknown"
                if ts_raw and ts_raw != "N/A":
                    try:
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        if ts < CUTOFF_2Y:
                            dropped_old += 1
                            continue
                        in_window = "yes"
                    except ValueError:
                        pass
                rows.append(dict(
                    doc_id=f"reddit_{sub}_{i:05d}",
                    source_name=f"Reddit r/{sub} commute & school-bus threads",
                    organization=f"Reddit (r/{sub})",
                    category="Text - Parent & Rider Sentiment",
                    platform="reddit",
                    doc_type="comment",
                    url=row.get("post_url", ""),
                    title=normalize_text(row.get("post_title", "")),
                    text=text,
                    author=row.get("comment_author", ""),
                    score=row.get("comment_score", ""),
                    published_date=ts_raw if ts_raw != "N/A" else "",
                    scraped_at=datetime.now(timezone.utc).isoformat(),
                    scrape_method="old.reddit-requests",
                    relevance_score=8,
                    reliability_score=6,
                    authenticity_score=70,
                    in_2y_window=in_window,
                ))
    print(f"Reddit: kept {len(rows)}, dropped {dropped_old} older than 2y, {dropped_bot} bot/deleted")
    return rows


def main():
    web = load_web_sources()
    reddit = load_reddit()
    all_rows = web + reddit
    print(f"Loaded: {len(web)} web docs, {len(reddit)} reddit comments")

    # dedupe on normalized text hash (bots, crossposts, syndicated articles)
    seen, deduped, dupes = set(), [], 0
    for r in all_rows:
        h = sha1(r["text"])
        if h in seen:
            dupes += 1
            continue
        seen.add(h)
        r["text_sha1"] = h
        r["word_count"] = len(r["text"].split())
        deduped.append(r)
    print(f"Dedupe removed {dupes} exact duplicates")

    df = pd.DataFrame(deduped)
    col_order = ["doc_id", "source_name", "organization", "category", "platform",
                 "doc_type", "url", "title", "text", "author", "score",
                 "published_date", "scraped_at", "scrape_method", "word_count",
                 "relevance_score", "reliability_score", "authenticity_score",
                 "in_2y_window", "text_sha1"]
    df = df[col_order]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")
    print("\nBy doc_type:")
    print(df["doc_type"].value_counts().to_string())
    print("\nBy platform:")
    print(df["platform"].value_counts().to_string())
    print(f"\nTotal words in corpus: {df['word_count'].sum():,}")


if __name__ == "__main__":
    main()
