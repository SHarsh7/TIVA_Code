"""
MILESTONE 3 - Multimedia metadata processing (NO video downloads).

Inputs:  EV_Bus_Sources_Master_v2.xlsx  (Image_Sources, Video_Sources tabs)
Outputs: EV_Bus_Media_Data.csv          (one row per media asset, rich metadata)
         EV_Bus_Unified_Data.csv        (text corpus + media rows, shared schema)
         data/raw/oem_image_urls.json   (raw OEM page image harvest)

Processing:
 - Videos (80): extract video_id, verify thumbnail on YouTube CDN
   (maxresdefault -> hqdefault fallback via HTTP status), compute engagement
   rates from the pre-captured view/like/comment counts, build an embeddable
   text_summary. Contamination rows tagged status=exclude, never deleted.
 - Image tab (86): the 80 auto-generated thumbnail rows are 1:1 derivatives of
   the video rows -> merged into video records (verified by video_id match).
   The 6 curated rows are processed individually; OEM product pages are
   scraped for actual image URLs (og:image + content <img> tags).
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(BASE, "EV_Bus_Sources_Master_v2.xlsx")
TEXT_CSV = os.path.join(BASE, "EV_Bus_Text_Data.csv")
MEDIA_CSV = os.path.join(BASE, "EV_Bus_Media_Data.csv")
UNIFIED_CSV = os.path.join(BASE, "EV_Bus_Unified_Data.csv")
OEM_JSON = os.path.join(BASE, "data", "raw", "oem_image_urls.json")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Curation verdicts from title review (indices in Video_Sources tab).
# Tagged, not deleted - downstream filters on `status`.
VIDEO_EXCLUDES = {
    22: "off-topic contamination: car alternator repair tutorial (not EV bus)",
    23: "off-topic contamination: kids' battery toy jeep (not EV bus)",
    56: "off-topic: workplace privacy protest news, no EV-bus content",
}
VIDEO_REVIEW = {
    24: "marginal: e-scooter commute (UK), adjacent mode not bus",
    52: "marginal: e-scooters on Swedish regional buses, non-India context",
}

MEDIA_FIELDS = [
    "media_id", "media_type", "status", "status_reason", "source_name",
    "organization", "title", "url", "thumbnail_url", "thumbnail_resolution",
    "segment", "hypothesis", "view_count", "like_count", "comment_count",
    "like_rate_pct", "comment_rate_pct", "category", "relevance_score",
    "reliability_score", "authenticity_score", "text_summary", "scraped_at",
]


def video_id_from_url(url):
    m = re.search(r"(?:v=|youtu\.be/|/vi/)([A-Za-z0-9_-]{11})", str(url))
    return m.group(1) if m else None


def verify_thumbnail(session, vid):
    """maxresdefault only exists for HD uploads; hqdefault always resolves."""
    for res in ("maxresdefault", "hqdefault"):
        url = f"https://img.youtube.com/vi/{vid}/{res}.jpg"
        try:
            r = session.head(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return url, res
        except requests.RequestException:
            pass
        time.sleep(0.15)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg", "unverified"


def fmt_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return ""


def process_videos(session):
    df = pd.read_excel(XLSX, "Video_Sources")
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for i, r in df.iterrows():
        vid = video_id_from_url(r["video_url"])
        thumb_url, thumb_res = verify_thumbnail(session, vid) if vid else ("", "no_id")
        views, likes, comments = fmt_int(r["view_count"]), fmt_int(r["like_count"]), fmt_int(r["comment_count"])
        like_rate = round(100 * likes / views, 3) if views and likes != "" else ""
        comment_rate = round(100 * comments / views, 3) if views and comments != "" else ""

        if i in VIDEO_EXCLUDES:
            status, reason = "exclude", VIDEO_EXCLUDES[i]
        elif i in VIDEO_REVIEW:
            status, reason = "review", VIDEO_REVIEW[i]
        else:
            status, reason = "include", ""

        title = str(r["title"]).replace("\n", " ").strip()
        channel = str(r["channel"]).replace("\n", " ").strip()
        eng = f"{views:,} views" if views != "" else "views n/a"
        if likes != "":
            eng += f", {likes:,} likes"
        if comments != "":
            eng += f", {comments:,} comments"
        summary = (f"YouTube video: \"{title}\" by {channel}. "
                   f"Segment: {r['segment']}; hypothesis {r['hypothesis']}. "
                   f"Engagement: {eng}"
                   + (f" (like rate {like_rate}%)." if like_rate != "" else "."))
        rows.append(dict(
            media_id=f"video_{i:03d}_{vid or 'noid'}", media_type="video",
            status=status, status_reason=reason,
            source_name=title[:120], organization=channel,
            title=title, url=str(r["video_url"]).strip(),
            thumbnail_url=thumb_url, thumbnail_resolution=thumb_res,
            segment=r["segment"], hypothesis=r["hypothesis"],
            view_count=views, like_count=likes, comment_count=comments,
            like_rate_pct=like_rate, comment_rate_pct=comment_rate,
            category="Video - YouTube", relevance_score="",
            reliability_score="", authenticity_score=fmt_int(r["Authenticity_Score"]),
            text_summary=summary, scraped_at=now,
        ))
        print(f"[{i+1}/{len(df)}] {vid} {thumb_res:14s} {status:8s} {title[:55]}")
    return rows


def harvest_oem_images(session, name, page_url, max_imgs=12):
    """Pull og:image + content <img> URLs from an OEM/press page."""
    try:
        resp = session.get(page_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return [], f"fetch failed: {type(e).__name__}"
    soup = BeautifulSoup(resp.text, "lxml")
    urls, seen = [], set()
    og = soup.select_one('meta[property="og:image"]')
    if og and og.get("content"):
        u = urljoin(page_url, og["content"])
        seen.add(u); urls.append(u)
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue
        u = urljoin(page_url, src)
        low = u.lower()
        if u in seen or low.endswith(".svg"):
            continue
        if any(k in low for k in ("logo", "icon", "sprite", "favicon", "placeholder", "blank")):
            continue
        seen.add(u); urls.append(u)
        if len(urls) >= max_imgs:
            break
    return urls, "ok"


def process_curated_images(session):
    df = pd.read_excel(XLSX, "Image_Sources")
    curated = df[df["Category"] != "Video Thumbnail (YouTube)"]
    auto = df[df["Category"] == "Video Thumbnail (YouTube)"]
    print(f"\nImage tab: {len(curated)} curated + {len(auto)} auto-generated thumbnail rows")

    rows, harvest_log = [], {}
    now = datetime.now(timezone.utc).isoformat()
    for i, r in curated.iterrows():
        base_urls = [u.strip() for u in str(r.get("Base URL") or "").split("|") if u.strip() and u.strip().lower() != "nan"]
        access = str(r.get("Access Type") or "")
        if "login" in access.lower() or "instagram" in str(r["Source Name"]).lower():
            status, reason, harvested = "blocked", "auth-gated platform (Instagram/X) - per M1 decision, no scraping", []
        elif not base_urls:
            status, reason, harvested = "no_url", "concept row without direct URL - case-by-case curation needed", []
        else:
            status, reason, harvested = "include", "", []
            for u in base_urls:
                imgs, note = harvest_oem_images(session, r["Source Name"], u)
                harvest_log[u] = {"status": note, "images": imgs}
                harvested.extend(imgs)
                print(f"  {u[:70]} -> {len(imgs)} images ({note})")
                time.sleep(2)
        desc = str(r.get("Description") or "").strip()
        summary = (f"Image source: {r['Source Name']} ({r['Organization']}). {desc} "
                   f"Category: {r['Category']}."
                   + (f" Harvested {len(harvested)} image URLs." if harvested else ""))
        rows.append(dict(
            media_id=f"image_{i:03d}", media_type="image_source",
            status=status, status_reason=reason,
            source_name=str(r["Source Name"]), organization=str(r["Organization"]),
            title=str(r["Source Name"]), url=" | ".join(base_urls),
            thumbnail_url=harvested[0] if harvested else "",
            thumbnail_resolution="page_harvest" if harvested else "",
            segment="", hypothesis="",
            view_count="", like_count="", comment_count="",
            like_rate_pct="", comment_rate_pct="",
            category=str(r["Category"]),
            relevance_score=fmt_int(r["Relevance Score (1-10)"]),
            reliability_score=fmt_int(r["Reliability Score (1-10)"]),
            authenticity_score=fmt_int(r["Authenticity_Score"]),
            text_summary=summary, scraped_at=now,
        ))
    # verify the 80 auto rows are true 1:1 derivatives of the video tab
    auto_ids = {video_id_from_url(u) for u in auto["Base URL"]}
    vids = {video_id_from_url(u) for u in pd.read_excel(XLSX, "Video_Sources")["video_url"]}
    print(f"Auto-thumbnail rows match video tab 1:1: {auto_ids == vids} "
          f"({len(auto_ids)} thumb ids vs {len(vids)} video ids)")
    os.makedirs(os.path.dirname(OEM_JSON), exist_ok=True)
    with open(OEM_JSON, "w", encoding="utf-8") as f:
        json.dump(harvest_log, f, indent=2)
    return rows


def build_unified(media_rows):
    text_df = pd.read_csv(TEXT_CSV, encoding="utf-8-sig")
    media_df = pd.DataFrame(media_rows)

    mapped = pd.DataFrame({
        "doc_id": media_df["media_id"],
        "source_name": media_df["source_name"],
        "organization": media_df["organization"],
        "category": media_df["category"],
        "platform": media_df["media_type"].map(
            {"video": "youtube", "image_source": "web"}),
        "doc_type": media_df["media_type"],
        "url": media_df["url"],
        "title": media_df["title"],
        "text": media_df["text_summary"],
        "author": media_df["organization"],
        "score": media_df["view_count"],
        "published_date": "",
        "scraped_at": media_df["scraped_at"],
        "scrape_method": "metadata-only",
        "word_count": media_df["text_summary"].str.split().str.len(),
        "relevance_score": media_df["relevance_score"],
        "reliability_score": media_df["reliability_score"],
        "authenticity_score": media_df["authenticity_score"],
        "in_2y_window": "",
        "text_sha1": "",
    })
    for col in ["thumbnail_url", "segment", "hypothesis",
                "view_count", "like_count", "comment_count", "status"]:
        mapped[col] = media_df[col]
    for col in ["thumbnail_url", "segment", "hypothesis",
                "view_count", "like_count", "comment_count"]:
        text_df[col] = ""
    text_df["status"] = "include"

    unified = pd.concat([text_df, mapped], ignore_index=True)
    unified.to_csv(UNIFIED_CSV, index=False, encoding="utf-8-sig")
    return unified


def main():
    session = requests.Session()
    print("=" * 80); print("VIDEO TAB (80 rows) - thumbnail verification + engagement")
    print("=" * 80)
    video_rows = process_videos(session)
    print("\n" + "=" * 80); print("IMAGE TAB - curated rows + OEM page harvest")
    print("=" * 80)
    image_rows = process_curated_images(session)

    media_df = pd.DataFrame(video_rows + image_rows)[MEDIA_FIELDS]
    media_df.to_csv(MEDIA_CSV, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(media_df)} media rows -> {MEDIA_CSV}")
    print(media_df.groupby(["media_type", "status"]).size().to_string())

    unified = build_unified(video_rows + image_rows)
    print(f"\nUnified dataset: {len(unified)} rows -> {UNIFIED_CSV}")
    print(unified["doc_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
