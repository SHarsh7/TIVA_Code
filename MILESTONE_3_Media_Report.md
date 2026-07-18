# MILESTONE 3 — Multimedia Metadata Processing: Status Report
**Deliverables:** `EV_Bus_Media_Data.csv` (86 media assets) · `EV_Bus_Unified_Data.csv` (2,852 rows)
**Status:** COMPLETE. No video files downloaded — metadata only, per directive.

---

## 1. Video Tab (80 rows)

Engagement metadata (views/likes/comments) was pre-captured in the source sheet — no re-scraping needed. Per video, this milestone:

- Extracted the 11-char YouTube video ID from each URL (80/80 parsed).
- **Verified every thumbnail live against the YouTube CDN** with HEAD requests: **66 serve `maxresdefault.jpg`** (HD); **14 fell back to `hqdefault.jpg`** (SD uploads — maxres returns 404 for these, so blind maxres URLs would have produced dead links). Every one of the 80 image data points is a confirmed-live URL.
- Computed engagement rates (`like_rate_pct`, `comment_rate_pct`) for the blueprint's engagement-correlation analysis.
- Built an embeddable `text_summary` per video (title + channel + segment + hypothesis + engagement) so video metadata participates in RAG retrieval.

### Contamination curation (tagged, never deleted)
| Row | Title | Verdict |
|---|---|---|
| 22 | "7 Causes Your Car Alternator not Charging" | `exclude` — car repair tutorial, zero EV-bus content |
| 23 | "Battery operated Jeep Charging Problem Solve … Electronic kids" | `exclude` — children's toy, not a vehicle |
| 56 | "Women workers at TATA Electronics protest after camera found in hostel bathroom" | `exclude` — workplace-privacy news event, no EV-bus content |
| 24 | "Riding an e-scooter to the office" (Pure Electric, UK) | `review` — adjacent mode, non-bus |
| 52 | "Travel tip - Bring electric scooters in our vehicles" (Sweden) | `review` — non-India, non-bus |

Result: **75 include · 2 review · 3 exclude**. The `status` column travels into the unified dataset so the RAG phase filters on it rather than losing provenance.

## 2. Image Tab (86 rows)

- **80 auto-generated thumbnail rows** verified as exact 1:1 derivatives of the video tab (video-ID set match: `True`). Merged into the video records (each video row carries its verified `thumbnail_url`) instead of being duplicated as standalone rows — the unified dataset stays dedupe-clean.
- **6 curated rows** processed individually:

| Row | Source | Outcome |
|---|---|---|
| OEM product pages (Tata Starbus, JBM Group, Olectra) | 3 URLs fetched | `include` — 34 image URLs harvested |
| JBM e-school-bus page | fetched | `include` — 12 URLs (incl. brake-disc / safety-feature imagery — direct H2 material) |
| Eicher Skyline Pro-E 9m page | fetched | `include` — 12 URLs (incl. EV-9m school-bus render; GD Goenka pilot model) |
| Tribune Delhi e-school-bus flag-off | fetched | `include` — 12 URLs (launch press imagery for H2 visual framing) |
| Fleet photo archives (no URL) | — | `no_url` — case-by-case curation slot |
| Instagram/X handles | — | `blocked` — auth-gated, per Milestone 1 decision |

**70 OEM/press image URLs** harvested total (og:image + content `<img>` tags, logo/icon/svg filtered) → `data/raw/oem_image_urls.json`. Known noise: a few tracking pixels and site-furniture images slipped the keyword filter; the image-analytics phase (CLIP lane) filters these at embed time.

## 3. Updated Unified Dataset Structure

**`EV_Bus_Unified_Data.csv` — 2,852 rows** = the full Milestone 2 text corpus + 86 media rows, in the shared 20-column schema **plus 7 media columns**:

`doc_id | source_name | organization | category | platform | doc_type | url | title | text | author | score | published_date | scraped_at | scrape_method | word_count | relevance_score | reliability_score | authenticity_score | in_2y_window | text_sha1 | thumbnail_url | segment | hypothesis | view_count | like_count | comment_count | status`

| doc_type | rows | notes |
|---|---|---|
| comment | 2,747 | Reddit, from M2 |
| video | 80 | `text` = embeddable metadata summary; thumbnail verified |
| article | 13 | from M2 |
| report_pdf | 6 | from M2 |
| image_source | 6 | curated image sources w/ harvest results |

Text rows carry `status=include` and blanks in media columns; media rows carry segment/hypothesis tags straight from the source sheet — free H1/H2 supervision labels for retrieval filtering.

## 4. Artifacts

| File | Role |
|---|---|
| `EV_Bus_Media_Data.csv` | Full-fidelity media metadata (86 rows, 23 columns) |
| `EV_Bus_Unified_Data.csv` | **The unified corpus for Milestone 4 ingestion** |
| `scrapers/media_metadata.py` | Milestone 3 processor |
| `data/raw/oem_image_urls.json` | Raw OEM page image harvest log |

`EV_Bus_Text_Data.csv` (M2 deliverable) left untouched.
