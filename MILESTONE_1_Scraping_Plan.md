# MILESTONE 1 — Source Triage & Scraping Strategy
**Project:** EV Bus Multimodal RAG Pipeline (TIVA AI 2026)
**Input:** `EV_Bus_Sources_Master_v2.xlsx` | **Status:** Awaiting authorization for Milestone 2

---

## 1. Inventory Snapshot

| Sheet | Rows | Composition |
|---|---|---|
| Text_Sources | 29 | News, govt data portals, PDFs, dashboards, forums |
| Image_Sources | 86 | 6 curated web/OEM sources + 80 auto-generated YouTube thumbnail rows |
| Video_Sources | 80 | All YouTube, with view/like/comment counts + segment/hypothesis tags already in-sheet |

---

## 2. Text Source Triage (29 rows → 5 tiers)

### TIER A — Standard scrape: `requests` + `trafilatura`/`BeautifulSoup` (13 sources) ✅
News and corporate pages that serve static/server-rendered HTML. Highest yield per effort.

| Source | Domain | Notes |
|---|---|---|
| PIB National E-Bus Programme release | pib.gov.in | Scrape-friendly, static HTML |
| Telangana Today fitness inspections | telanganatoday.com | Standard WordPress |
| Hans India / NewsMeter audit coverage | (via Telangana audit row) | Standard news templates |
| Tribune: Delhi 24 e-school bus flag-off | tribuneindia.com | Static; retry with AMP variant if blocked |
| Tribune: GD Goenka Eicher pilot | tribuneindia.com | URL is already `/amp` — easiest form |
| Deccan Herald: BMTC ride-quality | deccanherald.com | Soft paywall risk; AMP fallback |
| Deccan Herald: e-bus issues (minister) | deccanherald.com | URL already AMP-form |
| Deccan Herald: MoveInSync commute | deccanherald.com | URL already AMP-form |
| EVreporter school-bus economics | evreporter.com | Standard WordPress |
| 91Trucks buying guide | 91trucks.com | Standard |
| Uber Corporate Shuttle newsroom | uber.com/newsroom | Mostly server-rendered; Playwright fallback |
| YourStory corporate EV commute | yourstory.com | JS-heavy; trafilatura usually works, Playwright fallback |
| MoveInSync corporate site | moveinsync.com | Standard |

### TIER B — Direct PDF download: `requests` + `PyMuPDF` (6 sources) ✅
| Source | URL health check needed |
|---|---|
| ITDP "Status of E-buses in India" PDF | OK (itdp.in wp-content) |
| ITDP "Guidance for E-Bus Rollout" PDF | OK |
| UITP Performance Evaluation (landing page → PDF link) | UITP sometimes gates behind a form — flag, resolve at runtime |
| SIAM Roadmap Feb-2026 PDF | Direct |
| UNCRD/Ashok Leyland EST Forum deck (sdgs.un.org) | Direct; URL is %-encoded — pass as-is |
| CESL Grand Challenge case study | ⚠️ Sheet contains TWO path variants (`electricbus` vs `electric_bus`) — one will 404; test both |

### TIER C — Govt portals / dashboards: hybrid approach (4 sources) ⚠️
| Source | Constraint | Strategy |
|---|---|---|
| data.gov.in VAHAN e-bus counts | Download requires (free) registration; open API needs api-key | Scrape resource metadata page; use the public sample API key (`579b46...`) that data.gov.in documents for testing; else flag for manual CSV download |
| data.gov.in SRTU fleet catalog | Same | Same |
| NITI Aayog IEMI dashboard | Fully JS-rendered SPA; no static payload | Playwright headless to capture rendered metrics, or manual screenshot + OCR (routes into image pipeline). Low priority |
| WRI Electric School Bus dashboard | JS dashboard (US benchmark only) | WRI publishes the underlying dataset as a downloadable CSV — fetch that instead of scraping the dashboard |

### TIER D — Hardened platforms (2 sources) 🔴
| Source | Constraint | Proposed handling |
|---|---|---|
| Reddit (r/bangalore, r/hyderabad, r/delhi, r/mumbai) | Post-level access needs OAuth (PRAW) or rate-limited public JSON | **Primary:** old-reddit public `.json` endpoints with descriptive User-Agent + keyword search (`electric bus`, `school bus`, `BMTC`, `Cityflo`, `Shuttl`) — works without credentials at low request rates. **Fallback:** PRAW if you can supply a (free) Reddit API client-id/secret. No Selenium needed — Reddit's JSON API beats DOM scraping. |
| Kaggle EV reviews dataset | Requires Kaggle account/API token | **BLOCKED without credentials.** If you have a `kaggle.json` token, it's a one-line `kagglehub` download. Otherwise flag and drop (blueprint rates it 7/10, car-biased anyway). |

### TIER E — Blocked / manual-only (1 source) 🔴
| Source | Verdict |
|---|---|
| LinkedIn employee-transport posts | **Do not scrape.** LinkedIn ToS prohibits it and your own blueprint (§4 L0) mandates manual export + documented ethics note. Requires you to paste exported post text manually. |

---

## 3. Image Source Triage (86 rows)

| Group | Count | Strategy |
|---|---|---|
| YouTube thumbnails (`img.youtube.com`) | 80 | ✅ Trivial CDN fetch, no auth. Use `maxresdefault.jpg` with automatic fallback to `hqdefault.jpg` (maxres 404s on many older videos). Handled in **Milestone 3**. |
| OEM product pages (Tata busesandvans, JBM Group, jbmbuses, Olectra, Eicher) | 4 rows / ~6 domains | ✅ `requests` + BS4 `<img>`/`srcset` extraction, respecting robots.txt. Eicher & Tata pages may lazy-load — Playwright fallback ready. |
| Tribune press photos | 1 | ✅ Extract article `<img>` tags during the Tier-A text scrape (same fetch, dual use). |
| Instagram / X handles | 1 | 🔴 **BLOCKED without auth.** Post-level scraping requires login/API; the sheet itself says so. Apify actors (`apify/instagram-scraper`, `apidojo/tweet-scraper`) would work but need an Apify API token we don't possess. **Flagging per directive** — decide: (a) supply Apify token, (b) manual save of ~10 posts, or (c) drop with a documented note (Authenticity_Score is already lowest in sheet at 65). |
| "School & Corporate Fleet Photo Archives" | 1 | 🔴 **No URL provided in sheet** — un-actionable as specified. Will be partially covered by press-article images. |

---

## 4. Video Source Triage (80 rows — all YouTube)

Engagement metadata (views/likes/comments) already lives in the sheet — half the work is pre-done. Per the directive we do **not** download video files. Strategy for Milestone 3:

1. `yt-dlp --dump-json --skip-download` per URL → refresh stats + capture duration, upload date, channel, subscriber count (blueprint §6 L1 marks this P0).
2. Thumbnail URL generation (already materialized in Image_Sources).
3. Optional P1: `youtube-comment-downloader` for top-N comments per video → feeds the text corpus (no API key needed).

⚠️ One data-quality flag: a few rows are off-topic contamination (e.g., *"7 Causes Your Car Alternator not Charging"*, *"Battery operated Jeep Charging Problem"*, a kids-toy repair video). They inherited Authenticity_Score 80 despite zero relevance to EV buses. Recommend tagging these `exclude` during Milestone 3 rather than deleting rows.

---

## 5. Execution Plan for Milestone 2 (proposed)

```
Phase 2.1  Tier A (13 news/corporate pages)      requests → trafilatura → BS4 fallback
Phase 2.2  Tier B (6 PDFs)                       requests → PyMuPDF text extraction
Phase 2.3  Tier C (data.gov.in API, WRI CSV)     API/CSV pulls; IEMI dashboard deferred
Phase 2.4  Tier D (Reddit JSON search)           keyword-filtered, rate-limited, UA-stamped
Phase 2.5  Clean + normalize                     strip boilerplate/HTML, dedupe, lang-tag
Phase 2.6  Consolidate → EV_Bus_Text_Data.csv
```

**Unified CSV schema** (aligned to blueprint §3.3 data contract):
`doc_id | source_name | organization | category | url | domain | scrape_method | scraped_at | title | text | word_count | relevance_score | reliability_score | authenticity_score | access_type | status`

**Politeness controls:** 2–4 s randomized delay per domain, descriptive User-Agent, robots.txt check, 2-retry cap, every failure logged to a `failed_sources` report rather than silently dropped.

---

## 6. Items Requiring Your Decision Before/During Milestone 2

1. **Kaggle dataset** — supply `kaggle.json` token, or drop? (Recommend: drop — car-biased, lowest text-tier relevance.)
2. **Instagram/X** — Apify token, manual export, or drop with documented note? (Recommend: drop + note; score 65 is the sheet's lowest.)
3. **LinkedIn** — will you paste manually exported posts? (Scraping is off the table per ToS + blueprint.)
4. **NITI IEMI dashboard** — Playwright render attempt, or defer to manual screenshot? (Recommend: defer.)
