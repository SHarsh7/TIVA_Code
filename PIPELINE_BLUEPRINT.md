# Multimodal Analytics Pipeline Blueprint
## TIVA AI 2026 — Academic Group 1
### "Decoding electric bus adoption in India's school & employee transport segments"

**Status:** Design blueprint. No implementation code. Ready to execute the moment data lands.
**Written:** 17 July 2026 | **Stated final submission:** 18 July 2026 (EOD)

---

## 0. Read This First — Three Design Decisions That Deviate From The Brief

This blueprint deliberately departs from the requested 10-phase structure in three places. Each departure is justified by the assignment PDF itself.

### 0.1 RAG is added as a first-class phase (was absent from the brief)

The assignment's **Phase 3 (Module 4)** requires "an operational Multimodal RAG System" using "cross-modal embeddings to index both textual data and visual layouts," plus **Task 3.2 Executive Query Testing**, and the 3-Year Strategy must be compiled "relying on the outputs of your RAG pipeline."

The rubric names "RAG pipeline orchestration" as one of four bullets inside the 50%-weighted Technical Notebook component. A pipeline without RAG forfeits roughly a third of the technical marks and breaks the causal chain the graders are looking for: *unstructured data → retrieval → strategy*. RAG is therefore Phase 6 below, and it is **P0**.

### 0.2 Several requested techniques are dropped or substituted (with reasons)

The brief's text-analytics levels appear written for an interview/essay corpus. Our corpus is Reddit comments, YouTube comments, LinkedIn posts, news articles, and video transcripts. Techniques that are excellent on 800-word essays are noise on 14-word comments. Honest triage:

| Requested | Verdict | Reason |
|---|---|---|
| **LIWC** | **Cut — substitute** | Proprietary and licence-gated. Not pip-installable in Colab. Substitute: **NRC Emotion Lexicon** + **Empath** (free, LIWC-like categories, academically citable). Say so explicitly in the notebook — awareness of the constraint reads as rigor, silently faking LIWC categories reads as fraud. |
| **Evaluative Lexicon** (Rocklage & Fazio) | **Cut** | Access is by author request. Cannot be obtained in the time available. Its construct (evaluative extremity/emotionality) is approximated by VADER intensity + NRC arousal. |
| **PARA** | **Cut — flag** | I could not verify a standard lexicon by this name in the text-analytics literature relevant here (PARA is commonly a knowledge-management method). If a faculty member named a specific PARA lexicon in class, substitute it back in — otherwise do not cite a lexicon we cannot produce. |
| **Loughran-McDonald** | **Scoped, not global** | Built for 10-K/financial-disclosure language. Applying it to parent Reddit comments is a category error a sharp examiner will catch. **Use it only on the structured reference backbone** (IEA/ITDP/UITP/CESL/IFC reports) to score institutional-document uncertainty and litigiousness around battery supply chain and financing risk. That is defensible and directly feeds the strategy's risk section. |
| **Readability suite** (Flesch, FK, Gunning Fog, SMOG) | **Scoped, not global** | Statistically meaningless below ~100 words; a 14-word YouTube comment produces garbage grade-levels. **Apply only to** video transcripts, news articles, LinkedIn long-form posts, and OEM marketing copy. Report N and word-count floor. This becomes a genuine finding: *is OEM communication pitched above the reading level of the parent audience it must persuade?* |
| **Posture / gesture / eye-contact analytics** | **Cut** | Our video corpus is OEM ads, walkthroughs, and influencer reviews — mostly b-roll of buses, not talking-head interviews. MediaPipe pose over bus footage yields noise. The assignment asks for **"structural and pacing metrics"** mapped to engagement, which is a different and better-posed question. Build that instead. |
| **"Predictive insights"** | **Reframed as associative** | With n ≈ 40–60 videos, a predictive model overfits and any reported R² is theatre. Report **Spearman correlations with confidence intervals** and call them associations. Stating this limitation out loud scores on Pillar 3 (Methodological Rigor); hiding it loses more than the finding gains. |

### 0.3 The pipeline is triaged for a one-day runway

Marked throughout: **[P0]** submission-critical, **[P1]** do if time survives, **[P2]** only if data arrives early. A complete P0 path is a strong submission; a half-finished P1 path is not.

---

## 1. Assignment Understanding

### 1.1 The setup

We act as a **Strategic AI Taskforce for an automotive OEM** in the EV bus market. The work is to convert unstructured multimodal streams into strategic intelligence, ending in a RAG-backed executive decision tool.

### 1.2 Required dataset architecture (four packages)

| Package | Assignment spec | Our proposal's instantiation |
|---|---|---|
| **Text** | Reviews + social discussion across pre-purchase / purchase / post-use | Parent forums, city subreddits, LinkedIn facility-manager discourse, Google News (Telangana audits), YouTube comments |
| **Image** | UGC, competitor designs, dashboard UI/UX, charging infrastructure | Tata Starbus School/EV, SML Isuzu Hiroi.EV, JBM visuals; fleet & campus charging photos; safety-equipment imagery |
| **Video** | Influencer reviews, OEM commercials, test-drive footage | OEM launch/walkthroughs, fleet-induction coverage, parent/employee perspective content + engagement metrics |
| **Multimodal reference** | Unstructured text + visual sources | IFC, Telangana Transport, MoveInSync, YourStory, ITDP, UITP, CESL + Statista charts, IEA trends |

**Gap to close:** the assignment explicitly names **Statista charts and IEA market trends** as RAG inputs. The proposal's reference backbone does not list either. Add at minimum 3–5 Statista chart images (India EV bus sales/fleet) and the IEA Global EV Outlook PDF. These are named in the PDF; a grader will look for them.

### 1.3 Phased tasks → our mapping

| Assignment task | Our pipeline phase |
|---|---|
| 1.1 Sentiment across pre-purchase / post-use lifecycle | Phase 4, Levels 4–5 + **lifecycle-stage tagging** |
| 1.2 Topic modeling for latent themes | Phase 4, Level 6 (BERTopic + LDA baseline) |
| 2.1 Image features (low + high level) ↔ perception | Phase 5 |
| 2.2 Video structural/pacing metrics ↔ engagement | Phase 6 |
| 3.1 Multimodal RAG w/ cross-modal embeddings | Phase 7 |
| 3.2 Executive query testing | Phase 7, Level 5 |

### 1.4 Deliverables & weights (from the PDF)

| Component | Format | Weight |
|---|---|---|
| Technical Notebook & Code | Colab / GitHub | **50%** |
| Strategic Business Report | Executive PDF, max 20 pages | **20%** |
| Boardroom Pitch Presentation | Live + Q&A | **30%** |

### 1.5 Rubric pillars → where this blueprint earns each

| Pillar | Where earned |
|---|---|
| **1. Multimodal Technical Depth** — independent text/image/video pipelines *before* RAG | Phases 4/5/6 run standalone and emit clean feature tables; Phase 7 consumes them. Keep the separation visible in the notebook structure — the wording "independent pipelines before passing them into a functional RAG environment" is a structural requirement, not prose. |
| **2. Analytical & Strategic Cohesion** — ML metrics → business choices | The **Insight Ledger** (§10.1): every strategic claim carries a metric ID and a chart. |
| **3. Methodological Rigor** — preprocessing, validation, **bias awareness**, evaluation metrics | Validation on labelled subsets (§4.7), RAG eval harness (§7.5), explicit **Bias & Limitations register** (§11). Bias awareness is named in the rubric and is the cheapest pillar to win — most teams skip it. |
| **4. Professional Execution** — communication, visualization, executable pitch | Phase 9 viz system, Phase 11 report/pitch spine. |

---

## 2. Research & Business Objectives

### 2.1 Business objective
Enable an EV bus OEM to enter India's **institutional bus segment** (school + employee transport) — 1.2M private vs 0.8M public buses — by decoding what the four ecosystem actors actually say, show, and respond to, and converting that into a 3-year strategy on positioning, infrastructure partnerships, battery supply-chain risk, and marketing spend.

### 2.2 Research objective
Quantify adoption drivers and barriers across **OEM / operator / institutional buyer / rider-parent** using text, image, and video signals, and expose where the buyer conversation and the rider conversation diverge.

### 2.3 Research questions (from proposal hypotheses → testable form)

| ID | Hypothesis | Operational test | Primary metric |
|---|---|---|---|
| **H1** | Buyer talks cost/compliance/ESG; rider-parent talks safety/comfort/reliability; adoption stalls where they diverge | Voice-classify docs → topic distributions per voice → divergence | **Jensen-Shannon divergence** between voice-conditional topic distributions + Quantitative Reasoning Density gap |
| **H2** | In school content, safety framing beats green/cost framing on sentiment and engagement | Frame-score each text/image/video asset → regress sentiment & engagement on frame scores | **Safety Salience Index** vs **Green/Econ Salience Index**; Spearman ρ to engagement rate |
| **H3** | Employee transport electrifies car-first, bus-last; bus barriers differ measurably from car barriers | Split corpus car-mode vs bus-mode → compare barrier-topic prevalence | Barrier-topic prevalence deltas + **log-odds ratio with informative Dirichlet prior** (Monroe et al.) |

**Note on H1's framing.** "Adoption stalls where these two conversations diverge" is a causal claim our corpus cannot test — we have no adoption outcome variable per document. Reframe to the defensible version: *demonstrate and quantify the divergence*, then argue its strategic implication. Do not present divergence as proof of stalling. This distinction is exactly what Q&A will probe.

---

## 3. The Multimodal Analytics Framework

### 3.1 Canonical flow (every modality obeys this)

```
INPUT
  ↓  acquisition + provenance stamping
PREPROCESSING
  ↓  clean, normalize, dedupe, language-filter, segment
FEATURE EXTRACTION
  ↓  low-level → high-level → custom composite metrics
ANALYSIS
  ↓  hypothesis tests, correlations, models (+ validation)
INSIGHTS
  ↓  Insight Ledger entry: metric → claim → confidence
VISUALIZATION
  ↓  the one chart that carries the claim
REPORTING
     Report §, pitch slide, RAG-citable artifact
```

### 3.2 The three pipelines and their convergence

```
   TEXT PACKAGE          IMAGE PACKAGE          VIDEO PACKAGE
        │                      │                      │
   [Phase 4]              [Phase 5]              [Phase 6]
   clean/tag              quality/CLIP            shots/pacing
   sentiment              objects/OCR             Whisper ASR ──┐
   topics                 colour/saliency         audio energy  │
   frames/lexicons        frame scores            engagement    │
        │                      │                      │         │
        │                      │            transcripts ────────┘
        │                      │                      │      (re-enters Phase 4
        ▼                      ▼                      ▼       as documents)
   text_features.parquet  image_features.parquet  video_features.parquet
        │                      │                      │
        └──────────────┬───────┴──────────────────────┘
                       ▼
              [Phase 8] FUSION LAYER
              asset-level join, cross-modal correlation,
              H1/H2/H3 tests
                       │
                       ▼
              [Phase 7] MULTIMODAL RAG
              text chunks + CLIP-indexed visual layouts
              + Statista charts + IEA/ITDP/UITP/CESL PDFs
                       │
                       ▼
              EXECUTIVE QUERIES → 3-YEAR EV STRATEGY
```

**Why fusion sits beside RAG rather than inside it:** the fusion layer answers *our* research questions (H1–H3) with statistics. RAG answers *an executive's ad-hoc* questions with retrieval. Both are required; conflating them muddies both. Fusion findings become high-value documents *in* the RAG corpus — that is the join.

### 3.3 The unifying data contract

Every asset — a comment, a photo, a video — gets a row in a master registry before any analysis:

```
asset_id | modality | source_platform | source_url | collected_at
| voice (oem|operator|buyer|rider_parent|media|unknown)
| segment (school|employee|generic)
| mode (bus|car|mixed)
| lifecycle_stage (pre_purchase|purchase|post_use|na)
| lang | licence_note | sha256
```

These five tags do the entire analytical job. **H1 needs `voice`. H2 needs `segment`. H3 needs `mode`. Task 1.1 explicitly needs `lifecycle_stage`.** If tagging is wrong, every downstream number is wrong regardless of model quality. Budget real time here — it is the highest-leverage hour in the project.

---

## 4. Text Analytics Pipeline

**Input:** Reddit/forum comments, YouTube comments, LinkedIn posts, news articles, video transcripts (from Phase 6), OCR text (from Phase 5).
**Output:** `text_features.parquet` — one row per document, ~60 columns.

### Level 0 — Ingestion & provenance **[P0]**

- Sources per proposal §A. Loaders: `praw`/Pushshift-style export for Reddit, `youtube-comment-downloader` or YouTube Data API v3 for comments, `newspaper3k`/`trafilatura` for news, manual export for LinkedIn (**ToS: LinkedIn scraping violates their terms — collect manually, document that you did, and note it in the report; graders on Pillar 3 notice ethics**).
- Stamp every doc with the §3.3 contract. Store raw immutably; never edit in place.
- **Target N:** ≥1,500 documents post-clean, with ≥200 per voice for H1 to be more than anecdote. Below ~800 total, say so and downgrade claims to exploratory.

### Level 1 — Basic NLP **[P0]**

Tokenization, lemmatization, stop-word removal, n-grams — as taught (`nltk`, `spacy` en_core_web_sm).

Domain-specific decisions that matter more than the defaults:
- **Custom stopwords:** bus, electric, EV, school — they saturate every doc and drown topic models.
- **Protect multiword terms** before tokenizing: `panic button`, `speed governor`, `fire extinguisher`, `charging infrastructure`, `range anxiety`, `fitness certificate`, `staff transport`. Losing these to unigrams destroys H2.
- **Hinglish/code-mix:** Indian corpora will contain it. Detect with `langdetect`/`fasttext-langid`; route non-English to a translation pass or a documented exclusion. Do not silently drop — report the share.
- Dedupe: exact hash + near-dup via MinHash/`datasketch` (bots and crossposts inflate N).

### Level 2 — Linguistic features **[P0, cheap]**

Word count, sentence length, lexical diversity, **type-token ratio**, vocabulary richness.

**Use MTLD or MATTR, not raw TTR.** TTR is mechanically confounded by document length — comparing a 12-word comment's TTR to a 3,000-word transcript's is a known artifact, and it's the kind of thing that gets caught in Q&A. `lexical-diversity` package. Report raw TTR alongside only to show you know the difference.

### Level 3 — Readability **[P1, scoped]**

Flesch Reading Ease, Flesch-Kincaid, Gunning Fog, SMOG via `textstat`.

**Apply only to docs ≥100 words** (transcripts, news, LinkedIn long-form, OEM copy). Report N per corpus slice.

*Strategic payoff:* if OEM marketing copy reads at grade 14 while parent discourse sits at grade 7, that is a concrete, chartable communication finding for the playbook — and one of the few readability results in this corpus that means anything.

### Level 4 — Sentiment & emotion **[P0]**

Three-model ensemble, deliberately:

| Model | Role | Why |
|---|---|---|
| **VADER** | Social-media baseline | Built for exactly this register (emoji, caps, negation, intensifiers). Class-adjacent. |
| **TextBlob** | Polarity + **subjectivity** | Used in the Rashford class notebook — shows continuity with taught material. Subjectivity is a real feature for H1 (buyers factual, parents subjective). |
| **`cardiffnlp/twitter-roberta-base-sentiment-latest`** | Transformer accuracy | **[P1 / beyond-class]** Handles sarcasm and negation VADER misses. Free, Colab-fast on GPU. |

- **Aspect-Based Sentiment [P1, beyond-class, high value]:** document-level sentiment is nearly useless here — a comment reading *"love how quiet it is but I'd never put my kid on an unproven battery"* is neither positive nor negative. Split by aspect (safety, range, cost, comfort, charging, driver, brand) via dependency-window extraction around aspect anchors, score each window. **This is what actually answers "what consumers value or fear."** If one advanced technique survives triage, make it this one.
- **Disagreement flag:** where the three models disagree, flag for the manual audit set. Reporting inter-model agreement (Krippendorff's α) is Pillar-3 currency.

### Level 5 — Lexicon analysis **[P0 core / P1 extended]**

| Lexicon | Status | Use | Expected insight |
|---|---|---|---|
| **VADER** | **P0** | Social sentiment | Lifecycle polarity shift (Task 1.1) |
| **NRC Emotion (EmoLex)** | **P0** | 8 emotions + 2 sentiments | **Fear/trust/anticipation are the H1 story.** Expect fear↑ + trust↓ in parent voice, anticipation↑ in buyer voice. This is the single most quotable text chart in the deck. |
| **Empath** | **P1** | LIWC-substitute categories | Psychological/social framing without the licence |
| **Loughran-McDonald** | **P1, reference docs only** | Uncertainty, litigious, constraining | Institutional risk language around battery supply chain & financing → feeds strategy's risk section |
| ~~LIWC~~ / ~~Evaluative Lexicon~~ / ~~PARA~~ | **Cut** | — | See §0.2 |

**Lifecycle sentiment (Task 1.1, explicitly required):** sentiment × `lifecycle_stage`, per voice. The expected shape — pre-purchase anxiety → post-use pragmatism — is the assignment's own framing. Test it; if it doesn't hold, that's a finding, not a failure.

### Level 6 — Topic modeling **[P0 — required by Task 1.2]**

| Method | Role |
|---|---|
| **LDA** (`gensim`) | Baseline. Tune k by coherence (c_v) over k ∈ [4,16]. Satisfies "hyperparameter tuning" in the rubric — show the coherence curve, it's a free rubric hit. |
| **BERTopic** | **[Beyond-class]** Primary. Sentence-transformer embeddings + UMAP + HDBSCAN + c-TF-IDF. Vastly better on short noisy comments, where LDA is notoriously poor. |

- Name topics manually; auto-labels are unusable in a boardroom.
- **Topics × voice → the H1 test.** Compute **Jensen-Shannon divergence** between buyer and rider-parent topic distributions. One number, defensible, memorable.
- **Topics × mode → the H3 test**, via log-odds ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn) rather than raw frequency deltas — raw deltas are dominated by frequent words and will mislead.
- `topics_over_time` if timestamps allow (news cycle around the Telangana audit is a natural experiment).

### Level 7 — Custom composite metrics **[P1, beyond-class — this is our contribution]**

Each is a transparent, hand-auditable dictionary/rule metric. Explainability is the point: an examiner can check any score by hand.

| Metric | Definition | Hypothesis |
|---|---|---|
| **Safety Salience Index (SSI)** | Safety-lexicon tokens per 100 content tokens (CCTV, GPS, panic button, speed governor, fire, seatbelt, driver verification, overspeed, rash) | H2 |
| **Economics Salience Index (ESI)** | TCO, capex, subsidy, per-km, payback, fuel savings, lease, financing | H1, H2 |
| **Green Salience Index (GSI)** | Emission, pollution, carbon, ESG, sustainable, green | H1, H2 |
| **Reliability/Range Anxiety Index (RAI)** | Range, charge time, breakdown, stranded, battery life, mid-route | H1, H3 |
| **Quantitative Reasoning Density (QRD)** | Numerals + units (km, kWh, ₹, lakh, %, seats) per 100 tokens | **H1 — the sharpest single discriminator.** Buyers quantify; parents narrate. |
| **Certainty–Hedging Ratio (CHR)** | boosters (will, proven, guaranteed, definitely) ÷ hedges (might, maybe, seems, apparently, not sure) | H1 — operationalizes "parents see unproven technology" |
| **Evidence Density (ED)** | Attribution markers (according to, study, data, report, named orgs via NER) per 100 tokens | Pillar 2 |
| **Buyer–Rider Divergence Score (BRDS)** | JSD(topic dist. \| buyer ‖ topic dist. \| rider_parent), bootstrap CI | **H1 headline number** |

**Validation is mandatory, not optional.** Every dictionary metric must be validated against 100 hand-labelled docs (precision/recall of the trigger). An unvalidated custom index is a made-up number, and "we invented a metric" is a Q&A liability unless "and here's its precision" follows immediately.

### 4.7 Validation & bias protocol **[P0 — Pillar 3]**

1. **Gold set:** 200 docs, stratified by voice × segment, labelled by ≥2 team members. Report **Cohen's κ**; adjudicate disagreements.
2. **Sentiment validation:** accuracy/macro-F1 of VADER vs TextBlob vs RoBERTa on the gold set. Pick the winner *with evidence* — this is the rubric's "robust evaluation metrics."
3. **Voice classifier:** rules + TF-IDF logistic regression, report macro-F1 and confusion matrix. Do not hand-wave the tagging that H1 rests on.
4. **Bias register (§11):** platform bias (Reddit skews urban/male/English), language bias (English-only excludes exactly the parents most affected), selection bias (news over-indexes incidents), model bias (Western-trained sentiment models on Indian English), annotator bias (all four of us are MBA students, not parents).

---

## 5. Image Analytics Pipeline

**Input:** OEM product visuals, fleet/campus photos, dashboard UI/UX screenshots, charging infra, Statista charts, UGC.
**Output:** `image_features.parquet` — one row per image.
**Target N:** 150–300 images.

### Level 1 — Quality & low-level features **[P0 — "low-level" is named in Task 2.1]**

Brightness (HSV V mean), contrast (RMS/std), **blur (variance of Laplacian)**, resolution, aspect ratio, file size, colourfulness (Hasler-Süsstrunk), dominant palette (k-means, k=5), **HSV hue histogram**.

*Payoff:* green-hue share vs yellow-hue share across OEM assets tests whether "green" is being visually foregrounded to a school audience that (per H2) wants safety. Colour is cheap and it charts beautifully.

### Level 2 — Object detection **[P0 — class-taught]**

YOLOv8n/s (`ultralytics`) — as in the EA26 class notebook. COCO gives us: person, bus, truck, car, traffic light, backpack, chair, laptop.

- **Bounding-box frequencies are named in the rubric** ("bounding box frequencies") — surface them explicitly.
- Derived: **people-per-image**, **child-proxy presence** (person + backpack co-occurrence), bus-area share, environment class.
- **Honest limitation to state:** COCO has no class for CCTV camera, panic button, or speed governor. Do not pretend otherwise. Handle those via Level 3 + Level 5 (CLIP + OCR). Naming this gap and routing around it is stronger than a fine-tune we have no time to do.

### Level 3 — Scene understanding & zero-shot taxonomy **[P0 mixed / beyond-class]**

- **BLIP captioning** (`Salesforce/blip-image-captioning-base`) — class-taught (Image Summary Generator notebook). Captions also become **RAG-indexable text** (§7.2). The class notebook's own takeaway — BLIP fails on text-heavy images — is why Level 5 OCR exists; cite that continuity.
- **CLIP zero-shot classification [beyond-class, P0]:** `openai/clip-vit-base-patch32` scored against our custom taxonomy — *school bus exterior / staff bus exterior / bus interior / dashboard UI / charging depot / charging station / safety signage / marketing hero shot / chart-or-infographic*. This replaces manual labelling of 300 images and gives every image a strategic category in one pass. Validate on 50 hand-labelled images and report accuracy.

### Level 4 — Visual sentiment & framing **[P1]**

- Face detection (Haar cascade, as taught; or `retinaface` for quality) → **human presence**, face count, face-area share.
- **Facial expression [P1]:** `deepface`/`fer` for emotion on faces. **Caveat loudly:** FER models are trained largely on Western faces and are unreliable on Indian faces at small scale — this belongs in the bias register, and reporting it there is worth more than the feature itself.
- **Framing proxies** — brightness+saturation+warm-hue composite as a "positivity" proxy. Name it a proxy. Do not call it "visual sentiment" without the qualifier.

### Level 5 — OCR **[P0 — class-taught, feeds NLP]**

`pytesseract` (as taught) with the class notebook's grayscale + threshold preprocessing; upgrade to **EasyOCR [P1]** for angled/stylized marketing text where Tesseract collapses.

Extract: on-image marketing claims, safety-feature callouts, brand names, spec numbers, chart axis labels.

**Route all OCR text back into Phase 4** as documents with `modality=image_ocr`. This closes the loop the brief asks for and — more importantly — **OCR text from Statista charts is what makes those charts retrievable in Phase 7.**

### Level 6 — Custom visual metrics **[P1, beyond-class]**

| Metric | Method | Insight |
|---|---|---|
| **Visual Complexity** | Canny edge density + Shannon entropy of grayscale histogram | Cluttered marketing assets vs clean ones → engagement link |
| **Information Density** | OCR character count ÷ image area | Are OEMs spec-dumping where parents want reassurance? |
| **Branding Intensity** | Brand-token OCR area share + brand-colour mask share | Over/under-branding vs competitors |
| **Attention Hotspots** | `cv2.saliency.StaticSaliencySpectralResidual` → centroid + top-quartile mask | **Do safety cues fall inside the salient region, or are they decorative?** Overlay heatmaps on hero shots — the single most persuasive image slide available to us. |
| **Cognitive Load Estimate** | z-scored composite: complexity + info density + object count + palette entropy | Composite ad-legibility score for the playbook |
| **Safety-Cue Density** | CLIP safety-signage score + OCR safety-term hits + Level-2 objects | **H2's image-side measurement** |

**Composite-index honesty:** the Cognitive Load Estimate is a z-scored sum of correlated components with no external criterion. Present it as a *descriptive composite for ranking assets*, not a validated psychometric instrument. If it correlates with engagement, say "associated"; if it doesn't, report that too.

---

## 6. Video Analytics Pipeline

**Input:** OEM launches/walkthroughs, fleet inductions, influencer reviews, perspective content + public engagement metrics.
**Output:** `video_features.parquet` (one row per video) + `shot_features.parquet` (one row per shot).
**Target N:** 40–60 videos. Keep them short (≤10 min) — Colab CPU/GPU time is the binding constraint, not ambition.

### Level 1 — Acquisition & engagement metadata **[P0]**

`yt-dlp` — exactly as taught (EA26 + Rashford notebooks). Capture video + **metadata**: views, likes, comments, duration, upload date, channel, subscriber count.

**Task 2.2 requires mapping to "views, likes, shares." Engagement metadata is not optional — pull it at download time.** Videos downloaded without their stats have to be re-fetched, and that is a bad way to lose an hour on the last day.

### Level 2 — Shot segmentation **[P0 — upgraded from class]**

The class notebook detects shot boundaries from **jumps in YOLO object counts** (90th-percentile of `np.diff`). That's a clever improvisation but it's a proxy for a proxy: it fires on detection flicker and misses cuts between visually different scenes with equal object counts.

**Upgrade [beyond-class]:** **HSV colour-histogram correlation between consecutive frames**, cut when correlation drops below a tuned threshold — or `PySceneDetect`'s `ContentDetector`, which is the standard implementation. Then:

> **Validate the upgrade.** Hand-mark ground-truth cuts on 3 videos, report precision/recall for the class method vs the histogram method. This single comparison demonstrates (a) mastery of the taught technique, (b) a principled improvement, (c) evaluation rigor — three rubric pillars from one small experiment. Keep the class method in the notebook as the documented baseline; do not delete it.

### Level 3 — Structural & pacing metrics **[P0 — this is literally Task 2.2]**

Per video: **shot count, shots/minute, mean shot length, median shot length, shot-length variance, shot-length CV, cut rate in first 5s / first 15s, duration**.
Per shot: index, start, end, length, normalized position.

- **Pacing arc:** shot length vs normalized time → front-loaded / flat / accelerating. Classify each video's narrative shape.
- **Motion energy:** mean absolute frame difference per shot.
- **Hook composition:** what's on screen in the first 5 seconds (CLIP taxonomy from §5 L3 applied to the first keyframes) — *bus hero shot? child? charging? spec text?*

### Level 4 — Frame extraction **[P0 — class-taught]**

1 fps for dense analysis (class method), **plus one representative keyframe per shot** (mid-shot frame) for the CLIP/BLIP/OCR passes. Keyframe-per-shot is ~10× cheaper than 1 fps and better aligned to narrative structure — this is the difference between the video pipeline finishing and not finishing on Colab.

Run §5's Levels 2/3/5 over keyframes → **on-screen text density per second**, object timeline, safety-cue timeline.

### Level 5 — Speech-to-text **[P0 — upgraded from class]**

Class path was Vosk small (after Sphinx and Google API failed). **Upgrade [beyond-class]: `faster-whisper` (small/base)** — dramatically better WER on accented Indian English and noisy ad audio, gives segment timestamps free, and runs fine on a Colab T4. Vosk-small on Indian-English marketing audio produces transcripts too corrupt for downstream NLP; the class notebook's own transcript-cleaning hacks are evidence of this.

Output → **feeds Phase 4 as documents** with `modality=transcript`. These are long enough for readability (§4 L3) to be meaningful.

### Level 6 — Audio & speaker analytics **[P1]**

`librosa`, as taught (RMS energy in the Rashford notebook):
- **Speech rate (WPM)** from Whisper word timestamps — real, defensible, and it maps to "pacing."
- **Pause structure:** count and mean length of silences > 0.5s.
- RMS energy curve, energy variance, **music-vs-speech ratio** (spectral flatness / `librosa.effects.hpss`), tempo.

**On "confidence" and "engagement" from voice:** the brief asks for these. Vocal confidence is a *contested construct* even in the speech-science literature; inferring it from RMS on an OEM ad with a voiceover artist is not defensible. **Report the acoustic features (rate, pause ratio, energy variance) as what they are — pacing and delivery metrics — and let those map to engagement.** That is what Task 2.2 actually asks for, and it survives Q&A. Do not label a chart axis "confidence."

### Level 7 — Engagement modeling **[P0 — Task 2.2's payload]**

- **DV:** `engagement_rate = (likes + comments) / views`; secondary `log(views)`.
- **Confounds you must control or you will report nonsense:** channel subscriber count, video age (views accumulate), video type (ad vs review). A raw pacing↔views correlation is mostly measuring channel size.
- **Method:** Spearman ρ with bootstrap CIs; partial correlation controlling subscribers + age. **Ridge regression only as illustration** — with n≈50 and ~20 features, report it as descriptive with LOO-CV error, and say plainly that it is not a predictive model.
- **Multiple comparisons:** testing 20 features against engagement guarantees a spurious hit at p<0.05. Apply **Benjamini-Hochberg FDR** and say so. This costs one line and defuses the sharpest possible Q&A question.

### Level 8 — Multimodal fusion (video-internal) **[P1 — class-taught pattern]**

The Rashford notebook's pattern, applied to our domain: align **transcript sentiment × on-screen content (CLIP) × audio energy × shot boundaries** on one timeline (`broken_barh`, as taught).

*Payoff:* does the safety message land on a high-energy, high-salience beat, or is it buried in a mid-video lull? That's an ad-playbook prescription, not just a chart.

---

## 7. Multimodal RAG Pipeline **[P0 — Phase 3 / Module 4]**

*Absent from the original brief; required by the assignment and the rubric.*

**Input:** text corpus + Phase 4/5/6 outputs + **Statista charts** + **IEA market trends** + IFC/ITDP/UITP/CESL/MoveInSync/news PDFs.
**Output:** queryable engine + evaluated answers to executive queries + the evidence base for the 3-Year Strategy.

### Level 1 — Corpus assembly & chunking **[P0]**

| Source type | Handling |
|---|---|
| Reports/PDFs (IEA, ITDP, UITP, CESL, IFC) | `PyMuPDF` text extraction → **recursive chunking, ~500 tokens, 15% overlap** |
| PDF pages w/ charts | Render page to image → route to visual lane |
| Statista charts (images) | Visual lane + **OCR + BLIP caption → text lane** |
| News articles | `trafilatura` → chunk |
| Our own analysis outputs | **Findings memos as documents** — see below |
| Video transcripts | Segment-timestamped chunks (citable to a timecode) |

**Chunk metadata (this is what makes citation possible):** `chunk_id, source_id, source_type, page/timecode, url, modality, voice, date`.

> **The move that ties the whole project together:** write each Phase 4/5/6 finding as a short structured memo ("Parent-voice fear scores 2.3× buyer-voice; JSD = 0.41; n=1,240") and ingest those memos into the RAG corpus. Then executive queries retrieve *our own analysis* alongside IEA/Statista evidence, and the 3-Year Strategy genuinely is "relying on the outputs of your RAG pipeline" — as the assignment words it. Without this, the RAG system is a document search box bolted onto an unrelated analysis, which is exactly the failure mode Pillar 2 penalizes.

### Level 2 — Cross-modal embedding & indexing **[P0 — explicitly required]**

The assignment: *"utilizing cross-modal embeddings to index both textual data and visual layouts."* Two lanes, both required:

| Lane | Model | Indexes |
|---|---|---|
| **Text lane** | `sentence-transformers/all-MiniLM-L6-v2` (or `BAAI/bge-small-en-v1.5`) | Text chunks, OCR text, BLIP captions, transcript segments, findings memos |
| **Visual lane** | **`openai/clip-vit-base-patch32`** | Chart images, page renders, product images, keyframes — in CLIP's *shared* text-image space, so a text query retrieves an image directly |

Store in **FAISS** (`IndexFlatIP` on normalized vectors — at our corpus size, exact search; don't add IVF/HNSW complexity for a few thousand vectors and then have to defend it).

**The CLIP lane is the "cross-modal" in the requirement.** Captioning images and searching only text is *not* cross-modal retrieval — it's text retrieval over generated text. Do both; be able to articulate the difference in Q&A, because that is a likely question.

### Level 3 — Retrieval **[P0 core, P1 refinements]**

1. **Dense retrieval** — text lane, top-k=10. **[P0]**
2. **Cross-modal retrieval** — CLIP query→image, top-k=3. **[P0]**
3. **Hybrid: BM25 (`rank_bm25`) + dense, fused with Reciprocal Rank Fusion.** **[P1, beyond-class]** — dense embeddings miss exact terms like "Hiroi.EV" or "₹/km"; BM25 catches them. RRF needs no score normalization.
4. **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over top-20 → top-5. **[P1, beyond-class]** — the highest precision-per-line-of-code upgrade available.

### Level 4 — Generation **[P0]**

- **Generator:** Gemini API free tier (`gemini-2.0-flash`) or an open instruct model. Whichever you choose, **pin the model and temperature (0.2) in the notebook** — reproducibility is Pillar 3.
- **Prompt contract:** answer *only* from retrieved context; cite `chunk_id` inline for every claim; return "insufficient evidence in corpus" when retrieval is weak. **Demonstrate that refusal working on a deliberate out-of-corpus query** — showing the guardrail fire is worth more than ten successful answers.
- Assemble: query → hybrid retrieve → rerank → context pack (text chunks + retrieved chart images) → grounded answer + citations + thumbnails.

### Level 5 — Executive query testing **[P0 — Task 3.2 is its own graded task]**

Author **12–15 gold queries** spanning the four strategy areas the assignment names. Multi-layered, per the spec:

1. "What safety features do parents mention most, and which do OEMs actually show in school-bus marketing?" *(cross-modal: text + image)*
2. "How do stated barriers differ for electrifying employee *buses* vs employee *cars*?" *(H3)*
3. "What does the per-km economics evidence say about e-bus vs diesel for a fixed 60 km school route?" *(CESL + ITDP)*
4. "Which battery supply-chain risks appear in institutional reporting, and what mitigations are proposed?" *(strategy §3)*
5. "What video narrative structure correlates with peak engagement for school-segment content?" *(retrieves our own findings memo)*
6. "Where should charging infrastructure partnerships be prioritized, and on what evidence?"
7. "What is the India e-bus fleet trajectory per IEA/Statista, and what does it imply for a 3-year entry plan?" *(chart retrieval)*

**Evaluation harness [P0 — this is the rigor differentiator]:**

| Metric | Method |
|---|---|
| **Retrieval Hit@k, MRR** | Hand-label the gold chunk(s) per query |
| **Faithfulness** | Manual: is every sentence supported by a cited chunk? Score 0–2. Compare configs. |
| **Answer relevance** | 3-rater Likert, report mean + agreement |
| **Ablation** | dense-only vs hybrid vs hybrid+rerank — **a table showing retrieval improving across configs is the single most rubric-aligned artifact in the entire notebook** ("hyperparameter tuning" + "robust evaluation metrics" in one table) |
| **Failure analysis** | 3 queries where it fails, with diagnosis. Volunteering failures reads as confidence. |

`ragas` if it installs cleanly; **do not burn last-day hours fighting its dependency tree** — a hand-labelled 15-query eval is fully defensible and often more credible.

---

## 8. Fusion Layer & Hypothesis Testing **[P0]**

Join `text_features` + `image_features` + `video_features` on `asset_id` / `campaign_id` / `segment`.

| Test | Method | Output |
|---|---|---|
| **H1** | JSD(buyer ‖ rider_parent) over topics; QRD/CHR/NRC-fear gaps with bootstrap CIs | Divergence number + emotion radar |
| **H2** | SSI/GSI/ESI (text+image+video) → sentiment & engagement; Spearman + FDR | Frame-effectiveness ranking |
| **H3** | Barrier-topic prevalence, bus vs car; log-odds w/ Dirichlet prior | Barrier delta chart |
| **Cross-modal** | Do images with high Safety-Cue Density attract more positive comment sentiment? Correlate image features ↔ comment sentiment on the same post | Design→perception link (Task 2.1's "physical features correlate with perception") |

**Unit-of-analysis discipline.** Text is per-comment, images are per-image, videos are per-video. Correlating a per-video pacing metric against per-comment sentiment requires aggregating comments to video level *first* — mixing units silently inflates n and manufactures significance. Decide and document the unit for every test before running it.

---

## 9. Feature Inventory (Master Table)

**Legend:** P0 = submission-critical · P1 = if time · ★ = beyond classroom material

### 9.1 Text

| Feature | Modality | Purpose | Method | Expected Insight | Pri |
|---|---|---|---|---|---|
| Tokens/lemmas/n-grams | Text | Normalize | nltk, spaCy | Base layer | P0 |
| Word count, sentence length | Text | Effort/register | spaCy | Buyers write longer | P0 |
| MTLD / MATTR (+TTR) | Text | Lexical diversity | lexical-diversity | Expert vs lay vocabulary | P0 |
| Flesch RE, FK, Gunning Fog, SMOG | Text | Readability (≥100w) | textstat | **OEM copy over parents' heads** | P1 |
| VADER compound | Text | Social sentiment | vaderSentiment | Lifecycle polarity (Task 1.1) | P0 |
| TextBlob polarity/subjectivity | Text | Sentiment + objectivity | textblob | Buyers factual, parents subjective | P0 |
| RoBERTa sentiment ★ | Text | Accurate sentiment | cardiffnlp twitter-roberta | Catches sarcasm | P1 |
| Aspect-based sentiment ★ | Text | Per-aspect valence | Dependency windows + scorer | **"Quiet but unsafe" resolved** | P1 |
| NRC 8 emotions | Text | Emotional profile | NRCLex | **Fear↑ parents, anticipation↑ buyers** | P0 |
| Empath categories | Text | LIWC-substitute | empath | Psychosocial framing | P1 |
| LM uncertainty/litigious | Text | Institutional risk lang | pysentiment2 (ref docs only) | Supply-chain risk framing | P1 |
| LDA topics + coherence | Text | Baseline topics (Task 1.2) | gensim | Tuned-k curve = rubric hit | P0 |
| BERTopic topics ★ | Text | Primary topics | bertopic | Latent themes per voice | P0 |
| Topics-over-time ★ | Text | Narrative shift | bertopic | Telangana audit as shock | P1 |
| **SSI / ESI / GSI / RAI** ★ | Text | Frame salience | Validated dictionaries | **H2 core** | P1 |
| **QRD** ★ | Text | Quantitative reasoning | Regex numerals+units | **H1's sharpest discriminator** | P1 |
| **CHR** ★ | Text | Certainty vs hedging | Booster/hedge lexicon | "Unproven tech" quantified | P1 |
| **Evidence Density** ★ | Text | Reasoning quality | Attribution + spaCy NER | Who argues from data | P1 |
| **BRDS (JSD)** ★ | Text | Voice divergence | JSD + bootstrap | **H1 headline number** | P1 |
| voice / segment / mode / lifecycle | Text | Segmentation | Rules + TF-IDF LR (F1 reported) | **Everything depends on this** | P0 |

### 9.2 Image

| Feature | Modality | Purpose | Method | Expected Insight | Pri |
|---|---|---|---|---|---|
| Brightness, contrast, blur, resolution | Image | Quality (Task 2.1 low-level) | OpenCV, Laplacian var | Asset quality vs engagement | P0 |
| Colourfulness, dominant palette, hue hist | Image | Colour strategy | k-means, HSV | **Green vs safety-yellow framing** | P0 |
| Object classes + counts + **bbox freq** | Image | Content (rubric-named) | YOLOv8 | People/bus/infra composition | P0 |
| Child-proxy (person+backpack) | Image | School signal | YOLO co-occurrence | Are children shown? | P1 |
| BLIP caption | Image | Scene description | BLIP-base | **Also RAG-indexable text** | P0 |
| CLIP zero-shot taxonomy ★ | Image | Strategic category | CLIP ViT-B/32 | Auto-label 300 imgs, val. on 50 | P0 |
| Face count / area share | Image | Human framing | Haar / retinaface | Human-centred vs product-centred | P1 |
| Facial emotion ★ | Image | Visual affect | deepface/fer | **Bias-flagged** | P1 |
| OCR text + char count | Image | Embedded text | pytesseract → EasyOCR | **Feeds NLP; makes charts retrievable** | P0 |
| **Visual Complexity** ★ | Image | Clutter | Canny density + entropy | Clean vs busy → engagement | P1 |
| **Information Density** ★ | Image | Text load | OCR chars ÷ area | Spec-dumping detection | P1 |
| **Branding Intensity** ★ | Image | Brand presence | Logo OCR area + colour mask | Over/under-branding | P1 |
| **Attention Hotspots** ★ | Image | Saliency | cv2 spectral-residual | **Are safety cues salient or decorative?** | P1 |
| **Cognitive Load Estimate** ★ | Image | Composite legibility | z-scored composite | Ad-legibility ranking | P1 |
| **Safety-Cue Density** ★ | Image | Safety framing | CLIP + OCR + YOLO | **H2 image-side** | P1 |

### 9.3 Video

| Feature | Modality | Purpose | Method | Expected Insight | Pri |
|---|---|---|---|---|---|
| Duration, fps, resolution | Video | Basics | OpenCV | Format norms | P0 |
| **Views, likes, comments, subs, age** | Video | **Engagement DV (Task 2.2)** | yt-dlp metadata | **Non-negotiable — grab at download** | P0 |
| Shot boundaries ★ | Video | Segmentation | HSV-hist / PySceneDetect | **+ validated vs class method** | P0 |
| Shot count, shots/min, MSL, variance, CV | Video | **Pacing (Task 2.2)** | Derived | Fast cuts ↔ engagement? | P0 |
| First-5s cut rate | Video | Hook intensity | Derived | Hook design | P0 |
| Pacing arc shape | Video | Narrative structure | Shot length vs norm. time | Front-loaded vs flat | P1 |
| Motion energy | Video | Dynamism | Frame differencing | Static walkthrough vs cinematic | P1 |
| Keyframe objects / CLIP / OCR | Video | On-screen content | §5 over keyframes | **Safety-cue timeline** | P0 |
| On-screen text density/sec | Video | Legibility | OCR on keyframes | Text-dump ads | P1 |
| Whisper transcript ★ | Video | Speech→text | faster-whisper | **Feeds NLP; big WER win vs Vosk** | P0 |
| Speech rate (WPM) | Video | Delivery pacing | Whisper timestamps | Rushed spec-reading | P1 |
| Pause count / mean length | Video | Rhythm | Silence detection | Breathing room | P1 |
| RMS energy, variance | Video | Audio dynamics | librosa (class-taught) | Energy arc | P1 |
| Music-vs-speech ratio | Video | Ad style | Spectral flatness / HPSS | Emotional vs informational | P1 |
| Face presence timeline | Video | Human presence | Haar (class-taught) | Testimonial share | P1 |
| Multimodal event timeline | Video | Fusion viz | broken_barh (class-taught) | **Does safety land on a peak beat?** | P1 |

### 9.4 RAG

| Feature | Modality | Purpose | Method | Expected Insight | Pri |
|---|---|---|---|---|---|
| Text chunks + metadata | Multi | Retrieval units | PyMuPDF + recursive chunking | Citable evidence | P0 |
| Text embeddings | Text | Semantic index | MiniLM / bge-small | Semantic recall | P0 |
| **CLIP image embeddings** ★ | Image | **Cross-modal index (required)** | CLIP ViT-B/32 | **Text query → chart/image** | P0 |
| Findings memos | Multi | **Analysis→RAG bridge** | Our own outputs | **Strategy grounded in our analysis** | P0 |
| BM25 + RRF hybrid ★ | Text | Lexical+semantic | rank_bm25 + RRF | Catches "Hiroi.EV", "₹/km" | P1 |
| Cross-encoder rerank ★ | Text | Precision | ms-marco-MiniLM | Best precision/effort ratio | P1 |
| Hit@k, MRR, faithfulness | Eval | **Rigor** | 15 gold queries | **Ablation table** | P0 |
| Refusal guardrail | Gen | Anti-hallucination | Prompt contract | Demonstrated on OOC query | P0 |

---

## 10. Visualization Strategy

**System rules (Pillar 4):** one consistent palette across notebook/report/deck; semantic colour fixed globally (**safety = amber, economics = blue, green/ESG = green, reliability = red**) so a reader learns it once; every chart has a claim as its title ("Parents fear; buyers calculate" — not "Sentiment by voice"); every chart carries n and source.

### Text
- **Word clouds** *(class-taught, incl. the category-coloured variant from EA26)* — use the **category-coloured** version, per voice. Plain word clouds are decoration; category-coloured ones carry an argument.
- **Sentiment lifecycle timeline** — polarity across pre-purchase → purchase → post-use, split by voice. *(Task 1.1's picture.)*
- **NRC emotion radar** — buyer vs rider-parent overlaid. **The H1 money chart.**
- **Topic × voice heatmap** with the JSD number annotated. **The H1 evidence chart.**
- **Diverging bar** — barrier prevalence bus vs car (H3), log-odds scale.
- **Frame salience grouped bars** — SSI/ESI/GSI by voice and segment (H2).

### Image
- **Object/CLIP-category distribution** — what OEMs actually show.
- **Saliency heatmap overlays** on 3 hero shots, safety cues ringed. **Most persuasive single slide available.**
- **Colour palette strips** per brand — visual competitor audit.
- **Scatter:** Safety-Cue Density vs comment sentiment, points = thumbnails.

### Video
- **Engagement vs pacing scatter** — shots/min vs engagement rate, sized by views, with the fitted line **and its CI band** (the band is the honesty).
- **Pacing arc small-multiples** — shot length over normalized time, faceted high/low engagement.
- **Multimodal timeline** *(class-taught `broken_barh`)* — transcript sentiment + face presence + audio energy + shot cuts + safety-cue markers, one video, annotated.
- **Hook composition stacked bar** — first-5s content by engagement quartile.

### Multimodal / RAG
- **Cross-modal correlation matrix** — text/image/video features, FDR-masked (**show only what survives correction**).
- **Voice × modality comparison dashboard** — what each actor says, is shown, is sold.
- **RAG ablation table + Hit@k bars** — dense vs hybrid vs +rerank.
- **Query trace diagram** — one executive query → retrieved chunks + chart → grounded answer with citations. **This is the slide that proves the RAG is real and not a demo.**

---

## 11. Research Deliverables & Outputs

### 11.1 Insight Ledger (the spine — build it on day one)

One row per claim; nothing enters the report or deck without a row.

```
insight_id | claim | hypothesis | metric_id | value + CI | n | chart_ref
| confidence (high/med/low) | limitation | strategic_implication | report_§ | slide_#
```

This is the mechanism for Rubric Pillar 2. It also makes the report and deck near-mechanical to assemble at 2am, which — given the timeline — is the actual reason to build it first.

### 11.2 Output classes

| Class | Content |
|---|---|
| **Descriptive** | Corpus composition; what each voice talks about; what OEMs show; how ads are cut; sentiment distribution by lifecycle stage |
| **Comparative** | Buyer vs rider-parent (H1); safety vs green vs cost framing (H2); bus vs car (H3); brand-vs-brand visual audit; school vs employee segment |
| **Associative** *(not predictive — §0.2)* | Pacing ↔ engagement; safety-cue density ↔ sentiment; readability gap ↔ reception. Spearman + CI + FDR. |
| **Behavioural** | Where safety cues fall relative to attention hotspots; whether the safety beat lands on an energy peak; hedging as a trust proxy |
| **Strategic** | The 3-Year Strategy: positioning, infrastructure partnerships, battery supply-chain risk, marketing spend — each traced to a RAG query and an Insight Ledger row |

### 11.3 Bias & Limitations Register **[P0 — explicitly in Pillar 3]**

A standing table in both notebook and report:

| Bias | Manifestation | Mitigation | Residual risk |
|---|---|---|---|
| Platform | Reddit/LinkedIn skew urban, English, male | Multi-platform; report composition | Rural + non-English parents underrepresented — **the very people the school segment serves** |
| Language | English-only pipeline | Report excluded share; Hinglish flagged | Systematic voice exclusion |
| Selection | News over-indexes incidents/failures | Balance with OEM + neutral sources | Negativity inflation |
| Model | Sentiment/FER models Western-trained | Validate on our gold set; report accuracy | Miscalibration on Indian English/faces |
| Annotator | 4 MBA students, none are parents or fleet operators | Two-rater + κ | Framing bias in labels |
| Survivorship | Only published/successful videos exist | Acknowledge | Can't observe what failed |
| Small-n | ~50 videos | Associations + FDR, no prediction claims | Low power; real effects may be missed |

Volunteering this table is worth more marks than one more feature. Almost no team will write it.

### 11.4 Artifacts

1. **Colab notebooks** (§12) + GitHub repo, `requirements.txt` pinned, README, `data/` structure — **50%**
2. **Strategic Business Report** — Executive PDF, ≤20 pages — **20%**
3. **Boardroom Pitch** — live + Q&A — **30%**

---

## 12. Google Colab Blueprint

**Design rules.** Every notebook: (1) reads from and writes to Google Drive — never holds state in RAM across sessions (Colab *will* disconnect at the worst moment); (2) is independently runnable from its saved inputs; (3) ends by writing a `.parquet` + a findings memo `.md`; (4) opens with a markdown cell stating purpose, inputs, outputs, runtime. The rubric says "code cleanliness" — these four rules *are* code cleanliness at notebook scale.

**Runtime:** T4 GPU for NB02/03/04/05. Watch the free-tier budget: Whisper + YOLO + CLIP over 50 videos will eat it. **Cache every model output to Drive immediately** — never recompute an embedding you've already paid for.

```
/TIVA_Group1/
  data/raw/{text,images,videos,reference_pdfs,statista}/
  data/interim/{frames,keyframes,audio,transcripts}/
  data/processed/{text_features,image_features,video_features,shot_features}.parquet
  data/registry/asset_registry.csv          ← §3.3 contract
  data/gold/{sentiment_gold,voice_gold,rag_gold_queries,shot_gold}.csv
  data/rag/{chunks.parquet,faiss_text.index,faiss_clip.index,memos/}
  outputs/{figures,tables,insight_ledger.csv}
  notebooks/
```

| # | Notebook | Purpose | Inputs | Outputs | Key libraries | Depends on | Pri |
|---|---|---|---|---|---|---|---|
| **00** | **Setup & Data Ingestion** | Acquire everything; build the registry; stamp provenance | Source URLs/APIs | `raw/*`, `asset_registry.csv` | yt-dlp, praw, trafilatura, requests, pandas | — | **P0** |
| **01** | **Text Analytics** | L1–L7: clean → sentiment → topics → lexicons → custom metrics | raw text, registry, transcripts (NB03), OCR (NB02) | `text_features.parquet`, memos, figs | nltk, spacy, vaderSentiment, textblob, NRCLex, empath, textstat, lexical-diversity, gensim, **bertopic**, transformers | 00 (+03,02 for reruns) | **P0** |
| **02** | **Image Analytics** | Quality → YOLO → BLIP → **CLIP** → OCR → custom metrics | raw images, keyframes (NB03) | `image_features.parquet`, `ocr_text.csv`→NB01, memos | opencv, ultralytics, transformers (BLIP, **CLIP**), pytesseract, easyocr, scikit-image, deepface | 00 | **P0** |
| **03** | **Video Analytics** | Shots → keyframes → **Whisper** → pacing → audio → engagement | raw videos + metadata | `video_features.parquet`, `shot_features.parquet`, `transcripts/`→NB01, `keyframes/`→NB02 | opencv, **PySceneDetect**, **faster-whisper**, librosa, pydub, moviepy, ultralytics | 00 | **P0** |
| **04** | **Multimodal RAG** | Chunk → dual-index → hybrid retrieve → rerank → generate → **evaluate** | reference PDFs, Statista, text corpus, **memos from 01/02/03** | `chunks.parquet`, 2 FAISS indices, `rag_eval_results.csv`, query traces | PyMuPDF, sentence-transformers, **CLIP**, faiss-cpu, rank_bm25, cross-encoder, google-generativeai | 00, 01, 02, 03 | **P0** |
| **05** | **Fusion & Hypothesis Testing** | Join modalities; test H1/H2/H3; correlations + FDR | all `*_features.parquet` | `fusion_table.parquet`, test results, `insight_ledger.csv` | pandas, scipy, statsmodels, pingouin, scikit-learn | 01, 02, 03 | **P0** |
| **06** | **Visualization** | Every report/deck figure, one palette, claim-titled | fusion table, all features, ledger | `outputs/figures/*.png` @300dpi | matplotlib, seaborn, plotly, wordcloud, adjustText | 05 | **P0** |
| **07** | **Reporting & Strategy Synthesis** | Ledger → report skeleton; RAG queries → strategy sections | ledger, figures, RAG answers | report draft, slide spine, appendix tables | pandas, jinja2, IPython.display | 04, 05, 06 | **P1** |

**Notebook count note:** the brief asked for 7; this is 8 because RAG needs its own. If the runway forces a merge, merge **06 into 05** — never merge 04 into anything. RAG is 1/3 of the technical grade and needs to be visibly, separately excellent.

**Circular dependency, resolved:** NB01 needs transcripts (NB03) and OCR (NB02), but NB02/03 don't need NB01. Run **00 → 03 → 02 → 01 → 04 → 05 → 06 → 07**. Do not try to run 01 first and "add transcripts later" — you will run it twice and the second run is the one that gets skipped at 3am.

---

## 13. Implementation Roadmap

### 13.1 The timeline problem — read before planning anything

**The assignment PDF states Final Submission 18 July EOD. Today is 17 July.** Roughly 24 hours, ×4 people, ≈ 40–50 usable person-hours after sleep. The full P0+P1 blueprint above is a 120-hour design.

**Therefore: P0 only.** Every P1 is a bonus that must not be started until every P0 is done. The most common way this project fails is not weak analysis — it's an unfinished RAG section at midnight because someone was tuning BERTopic at 8pm.

**Please confirm the deadline before executing this plan.** If it has moved, §13.3 is the schedule. If it hasn't, §13.2 is.

### 13.2 P0 sprint (~24h, 4 people, parallel)

| Slot | Aathira | Harsh | Prakhya | Yedhu |
|---|---|---|---|---|
| **H+0–4** | Text collection (Reddit/news/YT comments) | Video collection **+ engagement metadata** | Image collection + Statista charts + IEA PDF | Registry schema, Drive scaffold, repo, `requirements.txt` |
| **H+4–6** | **All four: tag 200-doc gold set + registry `voice`/`segment`/`mode`/`lifecycle`.** Highest-leverage 2 hours of the project — everything downstream is conditioned on these tags. |
| **H+6–12** | NB01: clean, VADER+TextBlob, NRC, LDA+BERTopic | NB03: PySceneDetect, keyframes, Whisper, pacing | NB02: quality, YOLO, BLIP, CLIP, OCR | NB04: chunk, dual index, retrieval |
| **H+12–16** | Lifecycle sentiment, topic×voice, JSD | Engagement correlations + FDR | Frame scores, saliency on 3 heroes | NB04: generation + **15-query eval + ablation** |
| **H+16–19** | NB05 fusion + H1/H2/H3 | NB06 figures | NB06 figures | Findings memos → RAG re-index |
| **H+19–22** | Report §§1–3 | Report §§4–5 | Deck | Strategy §§ from RAG outputs |
| **H+22–24** | **All four: Insight Ledger reconciliation, bias register, restart-and-run-all every notebook, repo tidy, submit.** |

**Non-negotiables — cut anything else before these:**
1. Registry tagging quality
2. Video engagement metadata captured at download
3. RAG eval table (15 queries + ablation)
4. Findings memos ingested into RAG
5. Bias & limitations register
6. Restart-and-run-all on every notebook before submission

### 13.3 Full roadmap (if the deadline has moved)

| Phase | Inputs | Outputs | Libraries | Effort |
|---|---|---|---|---|
| **A. Data Collection** | Source list, APIs | raw/, registry | yt-dlp, praw, trafilatura | 12–16 ph |
| **B. Cleaning & Tagging** | raw/ | interim/, gold sets, tagged registry | pandas, spacy, langdetect, datasketch | 10–14 ph |
| **C. Feature Engineering** | interim/ | 4 feature parquets | full stack | 24–32 ph |
| **D. Analysis** (fusion + RAG + eval) | feature tables | tests, RAG, ledger | scipy, statsmodels, faiss, sentence-transformers | 20–28 ph |
| **E. Visualization** | ledger, features | figures | matplotlib, seaborn, plotly | 10–14 ph |
| **F. Reporting** | ledger, figures, RAG | report, deck | — | 14–18 ph |
| **G. Final Submission** | everything | repo, PDF, deck, rehearsal | — | 8–10 ph |
| | | | **Total** | **98–132 ph** |

### 13.4 Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Data volume too low for H1** | High | Set the floor now (≥200 docs/voice). If unmet, **report as exploratory** — don't quietly ship an underpowered claim as a finding. |
| **Colab GPU quota exhausted** | High | Cache every model output to Drive on first compute. Cap videos ≤10 min. Two Google accounts. |
| **Whisper too slow on 50 videos** | Med | `faster-whisper` base, not large. Transcribe the top-20 by engagement first — they matter most for Task 2.2 anyway. |
| **RAG generator API fails/rate-limits** | High | Pre-test the key. Fallback: local Flan-T5. **Retrieval + eval is gradeable even if generation degrades** — build retrieval first. |
| **LinkedIn ToS** | Med | Manual collection only; document it; state it in the report. |
| **Notebook state rot** | High | Restart-and-run-all before submission. A notebook that only works in your session is a notebook that doesn't work. |
| **Statista paywall** | Med | Free chart previews are usually sufficient; else substitute IEA Global EV Outlook figures. **Don't skip charts — the assignment names them.** |

---

## 14. Where This Sits Against the Course Material

### Class-taught, retained and visibly credited
yt-dlp acquisition · OpenCV frame extraction · YOLO object detection + class counts + objects-per-frame trend · storyboard/busy/shot-change frames · pytesseract OCR + regex cleaning · BLIP captioning · Haar face detection · librosa RMS energy · pydub conversion · TextBlob polarity/subjectivity · nltk + wordcloud (incl. the category-coloured variant) · `broken_barh` multimodal timelines · the EA26 "media strategy + AI/ML + consumer behaviour + data storytelling" framing.

### Beyond-class (the ~15% delta)
| Upgrade | Replaces | Why it's defensible |
|---|---|---|
| **PySceneDetect / HSV-hist cuts** | Object-count-diff shot detection | Standard method; **validated against the class baseline on hand-marked ground truth** |
| **faster-whisper** | Vosk-small / Sphinx | Large WER win on Indian-accented English; class notebook's own cleanup hacks show the need |
| **BERTopic** | LDA only | Purpose-built for short noisy text; LDA retained as tuned baseline |
| **CLIP zero-shot + CLIP index** | Manual labels; caption-only search | **Cross-modal retrieval is explicitly required by Task 3.1** |
| **Hybrid BM25+dense w/ RRF, cross-encoder rerank** | Naive dense retrieval | Measured ablation gains |
| **Aspect-based sentiment** | Document-level sentiment | Resolves mixed-valence comments — the dominant form in this corpus |
| **Custom validated indices** (SSI/QRD/CHR/BRDS, saliency, cognitive load) | — | Original, explainable, hypothesis-linked, hand-auditable |
| **FDR correction, bootstrap CIs, partial correlation, κ, Dirichlet log-odds** | Raw correlations | Pillar 3 |

Each upgrade keeps the class method visible as a baseline. That's deliberate: it demonstrates mastery *and* improvement, which scores on two pillars where a silent replacement scores on neither — and it means every advanced choice has a ready Q&A answer ("we tried the taught method, measured it, and here's what it missed").

---

## 15. Open Questions For The Team

1. **Is the 18 July deadline real?** It determines whether §13.2 or §13.3 is the plan. Everything else is downstream of this.
2. **Which generator has a working API key today?** Test before H+6. This is a silent, discoverable-too-late blocker.
3. **Did faculty teach a specific "PARA" lexicon?** If yes, name it and it goes back in. If no, it stays cut.
4. **Statista access** through the SPJIMR library? Determines chart sourcing.
5. **Whose Google account hosts the Drive?** Notebook paths hard-code it; decide once, at H+0.
6. **How is `voice` assigned to a YouTube comment** where the commenter's role is unstated? Proposed default: `unknown`, excluded from H1 rather than guessed. Guessing inflates n and corrupts the headline number.
```
