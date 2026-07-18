"""
MILESTONE 2 - Tier A/B/C text extraction
Tier A: news/corporate HTML pages  -> requests + trafilatura (BS4 fallback)
Tier B: direct PDFs                -> requests + PyMuPDF
Tier C: data.gov.in API + WRI dataset CSV

Output: data/raw/web_scrape_results.jsonl  (one record per source)
        data/raw/failed_sources.json
"""

import json
import os
import random
import re
import time
from datetime import datetime, timezone

import fitz  # PyMuPDF
import requests
import trafilatura
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

UA_RESEARCH = ("Mozilla/5.0 (compatible; TIVA-Academic-Research/1.0; "
               "+mailto:pgp25.harshkumar@spjimr.org)")
UA_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

DATA_GOV_SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"

SOURCES = [
    # ---- TIER A: HTML ----
    dict(id="pib_ebus_programme", name="National Electric Bus Programme press release",
         org="PIB / MoRTH", category="Market & Fleet Reference", method="html",
         url="https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2036676",
         relevance=8, reliability=9, authenticity=85),
    dict(id="telangana_today_fitness", name="Telangana school-bus fitness inspections (June 2026 reopening)",
         org="Telangana Today / TG Transport Dept", category="Text - Regulatory Scrutiny", method="html",
         url="https://telanganatoday.com/school-bus-fitness-inspections-intensify-in-hyderabad-ahead-of-june-12-reopening",
         relevance=10, reliability=9, authenticity=95),
    dict(id="tribune_delhi_flagoff", name="Delhi CM flags off 24 electric school buses (Aug 2025)",
         org="The Tribune / Delhi Govt", category="Text - News: School EV Launches", method="html",
         url="https://www.tribuneindia.com/news/delhi/cm-flags-off-24-electric-school-buses-to-curb-delhis-pollution",
         relevance=9, reliability=8, authenticity=85),
    dict(id="tribune_gd_goenka", name="GD Goenka Vasant Kunj first private e-school-bus pilot",
         org="The Tribune / VECV", category="Text - News: School EV Pilots", method="html",
         url="https://www.tribuneindia.com/news/delhi/school-starts-pilot-project/amp",
         relevance=10, reliability=8, authenticity=90),
    dict(id="dh_bmtc_ride_quality", name="BMTC AC e-bus rollout: passenger ride-quality complaints",
         org="Deccan Herald", category="Text - Rider Sentiment (Proxy)", method="html",
         url="https://www.deccanherald.com/india/karnataka/bengaluru/bmtc-to-launch-ac-e-bus-services-to-bengaluru-airport-after-june-15-3578338",
         relevance=9, reliability=8, authenticity=85),
    dict(id="dh_bmtc_image", name="Minister: e-bus issues damaging BMTC's image (Dec 2025)",
         org="Deccan Herald / Karnataka Transport Ministry", category="Text - Reliability & Operations", method="html",
         url="https://www.deccanherald.com/amp/story/india%2Fkarnataka%2Fbengaluru%2Fissues-with-electric-buses-damaging-bmtcs-image-says-minister-ramalinga-reddy-3827108",
         relevance=8, reliability=9, authenticity=85),
    dict(id="dh_moveinsync_commute", name="MoveInSync GCC commute report: Bengaluru workforce & EV trips",
         org="Deccan Herald / MoveInSync", category="Text - Employee Commute Data", method="html",
         url="https://www.deccanherald.com/amp/story/india%2Fkarnataka%2Fbengaluru%2Fbengalurus-tech-workforce-spends-50-mins-commuting-to-work-one-way-3495578",
         relevance=9, reliability=8, authenticity=85),
    dict(id="evreporter_school_ev", name="The Usecase for School Buses To Go Electric",
         org="EVreporter / Volttic EV Charging", category="Text - Opinion: School EV Economics", method="html",
         url="https://evreporter.com/the-big-yellow-school-bus-time-to-go-electric/",
         relevance=9, reliability=8, authenticity=85),
    dict(id="trucks91_buying_guide", name="Electric School Buses in India 2025: Prices, Brands & How to Choose",
         org="91Trucks", category="Text - Buyer Guide: School EVs", method="html",
         url="https://www.91trucks.com/news/electric-school-buses-in-india-2025-prices-brands-how-to-choose",
         relevance=8, reliability=7, authenticity=75),
    dict(id="uber_corporate_shuttle", name="Uber Corporate Shuttle launch in India",
         org="Uber Newsroom", category="Text - Corporate Shuttle Market", method="html",
         url="https://www.uber.com/en-IN/newsroom/as-indian-workers-tentatively-return-to-the-office-uber-launches-corporate-shuttle-service",
         relevance=7, reliability=8, authenticity=75),
    dict(id="yourstory_corporate_ev", name="Corporates shift to greener EVs for employee commute but hiccups stay",
         org="YourStory", category="Text - Corporate EV Adoption", method="html",
         url="https://yourstory.com/2023/09/corporates-shift-to-greener-evs-for-employee-commute",
         relevance=9, reliability=8, authenticity=85),
    dict(id="moveinsync_site", name="MoveInSync employee transport platform (6,500+ vehicles, ~925 EVs)",
         org="MoveInSync", category="Text - Employee Transport Ops", method="html",
         url="https://moveinsync.com/in",
         relevance=9, reliability=8, authenticity=85),
    dict(id="ifc_jbm_greencell", name="IFC partnership with JBM Group & GreenCell Mobility",
         org="IFC (World Bank Group)", category="Text - Fleet Scale & Financing", method="html",
         url="https://www.ifc.org/en/pressroom/2025/ifc-partners-with-jbm-group-and-greencell-mobility-to-accelerate-electric-bus-depl",
         relevance=9, reliability=9, authenticity=90),
    # ---- TIER B: PDFs ----
    dict(id="itdp_status_ebuses", name="Status of E-buses in India",
         org="ITDP India", category="Market & Fleet Reference", method="pdf",
         url="https://www.itdp.in/wp-content/uploads/2022/10/Status-of-E-buses-in-India.pdf",
         relevance=8, reliability=9, authenticity=85),
    dict(id="itdp_rollout_guidance", name="Guidance for Electric Bus Rollout in Indian Cities",
         org="ITDP India", category="Text - E-bus Rollout Guidance", method="pdf",
         url="https://www.itdp.in/wp-content/uploads/2022/10/Guidance-for-e-Bus-Rollout-in-Indian-Cities.pdf",
         relevance=8, reliability=9, authenticity=85),
    dict(id="siam_roadmap", name="Roadmap for Accelerated Adoption of E-buses in India",
         org="SIAM", category="Market & Fleet Reference", method="pdf",
         url="https://www.siam.in/uploads/Know/AnnualReport/10999file_Roadmap-for-Accelerated-Adoption-of-E-buses-in-India-Feb2026.pdf",
         relevance=9, reliability=9, authenticity=90),
    dict(id="uncrd_ashok_leyland", name="Techno-commercial options for e-bus electrification (UNCRD EST Forum)",
         org="Ashok Leyland / UN DESA", category="Text - OEM Segment Forecast", method="pdf",
         url="https://sdgs.un.org/sites/default/files/2020-12/UNCRD_13th%20EST%20Forum_Plenary%20Session%203-Presentation%203-%20Athmanathan%28Ashok%20Leyland%20%29%20-Final-6Nov.pdf",
         relevance=9, reliability=9, authenticity=90),
    dict(id="cesl_grand_challenge", name="CESL Grand Challenge Case Study (e-bus vs diesel per-km)",
         org="CESL / Convergence Energy", category="Text - E-bus Economics", method="pdf",
         url="https://www.convergence.co.in/public/images/electric_bus/Grand-Challenge-Case-Study-Final-Web-Version.pdf",
         alt_urls=["https://www.convergence.co.in/public/images/electricbus/Grand-Challenge-Case-Study-Final-Web-Version.pdf"],
         relevance=8, reliability=9, authenticity=85),
    dict(id="uitp_performance_eval", name="Performance Evaluation of Electric Bus from Six Indian Cities",
         org="UITP India", category="Market & Fleet Reference", method="uitp",
         url="https://www.uitp.org/publications/performance-evaluation-framework-for-electric-buses-in-india/",
         relevance=8, reliability=9, authenticity=85),
    # ---- TIER C ----
    dict(id="datagov_vahan_ebuses", name="State/UT-wise Number of Electric Buses (VAHAN)",
         org="MoRTH / data.gov.in", category="Market & Fleet Reference", method="datagov",
         url="https://www.data.gov.in/resource/stateut-wise-number-electric-buses-vahan-portal-22-11-2024",
         relevance=9, reliability=9, authenticity=90),
    dict(id="datagov_srtu_fleet", name="State/UT wise Total Bus Fleet and Buses in Public Sector (SRTUs)",
         org="data.gov.in / MoRTH", category="Market & Fleet Reference", method="datagov",
         url="https://www.data.gov.in/catalog/stateut-wise-total-bus-fleet-and-buses-public-sector-srtus",
         relevance=7, reliability=9, authenticity=80),
    dict(id="wri_esb_dashboard", name="Electric School Bus Data Dashboard (underlying dataset)",
         org="WRI / Electric School Bus Initiative", category="International School Bus Reference", method="wri",
         url="https://datasets.wri.org/dataset/electric_school_bus_adoption",
         relevance=6, reliability=9, authenticity=75),
]


def fetch(url, ua=UA_RESEARCH, timeout=40, stream=False):
    return requests.get(url, headers={"User-Agent": ua, "Accept-Language": "en"},
                        timeout=timeout, allow_redirects=True, stream=stream)


def fetch_with_fallback(url):
    """Try research UA first, then browser UA on 4xx."""
    try:
        r = fetch(url)
        if r.status_code < 400:
            return r
    except requests.RequestException:
        pass
    time.sleep(1.5)
    return fetch(url, ua=UA_BROWSER)


def extract_html(resp):
    downloaded = resp.text
    text = trafilatura.extract(downloaded, include_comments=False,
                               favor_recall=True, url=resp.url)
    meta = trafilatura.extract_metadata(downloaded)
    title = meta.title if meta and meta.title else None
    date = meta.date if meta and meta.date else None
    if not text:
        soup = BeautifulSoup(downloaded, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n", strip=True))
        title = title or (soup.title.string.strip() if soup.title and soup.title.string else None)
    return title, date, text


def extract_pdf(content, source_id):
    pdf_path = os.path.join(RAW, f"{source_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(content)
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages), len(pages)


def handle_uitp(src):
    """UITP landing page -> locate the actual PDF link, then extract."""
    r = fetch_with_fallback(src["url"])
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    pdf_link = None
    for a in soup.find_all("a", href=True):
        if ".pdf" in a["href"].lower():
            pdf_link = a["href"]
            break
    if not pdf_link:
        # fall back to landing-page text so the source is not empty
        title, date, text = extract_html(r)
        return dict(title=title, date=date, text=text, note="PDF link not found; landing page text only")
    if pdf_link.startswith("/"):
        pdf_link = "https://www.uitp.org" + pdf_link
    r2 = fetch_with_fallback(pdf_link)
    r2.raise_for_status()
    text, npages = extract_pdf(r2.content, src["id"])
    return dict(title=src["name"], date=None, text=text, note=f"PDF ({npages} pages) via {pdf_link}")


def handle_datagov(src):
    """Resource page -> find resource UUID -> open API with public sample key."""
    r = fetch_with_fallback(src["url"])
    r.raise_for_status()
    uuids = re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", r.text)
    if not uuids:
        raise RuntimeError("No resource UUID found on page (JS-rendered) - needs manual download")
    last_err = None
    for uuid in dict.fromkeys(uuids):
        api = (f"https://api.data.gov.in/resource/{uuid}"
               f"?api-key={DATA_GOV_SAMPLE_KEY}&format=json&limit=200")
        try:
            rj = fetch(api)
            data = rj.json()
            if data.get("records"):
                lines = [json.dumps(rec, ensure_ascii=False) for rec in data["records"]]
                text = (f"Dataset: {data.get('title', src['name'])}\n"
                        f"Records ({len(lines)}):\n" + "\n".join(lines))
                return dict(title=data.get("title", src["name"]), date=data.get("updated_date"),
                            text=text, note=f"data.gov.in API, uuid={uuid}, {len(lines)} records")
        except Exception as e:  # try next uuid
            last_err = e
    raise RuntimeError(f"No UUID returned records (last error: {last_err})")


def handle_wri(src):
    """WRI dataset page -> first CSV resource -> header + sample rows as text."""
    r = fetch_with_fallback(src["url"])
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    csv_link = None
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".csv") or "download" in a["href"].lower():
            csv_link = a["href"]
            if csv_link.startswith("/"):
                csv_link = "https://datasets.wri.org" + csv_link
            if csv_link.lower().endswith(".csv"):
                break
    if not csv_link:
        raise RuntimeError("No CSV resource link found on WRI dataset page")
    r2 = fetch_with_fallback(csv_link)
    r2.raise_for_status()
    csv_path = os.path.join(RAW, f"{src['id']}.csv")
    with open(csv_path, "wb") as f:
        f.write(r2.content)
    lines = r2.content.decode("utf-8", errors="replace").splitlines()
    text = "\n".join(lines[:400])
    return dict(title=src["name"], date=None, text=text,
                note=f"CSV saved ({len(lines)} lines) from {csv_link}")


def main():
    results, failures = [], []
    for i, src in enumerate(SOURCES, 1):
        print(f"[{i}/{len(SOURCES)}] {src['id']} ({src['method']}) ... ", end="", flush=True)
        record = dict(doc_id=src["id"], source_name=src["name"], organization=src["org"],
                      category=src["category"], url=src["url"], scrape_method=src["method"],
                      scraped_at=datetime.now(timezone.utc).isoformat(),
                      relevance_score=src["relevance"], reliability_score=src["reliability"],
                      authenticity_score=src["authenticity"])
        try:
            if src["method"] == "html":
                r = fetch_with_fallback(src["url"])
                r.raise_for_status()
                title, date, text = extract_html(r)
                record.update(title=title, published_date=date, text=text, final_url=r.url)
            elif src["method"] == "pdf":
                urls = [src["url"]] + src.get("alt_urls", [])
                content, used = None, None
                for u in urls:
                    try:
                        r = fetch_with_fallback(u)
                        if r.status_code < 400 and r.content[:5] == b"%PDF-":
                            content, used = r.content, u
                            break
                    except requests.RequestException:
                        continue
                if content is None:
                    raise RuntimeError(f"All URL variants failed: {urls}")
                text, npages = extract_pdf(content, src["id"])
                record.update(title=src["name"], published_date=None, text=text,
                              final_url=used, note=f"PDF, {npages} pages")
            elif src["method"] == "uitp":
                record.update(final_url=src["url"], published_date=None, **handle_uitp(src))
            elif src["method"] == "datagov":
                out = handle_datagov(src)
                record.update(final_url=src["url"], published_date=out.pop("date"), **out)
            elif src["method"] == "wri":
                record.update(final_url=src["url"], published_date=None, **handle_wri(src))

            wc = len((record.get("text") or "").split())
            record["word_count"] = wc
            if wc < 50:
                raise RuntimeError(f"Extraction produced only {wc} words - treating as failure")
            record["status"] = "ok"
            results.append(record)
            print(f"OK ({wc} words)")
        except Exception as e:
            record.update(status="failed", error=str(e)[:300], text=None, word_count=0)
            failures.append(record)
            print(f"FAILED: {str(e)[:120]}")
        time.sleep(random.uniform(2.0, 4.0))

    with open(os.path.join(RAW, "web_scrape_results.jsonl"), "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(RAW, "failed_sources.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    print(f"\nDONE: {len(results)} ok, {len(failures)} failed")
    print(f"Results: {os.path.join(RAW, 'web_scrape_results.jsonl')}")


if __name__ == "__main__":
    main()
