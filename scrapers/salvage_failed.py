"""
MILESTONE 2 - salvage pass for the 4 Tier A/C failures.
- Uber newsroom (406 to plain requests)  -> headless Selenium page text
- data.gov.in x2 (JS-rendered)           -> Selenium render, regex UUID, open API
- WRI ESB dataset (page moved)           -> CKAN package_search API

Appends successful records to data/raw/web_scrape_results.jsonl and
rewrites data/raw/failed_sources.json with whatever is still dead.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import requests
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
DATA_GOV_SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    return webdriver.Chrome(options=opts)


def base_record(doc_id, name, org, category, url, method, rel, reli, auth):
    return dict(doc_id=doc_id, source_name=name, organization=org, category=category,
                url=url, scrape_method=method,
                scraped_at=datetime.now(timezone.utc).isoformat(),
                relevance_score=rel, reliability_score=reli, authenticity_score=auth)


def salvage_uber(driver):
    rec = base_record("uber_corporate_shuttle", "Uber Corporate Shuttle launch in India",
                      "Uber Newsroom", "Text - Corporate Shuttle Market",
                      "https://www.uber.com/en-IN/newsroom/as-indian-workers-tentatively-return-to-the-office-uber-launches-corporate-shuttle-service",
                      "selenium", 7, 8, 75)
    driver.get(rec["url"])
    time.sleep(6)
    html = driver.page_source
    text = trafilatura.extract(html, include_comments=False, favor_recall=True)
    if not text or len(text.split()) < 50:
        raise RuntimeError("Selenium render still yielded no article text")
    rec.update(title=driver.title, published_date=None, text=text,
               final_url=driver.current_url, word_count=len(text.split()), status="ok",
               note="Recovered via headless Selenium after 406 to requests")
    return rec


def datagov_records_from_page(driver, page_url):
    driver.get(page_url)
    time.sleep(8)
    uuids = list(dict.fromkeys(re.findall(UUID_RE, driver.page_source)))
    for uuid in uuids:
        api = (f"https://api.data.gov.in/resource/{uuid}"
               f"?api-key={DATA_GOV_SAMPLE_KEY}&format=json&limit=200")
        try:
            data = requests.get(api, timeout=30).json()
        except Exception:
            continue
        if data.get("records"):
            return uuid, data
    raise RuntimeError(f"{len(uuids)} UUIDs found, none returned records")


def salvage_datagov(driver, doc_id, name, url, category, rel, reli, auth):
    rec = base_record(doc_id, name, "MoRTH / data.gov.in", category, url,
                      "datagov-api", rel, reli, auth)
    uuid, data = datagov_records_from_page(driver, url)
    lines = [json.dumps(r, ensure_ascii=False) for r in data["records"]]
    text = (f"Dataset: {data.get('title', name)}\n"
            f"Records ({len(lines)}):\n" + "\n".join(lines))
    rec.update(title=data.get("title", name), published_date=data.get("updated_date"),
               text=text, final_url=url, word_count=len(text.split()), status="ok",
               note=f"data.gov.in API via Selenium-rendered page, uuid={uuid}, {len(lines)} records")
    return rec


def salvage_wri():
    rec = base_record("wri_esb_dashboard", "Electric School Bus Data Dashboard (underlying dataset)",
                      "WRI / Electric School Bus Initiative", "International School Bus Reference",
                      "https://datasets.wri.org", "ckan-api", 6, 9, 75)
    search = requests.get(
        "https://datasets.wri.org/api/3/action/package_search",
        params={"q": "electric school bus", "rows": 5}, timeout=30,
        headers={"User-Agent": "TIVA-Academic-Research/1.0"})
    search.raise_for_status()
    js = search.json()
    results = js.get("result", {}).get("results", [])
    if not results:
        raise RuntimeError("CKAN search returned no packages")
    pkg = results[0]
    csv_url = None
    for res in pkg.get("resources", []):
        if (res.get("format", "").lower() == "csv") or str(res.get("url", "")).lower().endswith(".csv"):
            csv_url = res["url"]
            break
    if not csv_url:
        raise RuntimeError(f"Package '{pkg.get('name')}' has no CSV resource")
    r = requests.get(csv_url, timeout=60, headers={"User-Agent": "TIVA-Academic-Research/1.0"})
    r.raise_for_status()
    path = os.path.join(RAW, "wri_esb_dashboard.csv")
    with open(path, "wb") as f:
        f.write(r.content)
    lines = r.content.decode("utf-8", errors="replace").splitlines()
    text = f"WRI dataset: {pkg.get('title')}\n" + "\n".join(lines[:400])
    rec.update(title=pkg.get("title"), published_date=pkg.get("metadata_modified"),
               text=text, final_url=csv_url, word_count=len(text.split()), status="ok",
               note=f"CKAN API package '{pkg.get('name')}', CSV {len(lines)} lines saved")
    return rec


def main():
    salvaged, still_failed = [], []
    driver = make_driver()
    tasks = [
        ("uber", lambda: salvage_uber(driver)),
        ("datagov_vahan", lambda: salvage_datagov(
            driver, "datagov_vahan_ebuses", "State/UT-wise Number of Electric Buses (VAHAN)",
            "https://www.data.gov.in/resource/stateut-wise-number-electric-buses-vahan-portal-22-11-2024",
            "Market & Fleet Reference", 9, 9, 90)),
        ("datagov_srtu", lambda: salvage_datagov(
            driver, "datagov_srtu_fleet", "State/UT wise Total Bus Fleet and Buses in Public Sector (SRTUs)",
            "https://www.data.gov.in/catalog/stateut-wise-total-bus-fleet-and-buses-public-sector-srtus",
            "Market & Fleet Reference", 7, 9, 80)),
        ("wri", salvage_wri),
    ]
    for label, fn in tasks:
        print(f"Salvaging {label} ... ", end="", flush=True)
        try:
            rec = fn()
            salvaged.append(rec)
            print(f"OK ({rec['word_count']} words)")
        except Exception as e:
            still_failed.append(dict(source=label, error=str(e)[:300]))
            print(f"FAILED: {str(e)[:150]}")
        time.sleep(2)
    driver.quit()

    if salvaged:
        with open(os.path.join(RAW, "web_scrape_results.jsonl"), "a", encoding="utf-8") as f:
            for rec in salvaged:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(RAW, "failed_sources.json"), "w", encoding="utf-8") as f:
        json.dump(still_failed, f, indent=2, ensure_ascii=False)
    print(f"\nSalvaged {len(salvaged)}, still failed {len(still_failed)}")


if __name__ == "__main__":
    main()
