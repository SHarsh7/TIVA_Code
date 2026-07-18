"""
MILESTONE 2 - Reddit comment collection (requests transport).

Port of the user's reddit_old_scraper.py OldRedditScraper: identical search URL
format, selector logic (a.title / .comment / .md / .author / .score / time),
CSV schema, and politeness delays. Transport swapped from Selenium to
requests+BeautifulSoup because Reddit's network security blocks automated
Chrome (verified: block page in headless; plain requests returns 200).

Gains: post pages fetched with ?limit=500 (more comments than the browser
default). Limitation: 'load more comments' stubs need JS - deep threads
beyond the 500-comment page are not expanded (documented in report).

Usage:
  python reddit_requests_runner.py test  -> r/bangalore, 2 posts
  python reddit_requests_runner.py full  -> 4 subreddits x 20 posts
"""

import csv
import os
import random
import sys
import time
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "raw", "reddit")
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

SUBREDDIT_QUERIES = {
    "bangalore": '"electric bus" OR "school bus" OR BMTC OR Shuttl',
    "hyderabad": '"electric bus" OR "school bus" OR TSRTC',
    "delhi":     '"electric bus" OR "school bus" OR DTC',
    "mumbai":    '"electric bus" OR "school bus" OR Cityflo',
}

CSV_FIELDS = ["post_url", "post_title", "post_author", "post_score",
              "comment_author", "comment_text", "comment_score", "timestamp"]


def get(url, session, attempts=4):
    """Reddit rate-limits bursts with transient 403s - back off and retry."""
    for attempt in range(1, attempts + 1):
        r = session.get(url, headers=HEADERS, timeout=30)
        blocked = r.status_code == 403 or "blocked by network security" in r.text
        if not blocked:
            r.raise_for_status()
            return r
        if attempt < attempts:
            wait = 20 * attempt + random.uniform(0, 10)
            print(f"\n  [rate-limited, waiting {wait:.0f}s (attempt {attempt}/{attempts})] ", end="", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Reddit still blocking after {attempts} attempts: {url[:100]}")


def search_posts(session, subreddit, query, time_filter="all", max_posts=20):
    """Same URL format as OldRedditScraper.search_subreddit + get_post_links."""
    url = (f"https://old.reddit.com/r/{subreddit}/search"
           f"?q={quote_plus(query)}&restrict_sr=on&sort=relevance&t={time_filter}")
    print(f"Search URL: {url}")
    posts, seen = [], set()
    while url and len(posts) < max_posts:
        soup = BeautifulSoup(get(url, session).text, "lxml")
        anchors = soup.select("a.search-title") or soup.select("a.title")
        if not anchors:
            anchors = [a for a in soup.find_all("a", href=True) if "/comments/" in a["href"]]
        for a in anchors:
            href = a.get("href", "")
            if "/comments/" not in href:
                continue
            href = href.split("?")[0].replace("www.reddit.com", "old.reddit.com")
            if not href.startswith("http"):
                href = "https://old.reddit.com" + href
            pid = href.split("/comments/")[1].split("/")[0]
            if pid in seen:
                continue
            seen.add(pid)
            posts.append(dict(url=href, title=a.get_text(strip=True)))
            if len(posts) >= max_posts:
                break
        next_btn = soup.select_one("span.next-button a")
        url = next_btn["href"] if next_btn and len(posts) < max_posts else None
        time.sleep(random.uniform(5.0, 8.0))
    print(f"Found {len(posts)} posts")
    for i, p in enumerate(posts, 1):
        print(f"  {i}. {p['title'][:70]}")
    return posts


def extract_comments(session, post_url, post_title):
    """Same selectors as OldRedditScraper.extract_comments_from_post."""
    soup = BeautifulSoup(get(post_url + "?limit=500", session).text, "lxml")

    title_el = soup.select_one("a.title")
    post_title = title_el.get_text(strip=True) if title_el else post_title
    author_el = soup.select_one("#siteTable .tagline a.author")
    post_author = author_el.get_text(strip=True) if author_el else "N/A"
    score_el = soup.select_one("#siteTable .score.unvoted")
    post_score = score_el.get("title") or score_el.get_text(strip=True) if score_el else "N/A"

    comments = []
    for c in soup.select("div.comment"):
        md = c.select_one("div.md")
        if not md:
            continue
        text = md.get_text("\n", strip=True)
        if not text or text in ("[deleted]", "[removed]"):
            continue
        author = c.select_one("a.author")
        score = c.select_one("span.score")
        t = c.select_one("time")
        comments.append(dict(
            post_url=post_url, post_title=post_title, post_author=post_author,
            post_score=post_score,
            comment_author=author.get_text(strip=True) if author else "[deleted]",
            comment_text=text,
            comment_score=score.get_text(strip=True) if score else "N/A",
            timestamp=t.get("datetime") if t else "N/A"))
    return comments


def run(subreddits, max_posts):
    session = requests.Session()
    grand_total, summary = 0, []
    for sub in subreddits:
        query = SUBREDDIT_QUERIES[sub]
        print("\n" + "=" * 80)
        print(f"SUBREDDIT r/{sub} | query: {query} | max_posts={max_posts}")
        print("=" * 80)
        try:
            posts = search_posts(session, sub, query, "all", max_posts)
            all_comments = []
            for i, p in enumerate(posts, 1):
                print(f"[{i}/{len(posts)}] {p['title'][:60]} ... ", end="", flush=True)
                try:
                    got = extract_comments(session, p["url"], p["title"])
                    all_comments.extend(got)
                    print(f"{len(got)} comments")
                except Exception as e:
                    print(f"FAILED: {str(e)[:100]}")
                time.sleep(random.uniform(5.0, 8.0))
            out = os.path.join(OUT_DIR, f"reddit_comments_{sub}.csv")
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(all_comments)
            print(f"r/{sub}: {len(all_comments)} comments -> {out}")
            grand_total += len(all_comments)
            summary.append((sub, len(posts), len(all_comments)))
        except Exception as e:
            print(f"r/{sub} FAILED: {e}")
            summary.append((sub, 0, 0))
        time.sleep(4)

    print("\n" + "=" * 80)
    for sub, nposts, ncomments in summary:
        print(f"r/{sub}: {nposts} posts, {ncomments} comments")
    print(f"TOTAL COMMENTS: {grand_total}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        run(["bangalore"], max_posts=2)
    else:
        run(list(SUBREDDIT_QUERIES), max_posts=20)
