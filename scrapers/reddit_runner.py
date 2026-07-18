"""
MILESTONE 2 - Reddit comment collection.
Drives the user-provided OldRedditScraper (reddit_old_scraper.py) across the
four city subreddits from EV_Bus_Sources_Master_v2.xlsx.

Notes:
 - Reddit search only supports t=year / t=all; the requested 2-year window is
   applied on comment timestamps during consolidation (t=all here for recall).
 - Top 20 relevance-sorted posts per subreddit (sort=relevance is the
   scraper's built-in search URL default).
 - Headless: old.reddit.com serves static HTML, no CAPTCHA expected.

Usage:
  python reddit_runner.py test   -> 1 subreddit, 2 posts (selector validation)
  python reddit_runner.py full   -> 4 subreddits x 20 posts
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reddit_old_scraper import OldRedditScraper

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "raw", "reddit")
os.makedirs(OUT_DIR, exist_ok=True)

SUBREDDIT_QUERIES = {
    "bangalore": '"electric bus" OR "school bus" OR BMTC OR Shuttl',
    "hyderabad": '"electric bus" OR "school bus" OR TSRTC',
    "delhi":     '"electric bus" OR "school bus" OR DTC',
    "mumbai":    '"electric bus" OR "school bus" OR Cityflo',
}


def run(subreddits, max_posts):
    grand_total = 0
    summary = []
    for sub in subreddits:
        query = SUBREDDIT_QUERIES[sub]
        print("\n" + "=" * 80)
        print(f"SUBREDDIT r/{sub} | query: {query} | max_posts={max_posts}")
        print("=" * 80)
        scraper = OldRedditScraper(sub, query, time_filter="all", headless=True)
        try:
            scraper.search_subreddit()
            links = scraper.get_post_links(max_posts)
            all_comments = []
            for i, url in enumerate(links, 1):
                print(f"\n[{i}/{len(links)}]")
                all_comments.extend(scraper.extract_comments_from_post(url))
                time.sleep(2)  # politeness delay between posts
            scraper.comments_data = all_comments
            out = os.path.join(OUT_DIR, f"reddit_comments_{sub}.csv")
            scraper.save_to_csv(out)
            grand_total += len(all_comments)
            summary.append((sub, len(links), len(all_comments)))
        except Exception as e:
            print(f"r/{sub} FAILED: {e}")
            summary.append((sub, 0, 0))
        finally:
            scraper.driver.quit()
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
