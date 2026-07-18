"""
Reddit Scraper using old.reddit.com
Much more reliable, no CAPTCHA, no rate limiting issues!
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import csv
import json
from datetime import datetime
import re


class OldRedditScraper:
    def __init__(self, subreddit, search_query, time_filter='year', headless=False):
        """
        Initialize scraper using old.reddit.com (no CAPTCHA!)
        
        Args:
            subreddit: Subreddit name (without r/)
            search_query: Search term
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            headless: Run in headless mode
        """
        self.subreddit = subreddit
        self.search_query = search_query
        self.time_filter = time_filter
        self.comments_data = []
        
        # Setup Chrome with better stealth
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def search_subreddit(self):
        """Search using old.reddit.com - much more reliable!"""
        print(f"🔍 Searching r/{self.subreddit} for '{self.search_query}'")
        print(f"⏰ Time filter: Past {self.time_filter}")
        
        # Use old.reddit.com - no CAPTCHA, simple HTML
        search_url = f"https://old.reddit.com/r/{self.subreddit}/search?q={self.search_query.replace(' ', '+')}&restrict_sr=on&sort=relevance&t={self.time_filter}"
        
        print(f"📍 URL: {search_url}")
        self.driver.get(search_url)
        time.sleep(3)
        
        print("✅ Search page loaded successfully!")
        
    def get_post_links(self, max_posts=10):
        """Get post links from search results"""
        print(f"\n📋 Gathering post links (max: {max_posts})...")
        
        post_links = []
        post_titles = []
        
        try:
            # Wait a bit for page to fully load
            time.sleep(2)
            
            # Old Reddit has simple class names - try multiple selectors
            post_elements = self.driver.find_elements(By.CSS_SELECTOR, "a.title")
            
            if not post_elements:
                # Try alternative selector
                print("   Trying alternative selector...")
                post_elements = self.driver.find_elements(By.XPATH, "//a[contains(@class, 'title')]")
            
            if not post_elements:
                # Try even broader search
                print("   Trying broader search...")
                post_elements = self.driver.find_elements(By.XPATH, "//div[@class='entry']//a[contains(@href, '/comments/')]")
            
            print(f"   Found {len(post_elements)} potential post elements")
            
            for elem in post_elements:
                try:
                    href = elem.get_attribute('href')
                    title = elem.text.strip()
                    
                    if href and '/comments/' in href and title:
                        # Convert to old.reddit.com URL
                        if 'old.reddit.com' not in href:
                            href = href.replace('www.reddit.com', 'old.reddit.com').replace('reddit.com', 'old.reddit.com')
                        
                        if href not in post_links:
                            post_links.append(href)
                            post_titles.append(title)
                            
                            if len(post_links) >= max_posts:
                                break
                except:
                    continue
            
            if post_links:
                print(f"\n✅ Found {len(post_links)} posts to scrape:")
                for i, (title, link) in enumerate(zip(post_titles, post_links), 1):
                    print(f"   {i}. {title[:70]}...")
            else:
                print(f"\n⚠️  No posts found!")
                print(f"   This could mean:")
                print(f"   - No posts match your search term")
                print(f"   - The subreddit name is incorrect")
                print(f"   - Try a broader search term or different time filter")
                
            return post_links
            
        except Exception as e:
            print(f"❌ Error gathering posts: {e}")
            return []
    
    def extract_comments_from_post(self, post_url):
        """Extract all comments from a post on old.reddit.com"""
        print(f"\n{'='*80}")
        print(f"📄 Processing: {post_url}")
        
        try:
            self.driver.get(post_url)
            time.sleep(2)
            
            # Get post info
            try:
                post_title = self.driver.find_element(By.CSS_SELECTOR, "a.title").text
            except:
                post_title = "N/A"
            
            try:
                post_author = self.driver.find_element(By.CSS_SELECTOR, ".tagline a.author").text
            except:
                post_author = "N/A"
            
            try:
                post_score = self.driver.find_element(By.CSS_SELECTOR, ".score.unvoted").text
            except:
                post_score = "N/A"
            
            print(f"📌 Title: {post_title[:60]}...")
            print(f"👤 Author: {post_author}")
            print(f"⬆️  Score: {post_score}")
            
            # Load all comments (click "load more" buttons if any)
            self.expand_comments()
            
            # Extract comments
            comments = []
            comment_elements = self.driver.find_elements(By.CSS_SELECTOR, ".comment")
            
            print(f"💬 Found {len(comment_elements)} comments")
            
            for idx, comment_elem in enumerate(comment_elements, 1):
                try:
                    # Get comment author
                    try:
                        author = comment_elem.find_element(By.CSS_SELECTOR, ".author").text
                    except:
                        author = "[deleted]"
                    
                    # Get comment text
                    try:
                        text_elem = comment_elem.find_element(By.CSS_SELECTOR, ".md")
                        text = text_elem.text.strip()
                    except:
                        continue
                    
                    # Get comment score
                    try:
                        score = comment_elem.find_element(By.CSS_SELECTOR, ".score").text
                    except:
                        score = "N/A"
                    
                    # Get timestamp
                    try:
                        time_elem = comment_elem.find_element(By.CSS_SELECTOR, "time")
                        timestamp = time_elem.get_attribute("datetime")
                    except:
                        timestamp = "N/A"
                    
                    if text and len(text) > 0:
                        comment_data = {
                            'post_url': post_url,
                            'post_title': post_title,
                            'post_author': post_author,
                            'post_score': post_score,
                            'comment_author': author,
                            'comment_text': text,
                            'comment_score': score,
                            'timestamp': timestamp
                        }
                        comments.append(comment_data)
                    
                    if idx % 50 == 0:
                        print(f"  ✓ Extracted {idx} comments...")
                        
                except Exception as e:
                    continue
            
            print(f"✅ Extracted {len(comments)} comments from this post")
            return comments
            
        except Exception as e:
            print(f"❌ Error processing post: {e}")
            return []
    
    def expand_comments(self):
        """Click 'load more comments' buttons"""
        try:
            # Find and click "load more comments" links
            more_links = self.driver.find_elements(By.CSS_SELECTOR, ".morecomments a")
            
            if more_links:
                print(f"  🔄 Expanding {len(more_links)} hidden comment sections...")
                for link in more_links[:10]:  # Limit to avoid too many requests
                    try:
                        self.driver.execute_script("arguments[0].click();", link)
                        time.sleep(1)
                    except:
                        continue
        except:
            pass
    
    def scrape(self, max_posts=10, output_format='csv'):
        """Main scraping method"""
        try:
            print("\n" + "="*80)
            print("🚀 OLD REDDIT SCRAPER")
            print("="*80)
            print(f"📍 Subreddit: r/{self.subreddit}")
            print(f"🔍 Query: '{self.search_query}'")
            print(f"⏰ Time: Past {self.time_filter}")
            print(f"📊 Max posts: {max_posts}")
            print("="*80)
            
            # Search
            self.search_subreddit()
            
            # Get posts
            post_links = self.get_post_links(max_posts)
            
            # If no posts found via search, try browsing subreddit directly
            if not post_links:
                print("\n" + "="*80)
                print("💡 TIP: Search found no results.")
                print(f"   Trying to browse r/{self.subreddit} directly instead...")
                print("="*80)
                
                # Browse subreddit directly
                browse_url = f"https://old.reddit.com/r/{self.subreddit}/new/"
                print(f"📍 URL: {browse_url}")
                self.driver.get(browse_url)
                time.sleep(3)
                
                post_links = self.get_post_links(max_posts)
                
                if post_links:
                    print(f"\n✅ Found posts! Will scrape all comments.")
                    print(f"   You can manually filter for '{self.search_query}' later in Excel/CSV")
            
            if not post_links:
                print("\n⚠️  Still no posts found!")
                print(f"\n📝 Suggestions:")
                print(f"   1. Check if r/{self.subreddit} exists and is spelled correctly")
                print(f"   2. Try without search - just browse the subreddit")
                print(f"   3. Try 'all time' instead of '{self.time_filter}'")
                return
            
            # Extract comments
            all_comments = []
            
            for i, post_url in enumerate(post_links, 1):
                print(f"\n[{i}/{len(post_links)}]")
                comments = self.extract_comments_from_post(post_url)
                all_comments.extend(comments)
                print(f"📈 Total comments so far: {len(all_comments)}")
                time.sleep(2)  # Be polite
            
            self.comments_data = all_comments
            
            # Save
            print("\n" + "="*80)
            if output_format in ['csv', 'both']:
                self.save_to_csv()
            if output_format in ['json', 'both']:
                self.save_to_json()
            
            print("="*80)
            print(f"✅ SCRAPING COMPLETED!")
            print(f"📊 Total posts: {len(post_links)}")
            print(f"💬 Total comments: {len(all_comments)}")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.driver.quit()
            print("🔒 Browser closed")
    
    def save_to_csv(self, filename=None):
        """Save to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_{self.subreddit}_{timestamp}.csv"
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['post_url', 'post_title', 'post_author', 'post_score',
                             'comment_author', 'comment_text', 'comment_score', 'timestamp']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.comments_data)
            
            print(f"💾 Saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
    
    def save_to_json(self, filename=None):
        """Save to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_{self.subreddit}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.comments_data, f, indent=4, ensure_ascii=False)
            
            print(f"💾 Saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("🎯 REDDIT SCRAPER - Using old.reddit.com")
    print("✅ No CAPTCHA")
    print("✅ No rate limits")
    print("✅ Simple & reliable")
    print("="*80)
    
    # Get inputs
    subreddit = input("\nEnter subreddit (without r/): ").strip()
    search_query = input("Enter search query: ").strip()
    
    # Time filter
    print("\nTime filter:")
    print("1. Past hour")
    print("2. Past 24 hours")
    print("3. Past week")
    print("4. Past month")
    print("5. Past year ⭐")
    print("6. All time")
    choice = input("Choose (1-6, default 5): ").strip() or "5"
    
    time_map = {'1': 'hour', '2': 'day', '3': 'week', '4': 'month', '5': 'year', '6': 'all'}
    time_filter = time_map.get(choice, 'year')
    
    # Max posts
    try:
        max_posts = int(input("\nMax posts to scrape (default 10): ").strip() or "10")
    except:
        max_posts = 10
    
    # Output format
    print("\nOutput format:")
    print("1. CSV")
    print("2. JSON")
    print("3. Both")
    fmt_choice = input("Choose (1-3): ").strip()
    fmt_map = {'1': 'csv', '2': 'json', '3': 'both'}
    output_format = fmt_map.get(fmt_choice, 'csv')
    
    # Headless
    headless = input("\nRun headless? (y/n): ").strip().lower() == 'y'
    
    # Run scraper
    scraper = OldRedditScraper(subreddit, search_query, time_filter, headless)
    scraper.scrape(max_posts, output_format)


if __name__ == "__main__":
    main()
