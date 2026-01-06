import os
import sys
import json
import time
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- CONFIG --------------------
HUB_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# -------------------- LOGGING HELPER --------------------
def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush() # Forces terminal to show the text immediately

# -------------------- 1. SCRAPER --------------------
def fetch_factual_news():
    log("Connecting to NextIAS Hub...")
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log(f"CRITICAL ERROR: Could not reach Hub URL: {e}")
        return []

    # Find today's headline link
    daily_link = None
    for a in soup.find_all('a', href=True):
        if "/ca/headlines-of-the-day/" in a['href']:
            daily_link = "https://www.nextias.com" + a['href'] if a['href'].startswith("/") else a['href']
            break
    
    if not daily_link:
        log("ERROR: Today's headline link not found on hub page.")
        return []

    log(f"Found daily page: {daily_link}. Fetching table...")
    try:
        res = requests.get(daily_link, headers=HEADERS, timeout=15)
        dsoup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        log(f"ERROR: Could not fetch daily page: {e}")
        return []
    
    news_items = []
    rows = dsoup.find_all('tr')
    log(f"Found {len(rows)} rows in table. Processing...")

    for i, row in enumerate(rows):
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline = cells[0].get_text(strip=True)
            link_tag = row.find('a', href=True)
            
            if link_tag and len(headline) > 20:
                full_url = "https://www.nextias.com" + link_tag['href'] if link_tag['href'].startswith("/") else link_tag['href']
                
                log(f"  -> Extracting details for: {headline[:30]}...")
                try:
                    art_res = requests.get(full_url, headers=HEADERS, timeout=10) # 10 sec timeout per article
                    art_soup = BeautifulSoup(art_res.text, "html.parser")
                    body = " ".join([p.text for p in art_soup.find_all(['p', 'li']) if len(p.text) > 50])
                    news_items.append({"headline": headline, "content": body[:3000], "url": full_url})
                except Exception as e:
                    log(f"     Skipping article: {e}")
                    continue
                
                if len(news_items) >= 10: break
    
    return news_items

# -------------------- 2. GENERATOR & POSTER --------------------
def run():
    log("=== STARTING DAILY CA BOT ===")
    
    # Check Environment Variables
    if not TELEGRAM_BOT_TOKEN or not client.api_key:
        log("FATAL: Missing API Tokens. Check your environment variables.")
        return

    data = fetch_factual_news()
    if not data:
        log("No data extracted. Exiting.")
        return

    log(f"Data ready. Sending to OpenAI ({OPENAI_MODEL})...")
    # [Insert your generate_mcqs logic here]
    
    log("MCQs generated. Posting to Telegram...")
    # [Insert your post_to_telegram logic here]
    
    log("=== TASK COMPLETE ===")

if __name__ == "__main__":
    run()
