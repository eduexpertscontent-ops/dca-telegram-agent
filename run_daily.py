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

def log(msg):
    # This 'flush' ensures the text appears in your terminal immediately
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

# -------------------- CORE LOGIC --------------------
def run_daily_mcq():
    log("=== SCRIPTS STARTED ===")
    
    # 1. Fetch Today's Link
    log(f"Step 1: Connecting to Hub URL: {HUB_URL}...")
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        log("Connection to Hub successful.")
    except Exception as e:
        log(f"CRITICAL ERROR: Failed to reach hub: {e}")
        return

    daily_link = None
    for a in soup.find_all('a', href=True):
        if "/ca/headlines-of-the-day/" in a['href']:
            daily_link = "https://www.nextias.com" + a['href'] if a['href'].startswith("/") else a['href']
            break
    
    if not daily_link:
        log("ERROR: Could not find today's specific news link.")
        return
    
    # 2. Extract Headlines & Details
    log(f"Step 2: Fetching news from: {daily_link}")
    try:
        res = requests.get(daily_link, headers=HEADERS, timeout=15)
        dsoup = BeautifulSoup(res.text, "html.parser")
    except Exception as e:
        log(f"ERROR: Failed to load daily page: {e}")
        return

    news_items = []
    rows = dsoup.find_all('tr')
    log(f"Found {len(rows)} potential news items. Starting deep extraction...")

    for i, row in enumerate(rows):
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline = cells[0].get_text(strip=True)
            link_tag = row.find('a', href=True)
            
            if link_tag and len(headline) > 20:
                full_url = "https://www.nextias.com" + link_tag['href'] if link_tag['href'].startswith("/") else link_tag['href']
                log(f"   -> Processing ({len(news_items)+1}/10): {headline[:40]}...")
                
                try:
                    art_r = requests.get(full_url, headers=HEADERS, timeout=10)
                    art_soup = BeautifulSoup(art_r.rtext, "html.parser")
                    content = art_soup.find("div", class_=["daily-ca-content", "article-details"])
                    text = content.get_text() if content else art_soup.get_text()
                    news_items.append({"headline": headline, "content": text[:3000], "url": full_url})
                except: continue
                
                if len(news_items) >= 10: break

    # 3. Generate MCQs
    log(f"Step 3: Sending {len(news_items)} news items to OpenAI...")
    try:
        prompt = f"Generate 10 factual UPSC MCQs from this news: {json.dumps(news_items)}"
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=30 # Prevent AI from hanging forever
        )
        mcqs = json.loads(resp.choices[0].message.content).get("mcqs", [])
        log(f"Successfully generated {len(mcqs)} MCQs.")
    except Exception as e:
        log(f"AI ERROR: {e}")
        return

    # 4. Post to Telegram
    log("Step 4: Posting to Telegram...")
    for m in mcqs:
        # (Insert your Telegram sendPoll code here)
        log(f"Posted: {m['question'][:30]}...")
        time.sleep(2)

    log("=== TASK COMPLETE ===")

if __name__ == "__main__":
    run_daily_mcq()
