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
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

# -------------------- CORE LOGIC --------------------
def run_daily_mcq():
    log("=== SCRIPTS STARTED ===")
    
    # 1. Fetch Today's Link
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        daily_link = next(( "https://www.nextias.com" + a['href'] for a in soup.find_all('a', href=True) if "/ca/headlines-of-the-day/" in a['href']), None)
    except: return

    if not daily_link:
        log("ERROR: No news link found.")
        return
    
    # 2. Extract Details
    log(f"Step 2: Fetching from: {daily_link}")
    try:
        res = requests.get(daily_link, headers=HEADERS, timeout=15)
        dsoup = BeautifulSoup(res.text, "html.parser")
        news_items = []
        for row in dsoup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                headline = cells[0].get_text(strip=True)
                link_tag = row.find('a', href=True)
                if link_tag and len(headline) > 20:
                    full_url = "https://www.nextias.com" + link_tag['href'] if link_tag['href'].startswith("/") else link_tag['href']
                    log(f"   -> Scraping: {headline[:40]}...")
                    art_r = requests.get(full_url, headers=HEADERS, timeout=10)
                    art_soup = BeautifulSoup(art_r.text, "html.parser")
                    content = art_soup.find("div", class_=["daily-ca-content", "article-details"])
                    news_items.append({"headline": headline, "content": content.get_text()[:3000] if content else headline, "url": full_url})
                if len(news_items) >= 10: break
    except Exception as e:
        log(f"Scraping Error: {e}")
        return

    # 3. Generate MCQs
    log(f"Step 3: Sending {len(news_items)} news items to OpenAI...")
    try:
        prompt = f"""Generate EXACTLY 10 factual UPSC MCQs in JSON format based on the following news. 
        Each MCQ must have a 'question', 'options' (list of 4 strings), 'correct_index' (0-3), and 'source' (the provided URL).
        
        NEWS DATA: {json.dumps(news_items)}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional examiner. You must return a JSON object with a single key named 'mcqs' containing a list of objects."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_res = json.loads(resp.choices[0].message.content)
        # Dynamic key checking to ensure we get the list
        mcqs = raw_res.get("mcqs") or raw_res.get("questions") or list(raw_res.values())[0]
        log(f"Generated {len(mcqs)} MCQs.")
    except Exception as e:
        log(f"AI ERROR: {e}")
        return

    # 4. Post to Telegram
    if mcqs:
        log("Step 4: Posting to Telegram...")
        for m in mcqs:
            try:
                payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "question": f"{m['question'][:250]}\n\nSource: {m.get('source', daily_link)}",
                    "options": json.dumps(m['options']),
                    "is_anonymous": True, "type": "quiz", "correct_option_id": m['correct_index']
                }
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll", data=payload)
                time.sleep(2)
            except: continue

    log("=== TASK COMPLETE ===")

if __name__ == "__main__":
    run_daily_mcq()
