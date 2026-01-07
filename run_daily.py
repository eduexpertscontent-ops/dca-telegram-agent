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
        
        daily_link = None
        for a in soup.find_all('a', href=True):
            if "/ca/headlines-of-the-day/" in a['href']:
                href = a['href']
                # FIX: Smart URL joining to prevent doubling https://
                daily_link = href if href.startswith("http") else "https://www.nextias.com" + href
                break
    except Exception as e:
        log(f"Hub Error: {e}")
        return

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
                    h = link_tag['href']
                    full_url = h if h.startswith("http") else "https://www.nextias.com" + h
                    
                    log(f"   -> Scraping: {headline[:40]}...")
                    try:
                        art_r = requests.get(full_url, headers=HEADERS, timeout=10)
                        art_soup = BeautifulSoup(art_r.text, "html.parser")
                        content = art_soup.find("div", class_=["daily-ca-content", "article-details"])
                        news_items.append({
                            "headline": headline, 
                            "content": content.get_text()[:3000] if content else headline, 
                            "url": full_url
                        })
                    except:
                        continue
                if len(news_items) >= 10: break
    except Exception as e:
        log(f"Scraping Error: {e}")
        return

    # 3. Generate MCQs
    if not news_items:
        log("ERROR: No news items collected.")
        return

    log(f"Step 3: Sending {len(news_items)} news items to OpenAI...")
    try:
        prompt = f"""Generate EXACTLY 10 factual UPSC MCQs in JSON format.
        Structure: {{"mcqs": [{{"question": "...", "options": ["A","B","C","D"], "correct_index": 0, "source": "..."}}]}}
        
        NEWS DATA: {json.dumps(news_items)}"""

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a UPSC examiner. You must return valid JSON with the key 'mcqs'."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_res = json.loads(resp.choices[0].message.content)
        
        # SMART KEY FINDER: AI sometimes uses "mcqs", "questions", or no key at all
        mcqs = []
        if isinstance(raw_res, list):
            mcqs = raw_res
        elif "mcqs" in raw_res:
            mcqs = raw_res["mcqs"]
        elif "questions" in raw_res:
            mcqs = raw_res["questions"]
        else:
            # Fallback: get the first list found in the dictionary
            for val in raw_res.values():
                if isinstance(val, list):
                    mcqs = val
                    break
                    
        log(f"Generated {len(mcqs)} MCQs.")
    except Exception as e:
        log(f"AI ERROR: {e}")
        return

    # 4. Post to Telegram
    if not mcqs:
        log("No MCQs found in AI response.")
        return

    log("Step 4: Posting to Telegram...")
    for m in mcqs:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "question": f"{m['question'][:250]}\n\nSource: {m.get('source', 'NextIAS')}",
                "options": json.dumps(m['options']),
                "is_anonymous": True,
                "type": "quiz",
                "correct_option_id": int(m['correct_index'])
            }
            requests.post(url, data=payload)
            time.sleep(2)
        except Exception as e:
            log(f"Telegram Error: {e}")
            continue

    log("=== TASK COMPLETE ===")

if __name__ == "__main__":
    run_daily_mcq()
