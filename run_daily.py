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

# -------------------- 1. SCRAPER --------------------
def fetch_factual_news():
    log("Connecting to NextIAS Hub...")
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log(f"CRITICAL ERROR: {e}")
        return []

    daily_link = None
    for a in soup.find_all('a', href=True):
        if "/ca/headlines-of-the-day/" in a['href']:
            daily_link = "https://www.nextias.com" + a['href'] if a['href'].startswith("/") else a['href']
            break
    
    if not daily_link:
        log("ERROR: Today's headline link not found.")
        return []

    log(f"Found daily page: {daily_link}. Fetching details...")
    res = requests.get(daily_link, headers=HEADERS, timeout=15)
    dsoup = BeautifulSoup(res.text, "html.parser")
    
    news_items = []
    rows = dsoup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline = cells[0].get_text(strip=True)
            link_tag = row.find('a', href=True)
            if link_tag and len(headline) > 20:
                full_url = "https://www.nextias.com" + link_tag['href'] if link_tag['href'].startswith("/") else link_tag['href']
                log(f"  -> Extracting: {headline[:40]}...")
                try:
                    art_res = requests.get(full_url, headers=HEADERS, timeout=10)
                    art_soup = BeautifulSoup(art_res.text, "html.parser")
                    content_div = art_soup.find("div", class_=["daily-ca-content", "article-details"])
                    body = content_div.get_text() if content_div else art_soup.get_text()
                    news_items.append({"headline": headline, "content": body[:3500], "url": full_url})
                except: continue
                if len(news_items) >= 10: break
    return news_items

# -------------------- 2. AI GENERATOR --------------------
def generate_mcqs(news_list):
    prompt = f"""
    You are a UPSC Exam setter. Generate EXACTLY 10 factual MCQs.
    - Focus on data, years, ministries, and facts.
    - Framing: 'Consider the following statements...', 'Which is correct?'.
    - Use exactly one MCQ per news item.
    - JSON format: {{"mcqs": [{{"question": "...", "options": ["A","B","C","D"], "correct_index": 0, "source": "..."}}]}}
    NEWS: {json.dumps(news_list)}
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    return json.loads(resp.choices[0].message.content).get("mcqs", [])

# -------------------- 3. TELEGRAM POSTER --------------------
def post_to_telegram(mcqs):
    # 1. Post Header
    date_str = dt.datetime.now().strftime("%d %B %Y")
    header_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(header_url, json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 *Daily MCQ Bulletin: {date_str}*", "parse_mode": "Markdown"})
    
    # 2. Post Quiz Polls
    for m in mcqs:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"
        full_question = f"{m['question'][:250]}\n\nSource: {m['source']}"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": full_question,
            "options": json.dumps(m['options']),
            "is_anonymous": True,
            "type": "quiz",
            "correct_option_id": m['correct_index']
        }
        requests.post(url, data=payload)
        log(f"Posted Poll: {m['question'][:30]}...")
        time.sleep(2)

    # 3. Post Score Poll (Self-Report)
    score_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"
    score_payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": "📊 Final Score: How many did you get correct?",
        "options": json.dumps(["0-3 (Need Revision)", "4-6 (Good)", "7-8 (Very Good)", "9-10 (Excellent!)"]),
        "is_anonymous": True,
        "type": "regular",
        "allows_multiple_answers": False
    }
    requests.post(score_url, data=score_payload)
    log("Posted Score Poll.")

# -------------------- RUNNER --------------------
if __name__ == "__main__":
    log("=== STARTING DAILY CA BOT ===")
    news = fetch_factual_news()
    if news:
        log("Sending to OpenAI...")
        mcqs = generate_mcqs(news)
        log(f"Generated {len(mcqs)} MCQs. Posting...")
        post_to_telegram(mcqs)
    log("=== TASK COMPLETE ===")
