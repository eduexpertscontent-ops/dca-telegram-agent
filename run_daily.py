import os
import sys
import json
import time
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- 1. CONFIGURATION --------------------
# Replace these or set them as Environment Variables
HUB_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

# -------------------- 2. SCRAPER (DEEP CRAWL) --------------------
def fetch_factual_news():
    log("Connecting to Hub...")
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # Find today's headline page link
        daily_link = None
        for a in soup.find_all('a', href=True):
            if "/ca/headlines-of-the-day/" in a['href']:
                href = a['href']
                daily_link = href if href.startswith("http") else "https://www.nextias.com" + href
                break
        
        if not daily_link:
            log("Daily link not found.")
            return []

        log(f"Fetching news table from: {daily_link}")
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
                    h_url = link_tag['href']
                    full_url = h_url if h_url.startswith("http") else "https://www.nextias.com" + h_url
                    
                    log(f"   -> Scraping article: {headline[:35]}...")
                    try:
                        art_r = requests.get(full_url, headers=HEADERS, timeout=10)
                        art_soup = BeautifulSoup(art_r.text, "html.parser")
                        content = art_soup.find("div", class_=["daily-ca-content", "article-details"])
                        body_text = content.get_text() if content else art_soup.get_text()
                        
                        news_items.append({
                            "headline": headline,
                            "content": body_text[:3000].strip(),
                            "url": full_url
                        })
                    except: continue
                if len(news_items) >= 10: break
        return news_items
    except Exception as e:
        log(f"Scraping Error: {e}")
        return []

# -------------------- 3. AI EXAM-MCQ GENERATOR --------------------
def generate_mcqs(news_list):
    if not news_list: return []
    log(f"Generating 10 factual exam-standard MCQs via {OPENAI_MODEL}...")
    
    prompt = f"""
    You are a UPSC/UPPCS Paper Setter. Generate 10 Factual, Easy-to-Moderate MCQs.
    
    EXAM STANDARDS:
    1. BACKWARD LINKAGE: Link news to Static Facts (Geography, Ministry, Law, or Constitution).
    2. FACTUAL DEPTH: Focus on percentages (e.g., 31.8%), locations (e.g., Tiruchi), and organization names.
    3. SIMPLICITY: Direct questions. Single correct answer. No complex 'Statement 1 & 2' unless necessary.
    
    NEWS DATA: {json.dumps(news_list)}
    
    OUTPUT FORMAT (Strict JSON):
    {{
      "mcqs": [
        {{
          "question": "Example: In which Indian state is the harvest festival Pongal primarily celebrated?",
          "options": ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh"],
          "correct_index": 1,
          "source": "URL"
        }}
      ]
    }}
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Professional Factual Examiner. Output valid JSON with 'mcqs' key."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        raw = json.loads(resp.choices[0].message.content)
        # Find the list of MCQs regardless of the key the AI used
        return raw.get("mcqs") or next(iter(v for v in raw.values() if isinstance(v, list)), [])
    except Exception as e:
        log(f"AI Error: {e}")
        return []

# -------------------- 4. TELEGRAM POSTER --------------------
def post_to_telegram(mcqs):
    if not mcqs:
        log("No MCQs to post.")
        return

    # 1. Header DCA - DATE
    dca_date = dt.datetime.now().strftime("%d %B %Y")
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 *DCA - {dca_date}*", "parse_mode": "Markdown"})

    # 2. Quiz Polls
    for i, m in enumerate(mcqs):
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
            log(f"Posted {i+1}/10...")
            time.sleep(2.5)
        except Exception as e:
            log(f"Post Error: {e}")

    # 3. Final Score Poll
    score_payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": "📊 Final Score: How many did you get correct?",
        "options": json.dumps(["0-3 (Need Revision)", "4-6 (Good)", "7-8 (Very Good)", "9-10 (Excellent!)"]),
        "is_anonymous": True, "type": "regular"
    }
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll", data=score_payload)
    log("Score Poll Posted.")

# -------------------- MAIN RUNNER --------------------
if __name__ == "__main__":
    log("=== SCRIPTS STARTED ===")
    news = fetch_factual_news()
    if news:
        mcqs = generate_mcqs(news)
        post_to_telegram(mcqs)
    log("=== TASK COMPLETE ===")
