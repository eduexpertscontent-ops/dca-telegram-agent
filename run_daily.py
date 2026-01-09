import os
import sys
import json
import time
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- 1. CONFIGURATION --------------------
HUB_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

OPENAI_MODEL = "gpt-5.1" 
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

# -------------------- 2. SCRAPER --------------------
def fetch_factual_news():
    log("Step 1: Connecting to Hub...")
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        
        daily_link = None
        for a in soup.find_all('a', href=True):
            if "/ca/headlines-of-the-day/" in a['href']:
                href = a['href']
                daily_link = href if href.startswith("http") else "https://www.nextias.com" + href
                break
        
        if not daily_link: 
            log("No daily headlines link found.")
            return []

        log(f"Step 2: Fetching news from: {daily_link}")
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
                    
                    log(f"   -> Extracting: {headline[:40]}...")
                    try:
                        art_r = requests.get(full_url, headers=HEADERS, timeout=10)
                        art_soup = BeautifulSoup(art_r.text, "html.parser")
                        content = art_soup.find("div", class_=["daily-ca-content", "article-details"])
                        body = content.get_text() if content else art_soup.get_text()
                        
                        news_items.append({
                            "headline": headline,
                            "content": body[:3500].strip(),
                            "url": full_url
                        })
                    except: continue
                if len(news_items) >= 10: break
        return news_items
    except Exception as e:
        log(f"Scraping Error: {e}")
        return []

# -------------------- 3. GPT-5 MCQ GENERATOR --------------------
def generate_mcqs(news_list):
    if not news_list: return []
    log(f"Step 3: Generating 10 factual exam-standard MCQs using {OPENAI_MODEL}...")
    
    prompt = f"""
    You are a professional UPSC/UPPCS Paper Setter. Create 10 Factual Exam-Standard MCQs in JSON format.
    
    STRICT CONSTRAINTS (Telegram API Limits):
    1. QUESTION: Must be strictly under 250 characters.
    2. OPTIONS: Each option must be strictly under 90 characters.
    3. EXPLANATION: Max 200 chars.
    4. SUBJECT TAGS: Start question with: #Polity, #Economy, etc.
    5. STRUCTURE: Each object must have: "question", "options", "correct_index", "explanation", "source".
    
    NEWS DATA: {json.dumps(news_list)}
    """
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"},
            input=[
                {"role": "developer", "content": "Professional Factual Examiner. Return only JSON with key 'mcqs'."},
                {"role": "user", "content": prompt}
            ],
        )
        
        data = json.loads(resp.output_text)
        mcqs = data.get("mcqs") or next(iter(v for v in data.values() if isinstance(v, list)), [])
        
        normalized_mcqs = []
        for m in mcqs:
            idx = m.get("correct_index")
            if idx is None:
                idx = m.get("answer_index") or m.get("correct_option") or m.get("answer")
            
            try:
                m["correct_index"] = int(idx)
                normalized_mcqs.append(m)
            except (TypeError, ValueError):
                continue 
                
        log(f"Successfully prepared {len(normalized_mcqs)} MCQs.")
        return normalized_mcqs
    except Exception as e:
        log(f"GPT-5 API Error: {e}")
        return []

# -------------------- 4. TELEGRAM POSTER --------------------
def post_to_telegram(mcqs):
    if not mcqs: return
    
    dca_date = dt.datetime.now().strftime("%d %B %Y")
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 *DCA - {dca_date}*", "parse_mode": "Markdown"})

    for i, m in enumerate(mcqs):
        try:
            # --- FIX: CHARACTER LIMITS COMPLIANCE ---
            # Telegram question limit is 300. We truncate to 290 to be safe.
            raw_question = f"{m['question']}\n\nSource: {m.get('source', 'NextIAS')}"
            safe_question = (raw_question[:297] + '...') if len(raw_question) > 300 else raw_question

            # Telegram option limit is 100.
            safe_options = []
            for opt in m['options']:
                safe_options.append((opt[:97] + '...') if len(opt) > 100 else opt)

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "question": safe_question,
                "options": json.dumps(safe_options),
                "is_anonymous": True,
                "type": "quiz",
                "correct_option_id": m["correct_index"],
                "explanation": m.get('explanation', 'Factual revision.')[:200]
            }
            
            res = requests.post(url, data=payload)
            if res.status_code == 200:
                log(f"   -> Posted {i+1}/{len(mcqs)}...")
            else:
                log(f"   -> API Error MCQ {i+1}: {res.text}")
            
            time.sleep(2.5)
        except Exception as e:
            log(f"Post Error: {e}")

    # Final Score Poll
    score_payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": "📊 Final Score: How many did you get correct?",
        "options": json.dumps(["0-3 (Need Revision)", "4-6 (Good)", "7-8 (Very Good)", "9-10 (Excellent!)"]),
        "is_anonymous": True, "type": "regular"
    }
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll", data=score_payload)
    log("Final score poll posted.")

# -------------------- MAIN RUNNER --------------------
if __name__ == "__main__":
    log("=== SCRIPTS STARTED ===")
    news = fetch_factual_news()
    if news:
        mcqs = generate_mcqs(news)
        post_to_telegram(mcqs)
    log("=== TASK COMPLETE ===")
