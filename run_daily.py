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
OPENAI_MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

# -------------------- 2. SCRAPER: DEEP EXTRACTION --------------------
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

# -------------------- 3. AI EXAM-MCQ GENERATOR --------------------
def generate_mcqs(news_list):
    if not news_list: return []
    log(f"Step 3: Generating 10 factual exam-standard MCQs...")
    
    prompt = f"""
    You are a professional UPSC/UPPCS Paper Setter. Create 10 Factual Exam-Standard MCQs in JSON format.
    
    INSTRUCTIONS:
    1. SUBJECT TAGS: Include a subject hashtag at the start: #Polity, #Economy, #Geography, #History, #Environment, #IR, or #Science.
    2. BACKWARD LINKAGE: Don't just ask about the news. Link it to the associated Ministry, Act, Constitution Article, or Location.
    3. STRUCTURE: Each object must have: "question", "options", "correct_index", "explanation", "source".
    4. EXPLANATION: Max 200 chars.
    
    NEWS DATA: {json.dumps(news_list)}
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Professional Factual Examiner. Return only a JSON object with the key 'mcqs'."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        data = json.loads(resp.choices[0].message.content)
        mcqs = data.get("mcqs") or next(iter(v for v in data.values() if isinstance(v, list)), [])
        
        # --- ROBUST KEY NORMALIZATION ---
        normalized_mcqs = []
        for m in mcqs:
            # Check for alternative key names the AI might use
            idx = m.get("correct_index")
            if idx is None:
                idx = m.get("answer_index") or m.get("correct_option") or m.get("answer")
            
            try:
                m["correct_index"] = int(idx)
                normalized_mcqs.append(m)
            except (TypeError, ValueError):
                continue # Skip if no valid index found
                
        log(f"Successfully prepared {len(normalized_mcqs)} MCQs.")
        return normalized_mcqs
    except Exception as e:
        log(f"AI Error: {e}")
        return []

# -------------------- 4. TELEGRAM POSTER --------------------
def post_to_telegram(mcqs):
    if not mcqs: return
    
    # Header: DCA - DATE
    dca_date = dt.datetime.now().strftime("%d %B %Y")
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 *DCA - {dca_date}*", "parse_mode": "Markdown"})

    # Post MCQs
    for i, m in enumerate(mcqs):
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "question": f"{m['question'][:250]}\n\nSource: {m.get('source', 'NextIAS')}",
                "options": json.dumps(m['options']),
                "is_anonymous": True,
                "type": "quiz",
                "correct_option_id": m["correct_index"],
                "explanation": m.get('explanation', 'Factual revision from today\'s news.')[:200]
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
