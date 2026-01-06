import os
import re
import json
import time
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- CONFIG --------------------
BASE_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POLL_DELAY_SECONDS = 2.5 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------- SCRAPER LOGIC --------------------
def fetch_deep_content():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        r = requests.get(BASE_URL, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

    extracted_data = []
    # TARGETING: The 2026 Headlines Table
    rows = soup.find_all('tr')
    print(f"Scanning {len(rows)} table rows...")

    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline_text = cells[0].get_text(strip=True)
            # Find the first link in this row (usually the deep-dive link)
            link_tag = row.find('a', href=True)
            
            if link_tag and len(headline_text) > 25:
                href = link_tag['href']
                full_url = "https://www.nextias.com" + href if href.startswith("/") else href
                
                # Exclude marketing items
                if any(x in headline_text.lower() for x in ["course", "admission", "ias center", "batch"]):
                    continue
                
                extracted_data.append({"headline": headline_text, "url": full_url})

    # Deep-link extraction (Max 12 to ensure 10 quality MCQs)
    final_news = []
    for item in extracted_data[:12]:
        try:
            print(f"Deep-scraping: {item['headline'][:50]}...")
            article_res = requests.get(item['url'], headers=headers, timeout=12)
            a_soup = BeautifulSoup(article_res.text, "html.parser")
            
            # Content is usually in these specific divs
            content_div = a_soup.find("div", class_=["daily-ca-content", "article-details", "entry-content"])
            if content_div:
                body = " ".join([p.get_text() for p in content_div.find_all(["p", "li"]) if len(p.get_text()) > 40])
            else:
                body = " ".join([p.get_text() for p in a_soup.find_all("p") if len(p.get_text()) > 60])
            
            final_news.append({"headline": item['headline'], "content": body[:4500], "url": item['url']})
            time.sleep(1.2) # Avoid rate limits
        except: continue

    return final_news

def generate_exam_mcqs(news_list):
    if not news_list: return []

    prompt = f"""
    You are a Senior UPSC Examiner. Generate EXACTLY 10 High-Yield MCQs from this news:
    {json.dumps(news_list, indent=2)}
    
    STRICT EXAM RULES:
    1. FRAMING: Use professional phrasing: "Consider the following statements regarding...", "Which of the following is/are correct?", "The term 'X' recently in news refers to...".
    2. OPTIONS: Options must be logically distracting (A, B, C, D).
    3. TOPIC: One unique question per news item. Do NOT repeat topics.
    4. DATA: Include specific figures, years, or organizations mentioned in the text.
    
    Output JSON: {{"mcqs": [{{"question": "...", "options": ["A","B","C","D"], "correct_index": 0}}]}}
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "Professional UPSC Examiner. JSON output only."}, 
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content).get("mcqs", [])[:10]

# -------------------- MAIN --------------------
def run_script():
    print("Initiating deep-link extraction...")
    news_list = fetch_deep_content()
    
    if not news_list:
        print("No valid news found. Check the website manually.")
        return

    print(f"Generating 10 UPSC-standard MCQs...")
    mcqs = generate_exam_mcqs(news_list)
    
    date_str = dt.datetime.now().strftime("%d %B %Y")
    tg_call("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"🎯 *UPSC Prelims Daily Target: {date_str}*", "parse_mode": "Markdown"})
    
    for m in mcqs:
        tg_call("sendPoll", {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": m["question"][:300],
            "options": m["options"][:4],
            "is_anonymous": True,
            "type": "quiz",
            "correct_option_id": m["correct_index"]
        })
        time.sleep(POLL_DELAY_SECONDS)

def tg_call(method, payload):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    requests.post(url, json=payload, timeout=20)

if __name__ == "__main__":
    run_script()
