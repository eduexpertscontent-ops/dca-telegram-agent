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
POLL_DELAY_SECONDS = 2.0 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------- CORE FUNCTIONS --------------------
def tg_call(method, payload):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=20)
    return r.json()

def fetch_content():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(BASE_URL, headers=headers, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    
    # 1. Target only the news headline items, usually inside specific article lists
    headlines = []
    # Narrowing down to common news list patterns for NextIAS
    news_items = soup.select('div.daily-ca-list-title, h3.title, li.headline-item')
    for item in news_items:
        txt = item.get_text(strip=True)
        # Filter out advertisement keywords
        if any(word in txt.lower() for word in ["course", "admission", "center", "batch", "ias center"]):
            continue
        headlines.append(txt)
    
    # 2. Extract links to news articles specifically (ignoring static/course pages)
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/ca/current-affairs/" in href or "/ca/headlines-of-the-day/" in href:
            full_url = "https://www.nextias.com" + href if href.startswith("/") else href
            links.append(full_url)
    
    # 3. Scrape only the main article body from links
    articles = []
    for link in list(dict.fromkeys(links))[:8]: # Check top 8 links
        try:
            ar = requests.get(link, headers=headers, timeout=15)
            asoup = BeautifulSoup(ar.text, "html.parser")
            # TARGETING: NextIAS usually puts article content in a 'details' or 'content' div
            content_div = asoup.find("div", class_=["daily-ca-content", "article-details", "entry-content"])
            if content_div:
                text = " ".join([p.text for p in content_div.find_all(["p", "li"]) if len(p.text) > 40])
            else:
                # Fallback but with better filtering
                text = " ".join([p.text for p in asoup.find_all("p") if len(p.text) > 60 and "admission" not in p.text.lower()])
            
            articles.append({"url": link, "text": text[:2500]})
        except: continue

    return {"headlines": headlines[:15], "articles": articles}

def generate_10_mcqs(data):
    prompt = f"""
    You are a UPSC Prelims expert. Use ONLY the News Content provided below.
    
    STRICT RULE: 
    - IGNORE all text about NEXT IAS centers, course admissions, or test series.
    - ONLY create questions from International Relations, Economy, Environment, and National News.
    - If there is not enough news content, create fewer than 10 MCQs rather than using ad content.
    
    HEADLINES: {data['headlines']}
    ARTICLE CONTENT: {[a['text'] for a in data['articles']]}
    
    Output JSON: {{"mcqs": [{{"question": "...", "options": ["A","B","C","D"], "correct_index": 0}}]}}
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "Return JSON only. Focus exclusively on actual news topics."}, 
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content).get("mcqs", [])[:10]

# -------------------- MAIN EXECUTION --------------------
def run_daily_quiz():
    print("Scraping clean news data...")
    raw_data = fetch_content()
    
    print(f"Content found. Generating MCQs for {len(raw_data['headlines'])} headlines...")
    mcqs = generate_10_mcqs(raw_data)
    
    if not mcqs:
        print("AI could not find enough clean news to generate questions.")
        return

    date_label = dt.datetime.now().strftime("%d %B %Y")
    tg_call("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"✅ *Current Affairs Quiz: {date_label}*\n(Based on News Headlines)", "parse_mode": "Markdown"})
    
    for i, m in enumerate(mcqs):
        tg_call("sendPoll", {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": m["question"][:300],
            "options": m["options"][:4],
            "is_anonymous": True,
            "type": "quiz",
            "correct_option_id": m["correct_index"]
        })
        time.sleep(POLL_DELAY_SECONDS)

    print("Successfully posted MCQs.")

if __name__ == "__main__":
    run_daily_quiz()
