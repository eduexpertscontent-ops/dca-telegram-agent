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
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        r = requests.get(BASE_URL, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"Error fetching BASE_URL: {e}")
        return {"headlines": [], "articles": []}
    
    headlines = []
    
    # NEW LOGIC: Look for Table rows (NextIAS 2026 style)
    # Most headlines are inside a table with columns "Headline | Source | Syllabus"
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            txt = cells[0].get_text(strip=True)
            # Filter out advertisements and headers
            if len(txt) > 20 and not any(word in txt.lower() for word in ["course", "admission", "headline", "syllabus"]):
                headlines.append(txt)

    # Backup: Look for standard list items if table extraction fails
    if not headlines:
        items = soup.find_all(['li', 'h3', 'p'])
        for item in items:
            txt = item.get_text(strip=True)
            if 30 < len(txt) < 200 and not any(word in txt.lower() for word in ["next ias", "enroll", "batch"]):
                headlines.append(txt)
    
    # Extract Article Links (filtering for current affairs)
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/ca/current-affairs/" in href or "/ca/headlines-of-the-day/" in href:
            full_url = "https://www.nextias.com" + href if href.startswith("/") else href
            links.append(full_url)
    
    # Scrape detailed text from the top 5 links
    articles = []
    for link in list(dict.fromkeys(links))[:5]:
        try:
            ar = requests.get(link, headers=headers, timeout=15)
            asoup = BeautifulSoup(ar.text, "html.parser")
            # NextIAS usually wraps content in these classes
            content = asoup.find("div", class_=["daily-ca-content", "article-details"])
            if content:
                text = " ".join([p.text for p in content.find_all(["p", "li"]) if len(p.text) > 40])
            else:
                text = " ".join([p.text for p in asoup.find_all("p") if len(p.text) > 60])
            articles.append({"url": link, "text": text[:3000]})
        except: continue

    return {"headlines": headlines, "articles": articles}

def generate_10_mcqs(data):
    # If no data, return empty
    if not data['headlines'] and not data['articles']:
        return []

    prompt = f"""
    You are a UPSC Prelims paper setter. Create EXACTLY 10 MCQs based on these news items.
    
    NEWS CONTENT:
    HEADLINES: {data['headlines'][:30]}
    DETAILED ARTICLES: {[a['text'] for a in data['articles']]}
    
    STRICT RULES:
    1. DO NOT mention Next IAS or any courses.
    2. Focus only on factual and conceptual news (IR, Economy, Science, Environment).
    3. Return ONLY a JSON object.
    
    FORMAT:
    {{
      "mcqs": [
        {{
          "question": "...",
          "options": ["...", "...", "...", "..."],
          "correct_index": 0
        }}
      ]
    }}
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "You are a helpful assistant that outputs JSON."}, 
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content).get("mcqs", [])[:10]
    except Exception as e:
        print(f"AI Generation Error: {e}")
        return []

# -------------------- MAIN --------------------
def run_daily_quiz():
    print("Scraping data...")
    raw_data = fetch_content()
    
    print(f"Found {len(raw_data['headlines'])} candidate headlines.")
    
    mcqs = generate_10_mcqs(raw_data)
    
    if not mcqs:
        print("Final Check: No MCQs generated. Verify BASE_URL content.")
        return

    date_label = dt.datetime.now().strftime("%d %B %Y")
    tg_call("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"📚 *UPSC Daily MCQ: {date_label}*", "parse_mode": "Markdown"})
    
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

if __name__ == "__main__":
    run_daily_quiz()
