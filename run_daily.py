import os
import re
import json
import time
import random
import hashlib
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- CONFIG --------------------
BASE_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POLL_DELAY_SECONDS = 2.0  # Increased for stability
MAX_MCQS = 10 

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
    
    # Extract bullet points from 'Headlines of the Day'
    headlines = []
    heading = soup.find(lambda t: t.name in ["h1", "h2", "h3", "h4"] and "Headlines" in t.text)
    if heading:
        for sib in heading.find_all_next():
            if sib.name in ["h1", "h2", "h3", "h4"]: break
            if sib.name == "li": headlines.append(sib.get_text(strip=True))
    
    # Extract links to detailed articles
    links = []
    for a in soup.find_all("a", href=True):
        if "/ca/current-affairs/" in a["href"]:
            full_url = "https://www.nextias.com" + a["href"] if a["href"].startswith("/") else a["href"]
            links.append(full_url)
    
    # Get text from first 5 detailed articles
    articles = []
    for link in list(dict.fromkeys(links))[:5]:
        ar = requests.get(link, headers=headers, timeout=15)
        asoup = BeautifulSoup(ar.text, "html.parser")
        text = " ".join([p.text for p in asoup.find_all(["p", "li"]) if len(p.text) > 50])
        articles.append({"url": link, "text": text[:3000]}) # Limit text per article

    return {"headlines": headlines, "articles": articles}

def generate_10_mcqs(data):
    prompt = f"""
    Create EXACTLY 10 UPSC Prelims MCQs based on these current affairs:
    HEADLINES: {data['headlines']}
    ARTICLES: {[a['text'] for a in data['articles']]}
    
    RULES:
    - 1 MCQ per news topic.
    - Focus on facts, govt schemes, and international relations.
    - JSON Output format: {{"mcqs": [{{"question": "...", "options": ["A","B","C","D"], "correct_index": 0}}]}}
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)["mcqs"][:10]

# -------------------- MAIN EXECUTION --------------------
def run_daily_quiz():
    print("Fetching data from NextIAS...")
    raw_data = fetch_content()
    
    print("Generating 10 MCQs...")
    mcqs = generate_10_mcqs(raw_data)
    
    date_label = dt.datetime.now().strftime("%d %B %Y")
    tg_call("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 *Daily MCQ Bulletin: {date_label}*", "parse_mode": "Markdown"})
    
    for i, m in enumerate(mcqs):
        print(f"Posting MCQ {i+1}...")
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": m["question"][:300],
            "options": m["options"][:4],
            "is_anonymous": True,
            "type": "quiz",
            "correct_option_id": m["correct_index"]
        }
        tg_call("sendPoll", payload)
        time.sleep(POLL_DELAY_SECONDS)
        
    tg_call("sendPoll", {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": "Rate today's quiz difficulty:",
        "options": ["Easy", "Moderate", "Hard", "Very Hard"],
        "is_anonymous": True
    })
    print("Done!")

if __name__ == "__main__":
    run_daily_quiz()
