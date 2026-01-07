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

# -------------------- AI GENERATOR (EXAM LOGIC) --------------------
def generate_mcqs(news_list):
    log(f"Step 3: Generating {len(news_list)} exam-standard factual MCQs...")
    
    prompt = f"""
    You are a professional UPSC/UPPCS Paper Setter. Create 10 Factual Exam-Standard MCQs.
    
    EXAM RULES:
    1. STATIC LINKAGE: Don't just ask about the news. Ask about the Ministry, Location, Constitutional Article, or Parent Organization involved.
    2. FACTUAL DEPTH: Focus on specific percentages, years, and specific terminology (e.g., specific names of vessels or acts).
    3. NO TRIVIAL QUESTIONS: Questions must be relevant to Geography, Polity, Economy, or Art & Culture.
    
    NEWS DATA: {json.dumps(news_list)}

    OUTPUT FORMAT (JSON ONLY):
    {{
      "mcqs": [
        {{
          "question": "Example: The Indian Coast Guard, which recently added a new pollution control vessel, operates under which Union Ministry?",
          "options": ["Ministry of Home Affairs", "Ministry of Defence", "Ministry of Ports and Shipping", "Ministry of Environment"],
          "correct_index": 1,
          "source": "..."
        }}
      ]
    }}
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "You are a professional examiner. Output JSON only."},
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2 
        )
        raw_res = json.loads(resp.choices[0].message.content)
        return raw_res.get("mcqs") or list(raw_res.values())[0]
    except Exception as e:
        log(f"AI ERROR: {e}")
        return []

# -------------------- TELEGRAM POSTER (WITH SCORE POLL) --------------------
def post_to_telegram(mcqs):
    if not mcqs: return
    
    # Header
    date_str = dt.datetime.now().strftime("%d %B %Y")
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"📚 *Daily Exam-Standard MCQ: {date_str}*", "parse_mode": "Markdown"})

    # Post MCQs
    for m in mcqs:
        try:
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "question": f"{m['question'][:250]}\n\nSource: {m.get('source', 'NextIAS')}",
                "options": json.dumps(m['options']),
                "is_anonymous": True, "type": "quiz", "correct_option_id": int(m['correct_index'])
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll", data=payload)
            time.sleep(2)
        except: continue

    # Final Score Poll
    score_payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": "📊 Final Score: How many did you get correct today?",
        "options": json.dumps(["0-3 (Need more effort)", "4-6 (Decent)", "7-8 (Very Good)", "9-10 (Excellent!)"]),
        "is_anonymous": True,
        "type": "regular"
    }
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll", data=score_payload)
    log("Posted Score Poll.")

# -------------------- MAIN --------------------
def run():
    log("=== STARTING EXAM BOT ===")
    # (Scraper logic here same as before)
    # ...
    # mcqs = generate_mcqs(news_list)
    # post_to_telegram(mcqs)
    log("=== TASK COMPLETE ===")

if __name__ == "__main__":
    run()
