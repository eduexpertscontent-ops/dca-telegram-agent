import os
import re
import json
import time
import datetime as dt
from typing import List, Dict, Any, Optional

import requests
import feedparser
from bs4 import BeautifulSoup

# =========================
# CONFIG (ENV VARS)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@DCAUPSC").strip()

# MCQ settings
MCQ_MIN = int(os.getenv("MCQ_MIN", "10"))
MCQ_MAX = int(os.getenv("MCQ_MAX", "15"))
DIFFICULTY = os.getenv("DIFFICULTY", "moderate").strip().lower()

# Sources
PIB_RSS = os.getenv(
    "PIB_RSS",
    "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1"
)

INSIGHTS_URL = os.getenv(
    "INSIGHTS_URL",
    "https://www.insightsonindia.com/current-affairs-upsc/"
)

PWONLYIAS_URL = os.getenv(
    "PWONLYIAS_URL",
    "https://pwonlyias.com/current-affairs/"
)

USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))

# Telegram limits (important)
TG_POLL_Q_LIMIT = 300
TG_POLL_OPT_LIMIT = 100
TG_MSG_LIMIT = 4096


# =========================
# HELPERS
# =========================
def ist_today_date() -> dt.date:
    # Your timezone is Asia/Kolkata; in Render we just use UTC now + offset to compute "today"
    now_utc = dt.datetime.utcnow()
    now_ist = now_utc + dt.timedelta(hours=5, minutes=30)
    return now_ist.date()

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s

def truncate(s: str, n: int) -> str:
    s = clean_text(s)
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"

def safe_json_loads(txt: str) -> Any:
    # Extract first JSON object if model adds extra text
    txt = txt.strip()
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if m:
        txt = m.group(0)
    return json.loads(txt)

def http_get(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


# =========================
# FETCH HEADLINES
# =========================
def fetch_pib_rss(limit: int = 12) -> List[Dict[str, str]]:
    feed = feedparser.parse(PIB_RSS)
    items = []
    for e in feed.entries[:limit]:
        title = clean_text(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        published = clean_text(getattr(e, "published", "")) if hasattr(e, "published") else ""
        if title and link:
            items.append({"source": "PIB", "title": title, "url": link, "published": published})
    return items

def scrape_insights(limit: int = 10) -> List[Dict[str, str]]:
    html = http_get(INSIGHTS_URL)
    soup = BeautifulSoup(html, "html.parser")

    # Insights CA page typically has many links; we pick prominent post titles
    results = []
    for a in soup.select("h2 a, h3 a"):
        title = clean_text(a.get_text(" "))
        url = a.get("href", "")
        if not title or not url:
            continue
        # avoid nav links
        if "http" not in url:
            continue
        if "insightsonindia.com" not in url:
            continue
        results.append({"source": "Insights", "title": title, "url": url, "published": ""})

    # de-duplicate by title
    seen = set()
    uniq = []
    for it in results:
        key = it["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    return uniq[:limit]

def scrape_pwonlyias(limit: int = 10) -> List[Dict[str, str]]:
    html = http_get(PWONLYIAS_URL)
    soup = BeautifulSoup(html, "html.parser")

    results = []
    # PWOnlyIAS uses card-like headings; catch common patterns
    for a in soup.select("h2 a, h3 a, a"):
        title = clean_text(a.get_text(" "))
        url = a.get("href", "")
        if not title or not url:
            continue
        if "pwonlyias.com" not in url:
            continue
        # skip category/filter links
        if any(x in url for x in ["/category/", "/tag/", "#", "wp-login"]):
            continue
        # avoid super short junk
        if len(title) < 12:
            continue
        results.append({"source": "PWOnlyIAS", "title": title, "url": url, "published": ""})

    seen = set()
    uniq = []
    for it in results:
        key = it["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    return uniq[:limit]

def build_today_headlines() -> List[Dict[str, str]]:
    headlines = []
    try:
        headlines.extend(fetch_pib_rss(limit=14))
    except Exception as e:
        print(f"⚠ PIB RSS failed: {e}")

    try:
        headlines.extend(scrape_insights(limit=10))
    except Exception as e:
        print(f"⚠ Insights scrape failed: {e}")

    try:
        headlines.extend(scrape_pwonlyias(limit=10))
    except Exception as e:
        print(f"⚠ PWOnlyIAS scrape failed: {e}")

    # final dedupe by (title)
    seen = set()
    out = []
    for h in headlines:
        t = (h.get("title") or "").strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(h)

    # keep a reasonable cap
    return out[:25]


# =========================
# OPENAI (Responses API)
# =========================
def openai_responses(prompt: str, max_output_tokens: int = 2200, temperature: float = 0.4) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in environment variables.")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data = r.json()

    # Extract text from output items
    # The response format can vary; this handles common cases safely.
    if "output_text" in data and isinstance(data["output_text"], str) and data["output_text"].strip():
        return data["output_text"]

    text_parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    text_parts.append(c.get("text", ""))
    return "\n".join(text_parts).strip()


def generate_mcqs_and_mains(headlines: List[Dict[str, str]]) -> Dict[str, Any]:
    today = ist_today_date().strftime("%d %b %Y")
    headlines_block = "\n".join(
        [f"- ({h['source']}) {h['title']} | {h['url']}" for h in headlines]
    )

    prompt = f"""
You are creating UPSC current affairs questions.
Use ONLY the headlines provided below. Do NOT use outside knowledge or add facts not present/derivable from these items.
Difficulty: {DIFFICULTY}.
Return STRICT JSON only.

HEADLINES (today: {today}):
{headlines_block}

TASK:
1) Create {MCQ_MIN}-{MCQ_MAX} UPSC Prelims MCQs as TELEGRAM QUIZ polls.
   - Each MCQ must have exactly 4 options.
   - Provide correct_option_index (0-3).
   - Keep each question <= 240 characters to be safe for Telegram.
   - Each option <= 90 characters.
   - Explanation should be <= 220 characters (short).

2) Pick the SINGLE most relevant topic for UPSC Mains from these headlines and create:
   - mains_question (GS1/GS2/GS3/GS4 as appropriate)
   - mains_answer (Intro + Body + Conclusion), around 180-250 words, crisp.

JSON SCHEMA:
{{
  "mcqs":[
    {{
      "question":"...",
      "options":["A","B","C","D"],
      "correct_option_index": 0,
      "explanation":"..."
    }}
  ],
  "mains": {{
    "paper":"GS2/GS3/GS1/GS4",
    "topic":"...",
    "question":"...",
    "answer":"..."
  }}
}}
"""
    raw = openai_responses(prompt)
    return safe_json_loads(raw)


# =========================
# TELEGRAM
# =========================
def tg_api(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing in env vars.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
    return r.json()

def tg_send_message(text: str, parse_mode: Optional[str] = None) -> None:
    # Telegram max length handling
    text = text or ""
    if len(text) <= TG_MSG_LIMIT:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        tg_api("sendMessage", payload)
        return

    # split into chunks
    start = 0
    while start < len(text):
        chunk = text[start:start + TG_MSG_LIMIT]
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        tg_api("sendMessage", payload)
        start += TG_MSG_LIMIT
        time.sleep(0.3)

def tg_send_quiz_poll(question: str, options: List[str], correct_index: int, explanation: str = "") -> None:
    q = truncate(question, TG_POLL_Q_LIMIT)
    opts = [truncate(o, TG_POLL_OPT_LIMIT) for o in options][:4]

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": q,
        "options": opts,
        "type": "quiz",
        "correct_option_id": int(correct_index),
        "is_anonymous": True,  # IMPORTANT for channels
    }

    # Telegram explanation max is small; keep short
    if explanation:
        payload["explanation"] = truncate(explanation, 200)

    tg_api("sendPoll", payload)


# =========================
# MAIN
# =========================
def main():
    try:
        print("INFO | Fetching today's headlines...")
        headlines = build_today_headlines()

        if len(headlines) < 6:
            tg_send_message("⚠️ Not enough headlines fetched today. Please check source pages/RSS availability.")
            return

        # Intro post
        today = ist_today_date().strftime("%d %b %Y")
        tg_send_message(f"📌 UPSC Daily MCQ Polls ({today})\n(From PIB + Insights + PWOnlyIAS)")

        # Generate content
        bundle = generate_mcqs_and_mains(headlines)

        mcqs = bundle.get("mcqs", [])
        mains = bundle.get("mains", {})

        if not mcqs:
            tg_send_message("⚠️ Agent returned no MCQs. Please retry.")
            return

        # Send polls
        for i, q in enumerate(mcqs, start=1):
            question = f"Q{i}. {q.get('question','').strip()}"
            options = q.get("options", [])
            correct = q.get("correct_option_index", 0)
            explanation = q.get("explanation", "")
            if len(options) != 4:
                continue

            tg_send_quiz_poll(question, options, correct, explanation)
            time.sleep(0.8)

        # Send mains (one)
        if mains:
            paper = mains.get("paper", "GS")
            topic = mains.get("topic", "").strip()
            qn = mains.get("question", "").strip()
            ans = mains.get("answer", "").strip()

            msg = f"📝 UPSC MAINS ({paper})\nTopic: {topic}\n\nQ. {qn}\n\nAnswer:\n{ans}"
            tg_send_message(msg)

        print("✅ Done.")

    except Exception as e:
        print(f"❌ UPSC DCA Bot failed: {type(e).__name__}: {e}")
        # send short error to telegram (optional)
        try:
            tg_send_message(f"❌ Bot failed: {type(e).__name__}: {str(e)[:900]}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
