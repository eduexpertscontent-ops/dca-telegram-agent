import os
import re
import json
import time
import textwrap
import logging
from datetime import datetime, timezone, timedelta

import requests
import feedparser
from bs4 import BeautifulSoup

# OpenAI SDK (pip install openai)
from openai import OpenAI

# ---------------------------
# CONFIG
# ---------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

IST = timezone(timedelta(hours=5, minutes=30))

# Required ENV VARS on Render:
# OPENAI_API_KEY
# TELEGRAM_BOT_TOKEN
# TELEGRAM_CHAT_ID  (for channel use: @DCAUPSC)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# How many polls
MCQ_COUNT_MIN = 10
MCQ_COUNT_MAX = 15

DIFFICULTY = "moderate"

# If True, also send a short message after each poll with explanation
SEND_EXPLANATION_AFTER_POLL = True

# Headlines fetch limits per source
PER_SOURCE_LIMIT = 10
MAX_TOTAL_HEADLINES = 40

# Timeout
HTTP_TIMEOUT = 20

# ---------------------------
# SOURCES
# ---------------------------

# PIB official RSS endpoints (English press releases)
PIB_RSS_URLS = [
    "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1",  # Press Releases RSS :contentReference[oaicite:3]{index=3}
]

# SEBI RSS (very useful for markets/regulatory CA)
SEBI_RSS_URLS = [
    "https://www.sebi.gov.in/sebirss.xml",  # :contentReference[oaicite:4]{index=4}
]

# RBI RSS exists but often has anti-bot; scrape press release page instead :contentReference[oaicite:5]{index=5}
RBI_PRESS_RELEASES_URL = "https://www.rbi.org.in/commonman/english/scripts/PressReleases.aspx"  # :contentReference[oaicite:6]{index=6}

# PRS billtrack (scrape)
PRS_BILLTRACK_URL = "https://prsindia.org/billtrack"  # :contentReference[oaicite:7]{index=7}

# Insights / PWOnlyIAS current affairs pages (scrape)
INSIGHTS_CA_URL = "https://www.insightsonindia.com/current-affairs-upsc/"
PWONLYIAS_CA_URL = "https://pwonlyias.com/current-affairs/"

# ---------------------------
# HELPERS
# ---------------------------

def must_have_env():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")


def http_get(url, headers=None):
    headers = headers or {}
    # Slightly realistic UA helps
    headers.setdefault("User-Agent", "Mozilla/5.0 (compatible; DCA-UPSC-Bot/1.0)")
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s


def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        k = x.get("title", "").strip().lower()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def shorten(s: str, n: int) -> str:
    s = clean_text(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------------------------
# FETCHERS
# ---------------------------

def fetch_rss(urls, source_name):
    items = []
    for u in urls:
        try:
            feed = feedparser.parse(u)
            for e in feed.entries[:PER_SOURCE_LIMIT]:
                title = clean_text(getattr(e, "title", ""))
                link = getattr(e, "link", "")
                if title:
                    items.append({
                        "source": source_name,
                        "title": title,
                        "url": link
                    })
        except Exception as ex:
            logging.warning(f"{source_name} RSS failed: {u} | {ex}")
    return items


def fetch_rbi_press_releases():
    # Scrape RBI press releases list :contentReference[oaicite:8]{index=8}
    items = []
    try:
        html = http_get(RBI_PRESS_RELEASES_URL)
        soup = BeautifulSoup(html, "html.parser")
        # RBI page has links; we pick visible headline anchors
        # Look for anchors containing "RBI" or penalty / circular etc. but keep generic
        for a in soup.select("a"):
            t = clean_text(a.get_text(" "))
            href = a.get("href") or ""
            if not t or len(t) < 12:
                continue
            # Filter for press release-like items
            if "RBI" in t or "imposes" in t.lower() or "penalty" in t.lower():
                url = href
                if url and url.startswith("/"):
                    url = "https://www.rbi.org.in" + url
                if url.startswith("http"):
                    items.append({"source": "RBI", "title": t, "url": url})
            if len(items) >= PER_SOURCE_LIMIT:
                break
    except Exception as ex:
        logging.warning(f"RBI scrape failed: {ex}")
    return items


def fetch_prs_billtrack():
    items = []
    try:
        html = http_get(PRS_BILLTRACK_URL)
        soup = BeautifulSoup(html, "html.parser")
        # Billtrack lists bill titles in headings/links
        for a in soup.select("a"):
            t = clean_text(a.get_text(" "))
            href = a.get("href") or ""
            if not t or len(t) < 10:
                continue
            # Keep only bill-like titles
            if "Bill" in t or "Amendment" in t or "Draft" in t:
                url = href
                if url and url.startswith("/"):
                    url = "https://prsindia.org" + url
                if url.startswith("http"):
                    items.append({"source": "PRS", "title": t, "url": url})
            if len(items) >= PER_SOURCE_LIMIT:
                break
    except Exception as ex:
        logging.warning(f"PRS scrape failed: {ex}")
    return items


def fetch_insights_current_affairs():
    items = []
    try:
        html = http_get(INSIGHTS_CA_URL)
        soup = BeautifulSoup(html, "html.parser")
        # Usually posts are in h2 entry-title
        for h in soup.select("h2.entry-title a, h3.entry-title a, a"):
            t = clean_text(h.get_text(" "))
            href = h.get("href") or ""
            if not t or len(t) < 12:
                continue
            # Avoid menus etc.
            if "current affairs" in t.lower() and len(t) < 25:
                continue
            if href and href.startswith("http") and "insightsonindia.com" in href:
                items.append({"source": "Insights", "title": t, "url": href})
            if len(items) >= PER_SOURCE_LIMIT:
                break
    except Exception as ex:
        logging.warning(f"Insights scrape failed: {ex}")
    return items


def fetch_pwonlyias_current_affairs():
    items = []
    try:
        html = http_get(PWONLYIAS_CA_URL)
        soup = BeautifulSoup(html, "html.parser")
        # Pick likely post links
        for a in soup.select("a"):
            t = clean_text(a.get_text(" "))
            href = a.get("href") or ""
            if not t or len(t) < 12:
                continue
            if href.startswith("http") and "pwonlyias.com" in href and "current-affairs" in href:
                # Avoid category listing duplicates
                items.append({"source": "PWOnlyIAS", "title": t, "url": href})
            if len(items) >= PER_SOURCE_LIMIT:
                break
    except Exception as ex:
        logging.warning(f"PWOnlyIAS scrape failed: {ex}")
    return items


def fetch_all_headlines():
    logging.info("Fetching today's headlines...")

    items = []
    items += fetch_rss(PIB_RSS_URLS, "PIB")
    items += fetch_rss(SEBI_RSS_URLS, "SEBI")
    items += fetch_rbi_press_releases()
    items += fetch_prs_billtrack()
    items += fetch_insights_current_affairs()
    items += fetch_pwonlyias_current_affairs()

    items = dedupe_preserve_order(items)

    # Keep cap
    items = items[:MAX_TOTAL_HEADLINES]

    if not items:
        raise RuntimeError("No headlines fetched. Check site access / URLs.")
    return items


# ---------------------------
# OPENAI: GENERATE MCQs + MAINS
# ---------------------------

def build_generation_prompt(headlines):
    # Only feed titles + sources. (Strictly from fetched items)
    lines = []
    for i, it in enumerate(headlines, 1):
        lines.append(f"{i}. [{it['source']}] {it['title']}")
    headlines_block = "\n".join(lines)

    return f"""
You are an UPSC content generator.

INPUT = today's headlines (ONLY these; do NOT add any other facts/news).
Task:
1) Create {MCQ_COUNT_MIN}-{MCQ_COUNT_MAX} UPSC Prelims MCQs from these headlines ONLY.
   - Difficulty: {DIFFICULTY}
   - Each MCQ must be answerable from the headline itself + standard static knowledge logically tied to it.
   - Do NOT invent dates, numbers, places, names that are not present in the headline.
   - Options: exactly 4 (A,B,C,D)
   - Provide correct option index (0-3) AND a short explanation (2-4 lines).

2) After MCQs, choose the MOST UPSC MAINS-relevant headline and write:
   - 1 Mains question (GS tag)
   - Model answer (Intro + Body + Conclusion), ~180-220 words, simple and exam-ready.
   - Again: do not add unverifiable specifics not in headline. Use general, standard policy framing.

Output JSON EXACTLY in this schema:

{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["...","...","...","..."],
      "correct_index": 0,
      "explanation": "..."
    }}
  ],
  "mains": {{
    "topic_headline": "...",
    "question": "...",
    "answer": "..."
  }}
}}

HEADLINES:
{headlines_block}
""".strip()


def generate_content(headlines):
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = build_generation_prompt(headlines)

    # Responses API (recommended)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.4
    )

    text = resp.output_text.strip()

    # Parse JSON safely
    try:
        data = json.loads(text)
    except Exception:
        # Try to extract JSON substring if model added stray text
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise RuntimeError("OpenAI returned non-JSON output.")
        data = json.loads(m.group(0))

    # basic validation
    mcqs = data.get("mcqs", [])
    mains = data.get("mains", {})
    if not mcqs or not mains:
        raise RuntimeError("OpenAI JSON missing mcqs/mains.")

    # Clamp MCQ count
    mcqs = mcqs[:MCQ_COUNT_MAX]
    data["mcqs"] = mcqs
    return data


# ---------------------------
# TELEGRAM
# ---------------------------

def tg_api(method, payload):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
    return r.json()


def tg_send_message(text):
    # Telegram message hard limit is 4096 chars
    for chunk in textwrap.wrap(text, width=3800, break_long_words=False, break_on_hyphens=False):
        tg_api("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True
        })


def tg_send_quiz_poll(question, options, correct_index):
    # IMPORTANT FOR CHANNELS:
    # non-anonymous polls cannot be sent to channels => must be anonymous True
    # (You hit this exact error.)
    question = shorten(question, 280)
    options = [shorten(o, 90) for o in options[:4]]

    return tg_api("sendPoll", {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": question,
        "options": options,
        "type": "quiz",
        "correct_option_id": int(correct_index),
        "is_anonymous": True,              # ✅ must be True for channel polls
        "allows_multiple_answers": False
    })


# ---------------------------
# RENDER OUTPUT
# ---------------------------

def format_header():
    today = datetime.now(IST).strftime("%d %b %Y")
    return f"📌 UPSC Daily Quiz ({today})\n✅ 10–15 MCQ Polls ({DIFFICULTY}) + 1 Mains Q&A"


def main():
    must_have_env()

    # 1) fetch headlines
    headlines = fetch_all_headlines()

    # 2) generate MCQs + mains
    data = generate_content(headlines)

    mcqs = data["mcqs"]
    mains = data["mains"]

    # 3) post header
    tg_send_message(format_header())

    # 4) send MCQ polls (with explanation after each poll)
    for idx, m in enumerate(mcqs, 1):
        q = f"Q{idx}. {m['question']}"
        opts = m["options"]
        correct = int(m["correct_index"])

        tg_send_quiz_poll(q, opts, correct)

        if SEND_EXPLANATION_AFTER_POLL:
            correct_letter = ["A", "B", "C", "D"][correct]
            exp = m.get("explanation", "").strip()
            msg = f"✅ Answer: {correct_letter}\n🧠 Explanation: {exp}"
            tg_send_message(msg)

        # small delay to avoid Telegram flood limits
        time.sleep(1.2)

    # 5) send Mains Q&A
    topic = mains.get("topic_headline", "").strip()
    mq = mains.get("question", "").strip()
    ans = mains.get("answer", "").strip()

    mains_block = (
        f"📝 MAINS (Most Relevant)\n"
        f"📍 Topic: {topic}\n\n"
        f"❓ Question:\n{mq}\n\n"
        f"✅ Model Answer:\n{ans}"
    )

    tg_send_message(mains_block)

    logging.info("Done. Posted polls + mains.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"❌ UPSC DCA Bot failed: {type(e).__name__}: {e}")
        raise
