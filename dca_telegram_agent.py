import os
import re
import json
import time
import html
import traceback
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
import feedparser

# -----------------------------
# CONFIG (Set these in Render -> Environment)
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # e.g. @DCAUPSC (channel username) or numeric chat_id
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

DIFFICULTY = os.getenv("DIFFICULTY", "moderate").strip().lower()
NUM_POLLS = int(os.getenv("NUM_POLLS", "10"))
NUM_MAINS = int(os.getenv("NUM_MAINS", "2"))

# PIB official RSS (reliable)
PIB_RSS_URLS = [
    "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1",  # Press Releases RSS
]

# Non-RSS sites: scrape headlines
SCRAPE_SOURCES = {
    "RBI_PRESS_RELEASES": "https://www.rbi.org.in/commonman/english/scripts/PressReleases.aspx",
    "PRS_BILLTRACK": "https://prsindia.org/billtrack",
    "INSIGHTS_CA": "https://www.insightsonindia.com/current-affairs-upsc/",
    "PWONLYIAS_CA": "https://pwonlyias.com/current-affairs/",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"

# Telegram limits (important)
TG_POLL_Q_MAX = 290       # keep < 300 safe
TG_POLL_OPT_MAX = 90      # keep < 100 safe
TG_POLL_EXPL_MAX = 3500   # message limit is higher but keep practical

# -----------------------------
# Helpers
# -----------------------------
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def truncate(s: str, n: int) -> str:
    s = clean_text(s)
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"

def http_get(url: str, timeout=20) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

# -----------------------------
# Telegram API
# -----------------------------
def tg_api(method: str, payload: dict):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=30)
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} {r.text}")
    return r.json()

def tg_send_message(text: str, parse_mode: str = None, disable_preview=True):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": disable_preview
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return tg_api("sendMessage", payload)

def tg_send_poll(question: str, options: list, correct_index: int):
    # Telegram rules: 2-10 options, each <=100 chars, question <=300 chars
    options = [truncate(o, TG_POLL_OPT_MAX) for o in options][:10]
    question = truncate(question, TG_POLL_Q_MAX)
    if len(options) < 2:
        raise ValueError("Poll needs at least 2 options.")
    if correct_index < 0 or correct_index >= len(options):
        correct_index = 0

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": question,
        "options": options,
        "type": "quiz",
        "correct_option_id": int(correct_index),
        "is_anonymous": False,
        "allows_multiple_answers": False
    }
    return tg_api("sendPoll", payload)

# -----------------------------
# OpenAI (Responses API)
# -----------------------------
def openai_responses(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "input": prompt,
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    r.raise_for_status()
    data = r.json()

    # Extract text from response
    out_text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out_text += c.get("text", "")
    return out_text.strip()

# -----------------------------
# Fetch content
# -----------------------------
def fetch_pib_rss(max_items=12):
    items = []
    for rss_url in PIB_RSS_URLS:
        d = feedparser.parse(rss_url)
        for e in d.entries[:max_items]:
            title = clean_text(getattr(e, "title", ""))
            link = clean_text(getattr(e, "link", ""))
            summary = clean_text(getattr(e, "summary", "")) if hasattr(e, "summary") else ""
            if title:
                items.append({
                    "source": "PIB",
                    "title": title,
                    "link": link,
                    "snippet": summary[:300]
                })
    return items

def scrape_headlines(url: str, source_name: str, max_items=8):
    """
    Light scraping: extracts anchor texts that look like headlines.
    We keep it conservative to avoid junk.
    """
    html_text = http_get(url)
    soup = BeautifulSoup(html_text, "html.parser")

    candidates = []
    for a in soup.select("a"):
        txt = clean_text(a.get_text(" "))
        href = a.get("href") or ""
        if not txt or len(txt) < 35:
            continue
        # skip nav junk
        if any(x in txt.lower() for x in ["privacy", "terms", "login", "sign up", "cookie"]):
            continue
        # keep likely "news" titles
        candidates.append((txt, href))

    # de-dup
    seen = set()
    out = []
    for (t, h) in candidates:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        link = h
        if link and link.startswith("/"):
            # build absolute
            from urllib.parse import urljoin
            link = urljoin(url, link)

        out.append({
            "source": source_name,
            "title": t,
            "link": link if link else url,
            "snippet": ""
        })
        if len(out) >= max_items:
            break
    return out

def fetch_all_items():
    items = []
    # PIB RSS
    try:
        items += fetch_pib_rss()
        log(f"Fetched {len(items)} items from PIB RSS")
    except Exception as e:
        log(f"PIB RSS fetch failed: {e}")

    # Scrape other sites
    for name, url in SCRAPE_SOURCES.items():
        try:
            scraped = scrape_headlines(url, name, max_items=8)
            items += scraped
            log(f"Fetched {len(scraped)} items from {name}")
        except Exception as e:
            log(f"Scrape failed for {name}: {e}")

    # final cleanup and cap
    # remove extremely similar duplicates
    uniq = []
    seen = set()
    for it in items:
        k = (it["source"], it["title"].lower())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    return uniq[:35]

# -----------------------------
# Prompt building
# -----------------------------
def build_agent_prompt(items: list) -> str:
    today = datetime.now().strftime("%d %b %Y")
    items_txt = "\n".join(
        [f"- [{x['source']}] {x['title']} ({x['link']})" for x in items]
    )

    return f"""
You are a UPSC DCA content engine for Telegram.

DATE: {today}
DIFFICULTY: {DIFFICULTY}

INPUT: "Today's headlines" below. You must generate content STRICTLY from these items only.
If a fact is not supported by a headline, do NOT invent it.

TODAY'S HEADLINES:
{items_txt}

OUTPUT FORMAT (return valid JSON only, no markdown):
{{
  "dca": [
    {{
      "topic": "DCA 1 - <short topic>",
      "points": ["point 1", "point 2", "point 3", "point 4", "point 5"]
    }},
    {{
      "topic": "DCA 2 - <short topic>",
      "points": ["point 1", "point 2", "point 3", "point 4", "point 5"]
    }}
  ],
  "mcq_polls": [
    {{
      "question": "Q1 ...?",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "Short explanation based only on the headlines."
    }}
  ],
  "mains": [
    {{
      "question": "Mains Q1 (GS?): ...",
      "answer": "Structured answer: Intro + Body (3-5 bullets) + Conclusion"
    }},
    {{
      "question": "Mains Q2 (GS?): ...",
      "answer": "Structured answer: Intro + Body (3-5 bullets) + Conclusion"
    }}
  ]
}}

RULES:
- Create EXACTLY {NUM_POLLS} mcq_polls with 4 options each.
- Create EXACTLY {NUM_MAINS} mains questions with answers.
- DCA: Create 2 to 4 DCA topics. Each topic must have exactly 5 bullet points.
- Questions must be short enough for Telegram polls.
- Options must be short (prefer <= 8 words).
- Keep explanations crisp (2-4 lines).
"""

def parse_json_strict(s: str) -> dict:
    # try direct
    try:
        return json.loads(s)
    except Exception:
        # try extract first JSON object
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

# -----------------------------
# Rendering to Telegram
# -----------------------------
def post_dca(dca_list):
    today = datetime.now().strftime("%d %b %Y")
    header = f"🗞️ UPSC DCA ({today})"
    tg_send_message(header)

    for i, blk in enumerate(dca_list, start=1):
        topic = clean_text(blk.get("topic", f"DCA {i}"))
        points = blk.get("points", [])[:5]
        points = [clean_text(p) for p in points]
        msg = f"✅ {topic}\n" + "\n".join([f"{j+1}. {points[j]}" for j in range(min(5, len(points)))])
        tg_send_message(msg)
        time.sleep(0.8)

def post_polls(mcq_polls):
    tg_send_message(f"🟩 MCQ POLLS ({DIFFICULTY.title()}) — Answer in polls, explanation after each poll.")
    for idx, q in enumerate(mcq_polls, start=1):
        question = clean_text(q.get("question", f"Q{idx}?"))
        options = q.get("options", ["A", "B", "C", "D"])
        correct_index = int(q.get("correct_index", 0))
        expl = clean_text(q.get("explanation", ""))

        # Prefix question number (keep short)
        poll_q = f"Q{idx}. {question}"
        # Safety truncation is inside tg_send_poll()
        tg_send_poll(poll_q, options, correct_index)
        time.sleep(1.2)

        if expl:
            tg_send_message(f"📝 Explanation (Q{idx}): {truncate(expl, TG_POLL_EXPL_MAX)}")
            time.sleep(0.8)

def post_mains(mains):
    tg_send_message("🟦 MAINS (2 Questions) — with model answers")
    for i, m in enumerate(mains, start=1):
        q = clean_text(m.get("question", f"Mains Q{i}"))
        a = clean_text(m.get("answer", ""))
        msg = f"Q{i}. {q}\n\n✅ Answer:\n{a}"
        tg_send_message(msg)
        time.sleep(0.8)

# -----------------------------
# Main
# -----------------------------
def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")

    log("Fetching today's headlines...")
    items = fetch_all_items()
    if not items:
        tg_send_message("⚠️ No headlines fetched today. Check RSS URLs/site access.")
        raise RuntimeError("No headlines fetched. Check RSS URLs or site access.")

    # Build prompt and call model
    prompt = build_agent_prompt(items)
    log("Calling OpenAI...")
    raw = openai_responses(prompt)
    data = parse_json_strict(raw)

    dca = data.get("dca", [])
    polls = data.get("mcq_polls", [])
    mains = data.get("mains", [])

    # Guardrails
    if not dca:
        raise RuntimeError("Model returned empty DCA.")
    if len(polls) < NUM_POLLS:
        raise RuntimeError(f"Model returned {len(polls)} polls; expected {NUM_POLLS}.")
    if len(mains) < NUM_MAINS:
        raise RuntimeError(f"Model returned {len(mains)} mains; expected {NUM_MAINS}.")

    # Post in order
    post_dca(dca[:4])
    post_polls(polls[:NUM_POLLS])
    post_mains(mains[:NUM_MAINS])

    log("Done posting to Telegram.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("ERROR: " + str(e))
        log(traceback.format_exc())
        # try to notify on Telegram (best effort)
        try:
            tg_send_message("❌ Agent failed. Check Render logs.\n" + truncate(str(e), 1000))
        except Exception:
            pass
        raise
