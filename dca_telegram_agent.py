import os
import re
import json
import time
import html
import random
import traceback
from datetime import datetime, timezone

import requests
import feedparser
from bs4 import BeautifulSoup

# =========================
# CONFIG (ENV VARIABLES)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # e.g. "@DCAUPSC"

# Model: keep small/cheap; change if you want
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# How many items to feed agent
MAX_HEADLINES_TOTAL = int(os.getenv("MAX_HEADLINES_TOTAL", "25"))

# Output requirements
MCQ_COUNT = 10
MAINS_COUNT = 2
DIFFICULTY = "moderate"

# PIB RSS (official)
PIB_RSS_URLS = [
    "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=1",  # Press Releases (English)
]

# =========================
# TELEGRAM LIMITS (IMPORTANT)
# =========================
TG_POLL_Q_MAX = 300
TG_POLL_OPT_MAX = 100
TG_POLL_MAX_OPTIONS = 10
TG_MSG_MAX = 3900  # keep below 4096 safe

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

session = requests.Session()
session.headers.update(UA)

# =========================
# UTIL
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clip(s: str, n: int) -> str:
    s = clean_text(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"

def uniq_by_title(items):
    seen = set()
    out = []
    for it in items:
        t = clean_text(it.get("title", ""))
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def safe_get(url, timeout=20):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

# =========================
# FETCHERS
# =========================
def fetch_rss(url: str, source_name: str, limit: int = 12):
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:limit]:
        title = clean_text(getattr(e, "title", ""))
        link = clean_text(getattr(e, "link", ""))
        if title:
            out.append({"source": source_name, "title": title, "url": link})
    return out

def scrape_rbi_press_releases(limit: int = 10):
    # RBI RSS sometimes shows human-check; scraping PressReleases page is more reliable
    url = "https://www.rbi.org.in/Scripts/PressReleases.aspx"
    html_txt = safe_get(url)
    soup = BeautifulSoup(html_txt, "html.parser")

    out = []
    # Titles often appear as links in list
    for a in soup.select("a"):
        text = clean_text(a.get_text(" ", strip=True))
        href = a.get("href", "")
        if not href:
            continue
        if "Scripts/BS_PressReleaseDisplay.aspx" in href or "PressReleaseDisplay.aspx" in href:
            full = href if href.startswith("http") else "https://www.rbi.org.in/" + href.lstrip("/")
            if text and len(text) > 8:
                out.append({"source": "RBI", "title": text, "url": full})
        if len(out) >= limit:
            break
    return out

def scrape_prs_billtrack(limit: int = 10):
    url = "https://prsindia.org/billtrack"
    html_txt = safe_get(url)
    soup = BeautifulSoup(html_txt, "html.parser")

    out = []
    # Bill titles are usually in headings/links
    for a in soup.select("a"):
        text = clean_text(a.get_text(" ", strip=True))
        href = a.get("href", "")
        if not href or not text:
            continue
        # avoid navigation items; keep meaningful titles
        if len(text) < 12:
            continue
        if "/billtrack/" in href or "/billtrack" in href:
            full = href if href.startswith("http") else "https://prsindia.org" + href
            out.append({"source": "PRS", "title": text, "url": full})
        if len(out) >= limit:
            break
    return out

def scrape_insights_current_affairs(limit: int = 10):
    url = "https://www.insightsonindia.com/current-affairs-upsc/"
    html_txt = safe_get(url)
    soup = BeautifulSoup(html_txt, "html.parser")

    out = []
    # pick article links (common pattern: entry-title or h2 a)
    for a in soup.select("h2 a, h3 a, .entry-title a"):
        text = clean_text(a.get_text(" ", strip=True))
        href = a.get("href", "")
        if text and href and len(text) > 12:
            out.append({"source": "Insights", "title": text, "url": href})
        if len(out) >= limit:
            break
    return out

def scrape_pwonlyias_current_affairs(limit: int = 10):
    url = "https://pwonlyias.com/current-affairs/"
    html_txt = safe_get(url)
    soup = BeautifulSoup(html_txt, "html.parser")

    out = []
    # pick article links (common patterns)
    for a in soup.select("h2 a, h3 a, .elementor-post__title a, a"):
        text = clean_text(a.get_text(" ", strip=True))
        href = a.get("href", "")
        if not href or not text:
            continue
        # keep only internal current-affairs posts, avoid menu links
        if "pwonlyias.com" in href and "current-affairs" in href and len(text) > 12:
            out.append({"source": "PWOnlyIAS", "title": text, "url": href})
        if len(out) >= limit:
            break
    # dedupe because this site can repeat anchors
    return uniq_by_title(out)[:limit]

def fetch_all_headlines():
    items = []

    # PIB (RSS)
    for u in PIB_RSS_URLS:
        try:
            items += fetch_rss(u, "PIB", limit=12)
        except Exception:
            pass

    # RBI (scrape)
    try:
        items += scrape_rbi_press_releases(limit=10)
    except Exception:
        pass

    # PRS (scrape)
    try:
        items += scrape_prs_billtrack(limit=10)
    except Exception:
        pass

    # Insights (scrape)
    try:
        items += scrape_insights_current_affairs(limit=10)
    except Exception:
        pass

    # PWOnlyIAS (scrape)
    try:
        items += scrape_pwonlyias_current_affairs(limit=10)
    except Exception:
        pass

    items = uniq_by_title(items)
    random.shuffle(items)
    return items[:MAX_HEADLINES_TOTAL]

# =========================
# OPENAI (Responses API)
# =========================
def openai_generate_structured(headlines):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")

    today = datetime.now().strftime("%d %b %Y")
    headlines_txt = "\n".join([f"- ({h['source']}) {h['title']}" for h in headlines])

    system = (
        "You are an UPSC Current Affairs content generator.\n"
        "Rules:\n"
        "1) Use ONLY the provided headlines. Do NOT add extra facts beyond what is implied by the headline.\n"
        "2) Create output in STRICT JSON only (no markdown, no commentary).\n"
        "3) Difficulty of MCQs: moderate.\n"
        "4) MCQs must be answerable from headline understanding + common static concept, but do not invent factual details.\n"
        "5) 10 MCQs exactly. 2 mains exactly with model answers.\n"
        "6) DCA must be multiple items (at least 5) and each DCA item must have exactly 5 bullet points.\n"
        "7) Each MCQ must have exactly 4 options. Provide correct_index 0-3. Provide a short explanation.\n"
    )

    user = f"""
Date: {today}

HEADLINES (ONLY SOURCE OF TRUTH):
{headlines_txt}

Return JSON with this schema:

{{
  "dca": [
    {{
      "topic": "DCA 1 - <short topic title>",
      "bullets": ["1..","2..","3..","4..","5.."]
    }}
  ],
  "mcqs": [
    {{
      "question": "Q1 ...?",
      "options": ["A","B","C","D"],
      "correct_index": 0,
      "explanation": "1-2 lines"
    }}
  ],
  "mains": [
    {{
      "question": "GSx: ...",
      "answer": "Intro (2-3 lines)\\nBody (5-7 bullets)\\nConclusion (2 lines)"
    }}
  ]
}}
""".strip()

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.4,
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()

    # Extract text output
    text = ""
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                text += c.get("text", "")

    text = text.strip()
    # Sometimes model returns extra whitespace; force JSON parse
    try:
        return json.loads(text)
    except Exception:
        # Attempt to salvage JSON
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise RuntimeError("OpenAI did not return JSON.")
        return json.loads(m.group(0))

# =========================
# TELEGRAM API
# =========================
def tg_api(method: str, payload: dict):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=30)
    # Helpful debug:
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
    return r.json()

def tg_send_message(text: str):
    # split long messages
    text = clean_text(text)
    chunks = []
    while len(text) > TG_MSG_MAX:
        cut = text.rfind("\n", 0, TG_MSG_MAX)
        if cut < 500:
            cut = TG_MSG_MAX
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)

    for ch in chunks:
        tg_api("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": ch
        })
        time.sleep(0.7)

def tg_send_quiz_poll(question: str, options: list, correct_index: int):
    # Enforce Telegram poll constraints
    q = clip(question, TG_POLL_Q_MAX)
    opts = [clip(o, TG_POLL_OPT_MAX) for o in options][:TG_POLL_MAX_OPTIONS]
    if len(opts) < 2:
        raise RuntimeError("Poll must have at least 2 options.")
    if correct_index < 0 or correct_index >= len(opts):
        correct_index = 0

    return tg_api("sendPoll", {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": q,
        "options": opts,
        "type": "quiz",
        "correct_option_id": int(correct_index),
        "is_anonymous": False
    })

# =========================
# RENDER OUTPUT FORMAT
# =========================
def render_dca_block(dca_items, date_str):
    lines = []
    lines.append(f"🇮🇳 UPSC DCA ({date_str})")
    lines.append("")
    for i, item in enumerate(dca_items, start=1):
        topic = item.get("topic", f"DCA {i}")
        bullets = item.get("bullets", [])
        lines.append(f"{i}. {topic}")
        # Ensure exactly 5 bullets
        bullets = (bullets + [""] * 5)[:5]
        for bi, b in enumerate(bullets, start=1):
            if not b:
                b = "—"
            lines.append(f"   {bi}) {clean_text(b)}")
        lines.append("")
    return "\n".join(lines).strip()

def render_mains_block(mains_items):
    lines = []
    lines.append("📝 MAINS (2 Questions + Model Answers)")
    lines.append("")
    for i, m in enumerate(mains_items, start=1):
        q = clean_text(m.get("question", f"Q{i}"))
        ans = m.get("answer", "")
        lines.append(f"Q{i}. {q}")
        lines.append(clean_text(ans))
        lines.append("")
    return "\n".join(lines).strip()

# =========================
# MAIN
# =========================
def main():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Render env vars.")

    print("INFO | Fetching today's headlines...")
    headlines = fetch_all_headlines()

    if not headlines:
        # Do NOT crash the cronjob; just notify in Telegram
        tg_send_message("⚠️ UPSC DCA Bot: No headlines fetched today. Please check sources / site access.")
        return

    # Generate structured content
    content = openai_generate_structured(headlines)

    # Validate fields
    dca = content.get("dca", [])
    mcqs = content.get("mcqs", [])
    mains = content.get("mains", [])

    # Defensive: enforce counts
    if len(mcqs) < MCQ_COUNT:
        mcqs = (mcqs + [])[:MCQ_COUNT]
    mcqs = mcqs[:MCQ_COUNT]
    mains = mains[:MAINS_COUNT]

    date_str = datetime.now().strftime("%d %b %Y")
    dca_msg = render_dca_block(dca, date_str)
    tg_send_message(dca_msg)

    tg_send_message(f"✅ MCQ POLLS ({DIFFICULTY.title()}) — Answer in polls. Explanation after each poll.")

    # Send 10 polls
    for idx, q in enumerate(mcqs, start=1):
        question = q.get("question", f"Q{idx}?")
        options = q.get("options", ["A", "B", "C", "D"])
        correct = int(q.get("correct_index", 0))
        explanation = q.get("explanation", "").strip()

        # Make question short enough + include numbering
        question = f"Q{idx}. {clean_text(question)}"
        question = clip(question, TG_POLL_Q_MAX)

        # Fix options to 4
        if not isinstance(options, list):
            options = ["A", "B", "C", "D"]
        options = (options + [""] * 4)[:4]
        options = [o if o else "—" for o in options]

        tg_send_quiz_poll(question, options, correct)
        time.sleep(1.0)

        if explanation:
            tg_send_message(f"🧠 Explanation (Q{idx}): {clip(explanation, 700)}")
            time.sleep(0.7)

    # Send mains
    if mains:
        tg_send_message(render_mains_block(mains))
    else:
        tg_send_message("📝 MAINS: Not generated today (model output missing).")

if __name__ == "__main__":
    try:
        main()
        print("DONE")
    except Exception as e:
        # Don't hide errors; send in Telegram + logs
        err = f"❌ UPSC DCA Bot failed: {type(e).__name__}: {e}"
        print(err)
        print(traceback.format_exc())
        try:
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                tg_send_message(err)
        except Exception:
            pass
        raise
