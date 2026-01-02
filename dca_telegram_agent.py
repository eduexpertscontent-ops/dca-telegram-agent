#!/usr/bin/env python3
"""
dca_telegram_agent.py

What it does (daily at 7 AM via your Render Cron):
1) Fetch "today's headlines" from:
   - RSS feeds (PIB / PRS / RBI / Ministries)  [you add RSS URLs in env or list below]
   - InsightsOnIndia Current Affairs (scrape headlines)
   - PWOnlyIAS Current Affairs (scrape headlines)
2) Sends those items to OpenAI to generate:
   - Structured DCA (DCA 1..N, each with 5 bullets)
   - 10 MCQs (poll/quiz format) + explanations
   - 2 Mains questions with model answers
3) Posts to Telegram channel @DCAUPSC:
   - One DCA message (structured)
   - 10 quiz polls + explanation after each poll
   - One mains message

ENV REQUIRED:
- OPENAI_API_KEY
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID   (example: "@DCAUPSC" or numeric chat id)
Optional ENV:
- OPENAI_MODEL       (default: "gpt-4.1-mini")
- RSS_URLS           (comma-separated RSS URLs)
- MAX_ITEMS          (default: 18)  # total headlines fed to agent
- DCA_COUNT          (default: 2)   # number of DCA topics to output
- MCQ_COUNT          (default: 10)
- MAINS_COUNT        (default: 2)
- DIFFICULTY         (default: "moderate")
"""

import os
import re
import json
import time
import html
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import feedparser


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("dca-telegram-agent")


# -----------------------------
# Config
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@DCAUPSC").strip()

MAX_ITEMS = int(os.getenv("MAX_ITEMS", "18"))
DCA_COUNT = int(os.getenv("DCA_COUNT", "2"))
MCQ_COUNT = int(os.getenv("MCQ_COUNT", "10"))
MAINS_COUNT = int(os.getenv("MAINS_COUNT", "2"))
DIFFICULTY = os.getenv("DIFFICULTY", "moderate").strip().lower()

RSS_URLS_ENV = os.getenv("RSS_URLS", "").strip()


DEFAULT_RSS_URLS = [
    # Add/replace with your preferred official RSS sources.
    # PIB (some regions have RSS; you can paste exact RSS links here)
    # PRS (PRS has RSS on some pages; paste exact)
    # RBI press releases RSS if available; paste exact
    # Ministries RSS if available; paste exact
]
RSS_URLS = [u.strip() for u in RSS_URLS_ENV.split(",") if u.strip()] or DEFAULT_RSS_URLS

INSIGHTS_URL = "https://www.insightsonindia.com/current-affairs-upsc/"
PWONLYIAS_URL = "https://pwonlyias.com/current-affairs/"


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class NewsItem:
    source: str
    title: str
    url: str


def require_env() -> None:
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not TG_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TG_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def clean_text(s: str) -> str:
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def uniq_items(items: List[NewsItem], limit: int) -> List[NewsItem]:
    seen = set()
    out = []
    for it in items:
        key = (it.title.lower().strip(), it.url.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out


# -----------------------------
# Ingestion
# -----------------------------
def fetch_rss_items(rss_urls: List[str], per_feed_limit: int = 6) -> List[NewsItem]:
    items: List[NewsItem] = []
    for url in rss_urls:
        try:
            d = feedparser.parse(url)
            feed_title = clean_text(d.feed.get("title", "RSS"))
            for e in d.entries[:per_feed_limit]:
                title = clean_text(getattr(e, "title", ""))
                link = clean_text(getattr(e, "link", ""))
                if title and link:
                    items.append(NewsItem(source=feed_title, title=title, url=link))
        except Exception as ex:
            log.warning("RSS fetch failed: %s | %s", url, ex)
    return items


def scrape_insights_headlines(limit: int = 8) -> List[NewsItem]:
    """
    Pulls headline links from InsightsOnIndia current affairs page.
    """
    items: List[NewsItem] = []
    try:
        r = requests.get(INSIGHTS_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try common patterns: article titles are usually in <h2 class="entry-title"> or similar
        anchors = []
        for sel in ["h2.entry-title a", "h3.entry-title a", "article h2 a", "article h3 a"]:
            anchors = soup.select(sel)
            if anchors:
                break

        for a in anchors[:limit]:
            title = clean_text(a.get_text(" "))
            link = clean_text(a.get("href", ""))
            if title and link:
                items.append(NewsItem(source="InsightsOnIndia", title=title, url=link))

    except Exception as ex:
        log.warning("Insights scrape failed: %s", ex)

    return items


def scrape_pwonlyias_headlines(limit: int = 8) -> List[NewsItem]:
    """
    Pulls headline links from PWOnlyIAS current affairs page.
    """
    items: List[NewsItem] = []
    try:
        r = requests.get(PWONLYIAS_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try several selectors since site structure can change
        anchors = []
        for sel in [
            "h3 a", "h2 a",
            "article h2 a", "article h3 a",
            ".elementor-post__title a",
            ".post-title a",
        ]:
            anchors = soup.select(sel)
            # Keep only likely article links
            anchors = [a for a in anchors if a.get("href") and "current-affairs" in a.get("href")]
            if anchors:
                break

        for a in anchors[:limit]:
            title = clean_text(a.get_text(" "))
            link = clean_text(a.get("href", ""))
            if title and link:
                items.append(NewsItem(source="PWOnlyIAS", title=title, url=link))

    except Exception as ex:
        log.warning("PWOnlyIAS scrape failed: %s", ex)

    return items


def get_todays_headlines(max_items: int) -> List[NewsItem]:
    all_items: List[NewsItem] = []

    # RSS (official sources) – best for strict “from those items”
    if RSS_URLS:
        all_items.extend(fetch_rss_items(RSS_URLS, per_feed_limit=6))

    # Add two websites the user asked
    all_items.extend(scrape_insights_headlines(limit=8))
    all_items.extend(scrape_pwonlyias_headlines(limit=8))

    all_items = uniq_items(all_items, limit=max_items)
    return all_items


# -----------------------------
# OpenAI (Responses API via HTTP)
# -----------------------------
def openai_generate_payload(items: List[NewsItem]) -> str:
    """
    Prompt: generate strict JSON with DCA, MCQs, and mains.
    """
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. [{it.source}] {it.title} | {it.url}")

    headlines_block = "\n".join(lines)

    prompt = f"""
You are an UPSC current-affairs content generator for a Telegram channel.

IMPORTANT RULES:
- You MUST use ONLY the headlines list provided below. Do NOT invent extra news.
- Generate content in STRICT JSON only. No markdown, no extra text.
- Difficulty: {DIFFICULTY}.
- Output counts:
  - DCA topics: {DCA_COUNT}
  - MCQs: {MCQ_COUNT}
  - Mains questions: {MAINS_COUNT}

FORMAT REQUIREMENTS:
A) DCA
- Provide {DCA_COUNT} items.
- Each item: "topic" + exactly 5 bullets in "points".
- Bullets should be crisp, exam-oriented, and based ONLY on provided headlines.

B) MCQs (for Telegram Poll Quiz)
- Provide exactly {MCQ_COUNT} MCQs.
- Each MCQ:
  - "question" (single sentence if possible)
  - "options" (array of 4 short options)
  - "correct_index" (0..3)
  - "explanation" (2–4 lines, based ONLY on headlines. If a fact is uncertain, avoid that MCQ.)
- Make them moderate, mixed across Polity/Eco/IR/Env/S&T/Defence etc (as available in headlines).

C) MAINS
- Provide exactly {MAINS_COUNT} mains questions.
- Each mains item:
  - "question"
  - "answer" with structure: Intro, Body (2–4 subpoints), Conclusion (short).

STRICT JSON SCHEMA:
{{
  "date_label": "DD Mon YYYY",
  "dca": [
    {{
      "topic": "string",
      "points": ["p1","p2","p3","p4","p5"]
    }}
  ],
  "mcqs": [
    {{
      "question": "string",
      "options": ["A","B","C","D"],
      "correct_index": 0,
      "explanation": "string"
    }}
  ],
  "mains": [
    {{
      "question": "string",
      "answer": "string"
    }}
  ]
}}

HEADLINES (USE ONLY THESE):
{headlines_block}
""".strip()

    return prompt


def call_openai_responses(input_text: str) -> str:
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "input": input_text,
        # Keep it deterministic-ish
        "temperature": 0.4,
        "max_output_tokens": 2200,
    }

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    data = r.json()

    # Responses API: usually has output_text in convenience fields; otherwise parse output blocks.
    if "output_text" in data and isinstance(data["output_text"], str) and data["output_text"].strip():
        return data["output_text"].strip()

    # Fallback: collect text from output array
    chunks = []
    for out in data.get("output", []):
        for c in out.get("content", []):
            if c.get("type") in ("output_text", "text") and c.get("text"):
                chunks.append(c["text"])
    return "\n".join(chunks).strip()


def safe_json_load(s: str) -> Dict[str, Any]:
    """
    Attempts strict JSON parse; tries a small repair if model wraps with code fences.
    """
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return json.loads(s)


def generate_content(items: List[NewsItem]) -> Dict[str, Any]:
    prompt = openai_generate_payload(items)
    raw = call_openai_responses(prompt)
    try:
        obj = safe_json_load(raw)
        return obj
    except Exception as ex:
        log.error("Failed to parse JSON from OpenAI. Raw:\n%s", raw[:1500])
        raise RuntimeError(f"OpenAI returned non-JSON or invalid JSON: {ex}")


# -----------------------------
# Telegram
# -----------------------------
TG_API = "https://api.telegram.org"


def tg_post(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{TG_API}/bot{TG_BOT_TOKEN}/{method}"
    r = requests.post(url, data=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def send_message(text: str, disable_preview: bool = True) -> None:
    tg_post("sendMessage", {
        "chat_id": TG_CHAT_ID,
        "text": text,
        "disable_web_page_preview": "true" if disable_preview else "false",
    })


def send_quiz_poll(question: str, options: List[str], correct_index: int) -> int:
    """
    Returns poll message_id (for tracking if needed).
    """
    data = tg_post("sendPoll", {
        "chat_id": TG_CHAT_ID,
        "question": question,
        "options": json.dumps(options, ensure_ascii=False),
        "type": "quiz",
        "correct_option_id": str(int(correct_index)),
        "is_anonymous": "true",
        "allows_multiple_answers": "false",
    })
    return int(data["result"]["message_id"])


# -----------------------------
# Render output formatting
# -----------------------------
def format_dca_message(date_label: str, dca: List[Dict[str, Any]]) -> str:
    parts = []
    parts.append(f"📰 UPSC DCA ({date_label})\n")

    for i, item in enumerate(dca, 1):
        topic = clean_text(item.get("topic", f"DCA {i}"))
        points = item.get("points", [])[:5]
        parts.append(f"{i}. DCA {i} - {topic}")
        for j, p in enumerate(points, 1):
            parts.append(f"{j}. {clean_text(p)}")
        parts.append("")  # blank line between DCAs

    return "\n".join(parts).strip()


def format_mains_message(date_label: str, mains: List[Dict[str, Any]]) -> str:
    parts = []
    parts.append(f"📝 MAINS (UPSC) — {date_label}\n")
    for i, q in enumerate(mains, 1):
        parts.append(f"Q{i}. {clean_text(q.get('question', ''))}\n")
        parts.append(clean_text(q.get("answer", "")))
        parts.append("\n" + "—" * 18 + "\n")
    return "\n".join(parts).strip()


def validate_content(obj: Dict[str, Any]) -> None:
    if "dca" not in obj or "mcqs" not in obj or "mains" not in obj:
        raise ValueError("OpenAI JSON missing one of: dca/mcqs/mains")

    if not isinstance(obj["dca"], list) or len(obj["dca"]) < 1:
        raise ValueError("dca invalid")
    if not isinstance(obj["mcqs"], list) or len(obj["mcqs"]) != MCQ_COUNT:
        raise ValueError(f"mcqs must be exactly {MCQ_COUNT}")
    if not isinstance(obj["mains"], list) or len(obj["mains"]) != MAINS_COUNT:
        raise ValueError(f"mains must be exactly {MAINS_COUNT}")

    # Basic MCQ checks
    for idx, m in enumerate(obj["mcqs"], 1):
        opts = m.get("options", [])
        ci = m.get("correct_index", None)
        if not isinstance(opts, list) or len(opts) != 4:
            raise ValueError(f"MCQ {idx} options must be 4")
        if not isinstance(ci, int) or ci < 0 or ci > 3:
            raise ValueError(f"MCQ {idx} correct_index must be 0..3")


# -----------------------------
# Main
# -----------------------------
def main():
    require_env()

    log.info("Fetching today's headlines...")
    items = get_todays_headlines(MAX_ITEMS)
    if not items:
        raise RuntimeError("No headlines fetched. Check RSS_URLS or site access.")

    log.info("Fetched %d items.", len(items))
    for it in items[:8]:
        log.info(" - [%s] %s", it.source, it.title)

    log.info("Generating structured DCA + MCQs + Mains via OpenAI...")
    obj = generate_content(items)
    validate_content(obj)

    date_label = clean_text(obj.get("date_label", time.strftime("%d %b %Y")))
    dca = obj["dca"][:DCA_COUNT]
    mcqs = obj["mcqs"][:MCQ_COUNT]
    mains = obj["mains"][:MAINS_COUNT]

    # 1) Post structured DCA message
    dca_msg = format_dca_message(date_label, dca)
    send_message(dca_msg)
    time.sleep(1.0)

    # 2) Post MCQ polls (quiz) + explanation after each poll
    send_message(f"✅ MCQ POLLS ({DIFFICULTY.capitalize()}) — Answer in polls, explanation after each poll.")
    time.sleep(1.0)

    for i, m in enumerate(mcqs, 1):
        q = clean_text(m.get("question", f"MCQ {i}"))
        opts = [clean_text(x) for x in m.get("options", ["A", "B", "C", "D"])]
        ci = int(m.get("correct_index", 0))
        exp = clean_text(m.get("explanation", ""))

        # Telegram poll question length is limited; keep it short
        if len(q) > 290:
            q = q[:287] + "..."

        send_quiz_poll(f"Q{i}. {q}", opts, ci)
        time.sleep(1.2)

        # Explanation after poll
        answer_letter = ["A", "B", "C", "D"][ci]
        expl_msg = f"🧠 Explanation (Q{i})\n✅ Correct: {answer_letter}) {opts[ci]}\n{exp}"
        send_message(expl_msg)
        time.sleep(1.2)

    # 3) Post Mains Qs with answers
    mains_msg = format_mains_message(date_label, mains)
    send_message(mains_msg)

    log.info("Done. Posted DCA + %d polls + mains.", MCQ_COUNT)


if __name__ == "__main__":
    main()
