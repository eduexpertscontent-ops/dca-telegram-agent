import os
import re
import json
import time
import html
import textwrap
import datetime
from typing import List, Dict, Any, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup


# =========================
# CONFIG (ENV VARS)
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()   # e.g. "@DCAUPSC" or channel id like -100xxxxxxxxxx
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

# How many MCQs
MCQ_MIN = int(os.getenv("MCQ_MIN", "10"))
MCQ_MAX = int(os.getenv("MCQ_MAX", "15"))

# Difficulty
DIFFICULTY = os.getenv("DIFFICULTY", "moderate").strip().lower()

# Timeout
HTTP_TIMEOUT = 25


# =========================
# SOURCE URLS
# =========================
# PIB RSS (working page lists multiple; we are using press releases)
PIB_RSS = [
    # PIB Press Releases (English) - Regid may vary; this generally works
    "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
]

# MEA RSS page exists, but direct feed links vary by category; you can add later if you want.
# For now we keep ministries as optional RSS list you can add.
MINISTRY_RSS = [
    # Example: SEBI RSS (often used in CA); optional
    "https://www.sebi.gov.in/sebirss.xml",
]

# RBI: direct rss page triggers captcha for some routes; we will scrape press release page instead.
RBI_PRESS_RELEASE_PAGE = "https://www.rbi.org.in/commonman/english/scripts/PressReleases.aspx"

# PRS: no dependable public RSS; we will scrape pages.
PRS_HOME = "https://prsindia.org/"

# Extra websites (scrape)
INSIGHTS_CA = "https://www.insightsonindia.com/current-affairs-upsc/"
PWONLYIAS_CA = "https://pwonlyias.com/current-affairs/"


# =========================
# TELEGRAM LIMITS (IMPORTANT)
# =========================
TG_POLL_Q_MAX = 300
TG_POLL_OPT_MAX = 100
TG_POLL_OPT_COUNT_MAX = 10  # Telegram allows 2-10 options
TG_MESSAGE_MAX = 3500       # keep safe below 4096


# =========================
# HELPERS
# =========================
def now_ist_date_str() -> str:
    # IST = UTC+5:30
    utc_now = datetime.datetime.utcnow()
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    return ist_now.strftime("%d %b %Y")


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def truncate(s: str, n: int) -> str:
    s = clean_text(s)
    if len(s) <= n:
        return s
    return s[: max(0, n - 1)].rstrip() + "…"


def safe_poll_question(q: str) -> str:
    q = clean_text(q)
    # Remove option labels if model includes them
    q = re.sub(r"^\s*Q\d+[\.\)]\s*", "", q, flags=re.IGNORECASE)
    q = truncate(q, TG_POLL_Q_MAX)
    return q


def safe_poll_options(opts: List[str]) -> List[str]:
    clean_opts = []
    for o in opts:
        o = clean_text(o)
        # remove (a)/(b) etc
        o = re.sub(r"^\s*[\(\[]?[a-dA-D][\)\]]\s*", "", o)
        o = truncate(o, TG_POLL_OPT_MAX)
        if o:
            clean_opts.append(o)
    # Telegram requires 2-10
    clean_opts = clean_opts[:TG_POLL_OPT_COUNT_MAX]
    # Ensure at least 2
    if len(clean_opts) < 2:
        clean_opts = (clean_opts + ["Option 1", "Option 2"])[:2]
    return clean_opts


def dedupe_items(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("title", "").lower().strip(), it.get("link", "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# =========================
# FETCH: RSS + SCRAPE
# =========================
def fetch_rss(url: str, max_items: int = 25) -> List[Dict[str, str]]:
    try:
        feed = feedparser.parse(url)
        out = []
        for e in feed.entries[:max_items]:
            out.append({
                "source": "RSS",
                "title": clean_text(getattr(e, "title", "")),
                "link": getattr(e, "link", ""),
                "published": clean_text(getattr(e, "published", "")),
                "summary": clean_text(getattr(e, "summary", "")),
            })
        return out
    except Exception:
        return []


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DCA-Telegram-Agent/1.0; +https://example.com/bot)"
    }
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def scrape_rbi_press_releases(max_items: int = 20) -> List[Dict[str, str]]:
    # RBI page is stable and includes list items
    try:
        html_text = fetch_html(RBI_PRESS_RELEASE_PAGE)
        soup = BeautifulSoup(html_text, "html.parser")
        # Links typically inside table/list; collect a-tags that look like press releases
        links = soup.select("a[href]")
        out = []
        for a in links:
            t = clean_text(a.get_text(" "))
            href = a.get("href", "")
            if not t or len(t) < 10:
                continue
            # Heuristic: RBI press release titles often start with "RBI" or "Reserve Bank"
            if "imposes monetary penalty" in t.lower() or t.lower().startswith("rbi "):
                full = href
                if full.startswith("/"):
                    full = "https://www.rbi.org.in" + full
                out.append({
                    "source": "RBI",
                    "title": t,
                    "link": full,
                    "published": "",
                    "summary": ""
                })
            if len(out) >= max_items:
                break
        return dedupe_items(out)
    except Exception:
        return []


def scrape_insights(max_items: int = 20) -> List[Dict[str, str]]:
    try:
        html_text = fetch_html(INSIGHTS_CA)
        soup = BeautifulSoup(html_text, "html.parser")
        out = []
        # Articles typically in h2/h3 links
        for a in soup.select("h2 a[href], h3 a[href], .entry-title a[href]"):
            title = clean_text(a.get_text(" "))
            link = a.get("href", "")
            if title and link and "current-affairs" in link:
                out.append({"source": "Insights", "title": title, "link": link, "published": "", "summary": ""})
            if len(out) >= max_items:
                break
        return dedupe_items(out)
    except Exception:
        return []


def scrape_pwonlyias(max_items: int = 20) -> List[Dict[str, str]]:
    try:
        html_text = fetch_html(PWONLYIAS_CA)
        soup = BeautifulSoup(html_text, "html.parser")
        out = []
        for a in soup.select("h2 a[href], h3 a[href], .entry-title a[href]"):
            title = clean_text(a.get_text(" "))
            link = a.get("href", "")
            if title and link and "current-affairs" in link:
                out.append({"source": "PWOnlyIAS", "title": title, "link": link, "published": "", "summary": ""})
            if len(out) >= max_items:
                break
        return dedupe_items(out)
    except Exception:
        return []


def scrape_prs(max_items: int = 25) -> List[Dict[str, str]]:
    try:
        html_text = fetch_html(PRS_HOME)
        soup = BeautifulSoup(html_text, "html.parser")
        out = []
        for a in soup.select("a[href]"):
            title = clean_text(a.get_text(" "))
            link = a.get("href", "")
            if not title or len(title) < 10:
                continue
            if link.startswith("/"):
                link = "https://prsindia.org" + link
            # Heuristic keywords
            if any(k in title.lower() for k in ["bill", "parliament", "committee", "discussion paper", "policy brief", "session"]):
                out.append({"source": "PRS", "title": title, "link": link, "published": "", "summary": ""})
            if len(out) >= max_items:
                break
        return dedupe_items(out)
    except Exception:
        return []


def fetch_all_headlines() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    # PIB RSS
    for u in PIB_RSS:
        items.extend(fetch_rss(u, max_items=30))

    # Ministry RSS (optional)
    for u in MINISTRY_RSS:
        items.extend(fetch_rss(u, max_items=20))

    # Scrapes
    items.extend(scrape_rbi_press_releases(25))
    items.extend(scrape_prs(30))
    items.extend(scrape_insights(25))
    items.extend(scrape_pwonlyias(25))

    # Clean + dedupe
    items = [it for it in items if it.get("title")]
    items = dedupe_items(items)

    # Keep top N (too many makes prompt huge)
    return items[:60]


# =========================
# OPENAI (Responses API)
# =========================
def call_openai_generate(items: List[Dict[str, str]], mcq_count: int) -> Dict[str, Any]:
    """
    Returns JSON:
    {
      "mcqs": [
        {
          "question": "...",
          "options": ["...", "...", "...", "..."],
          "correct_index": 2,
          "explanation": "..."
        }, ...
      ],
      "mains": {
        "question": "...",
        "answer": "..."
      }
    }
    """

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing in env vars")

    # Build compact “today feed”
    feed_lines = []
    for i, it in enumerate(items[:50], 1):
        src = it.get("source", "")
        title = it.get("title", "")
        link = it.get("link", "")
        feed_lines.append(f"{i}. [{src}] {title} | {link}")
    feed_text = "\n".join(feed_lines)

    system = (
        "You are an UPSC current affairs question setter. "
        "You MUST generate MCQs strictly from the provided headlines/links only. "
        "If a fact is not present or cannot be safely inferred from the headline, avoid it. "
        "Keep difficulty MODERATE. "
        "Return ONLY valid JSON."
    )

    user = f"""
TODAY'S HEADLINES (use ONLY these):
{feed_text}

TASK:
1) Create exactly {mcq_count} UPSC-style MCQs for Prelims.
   - Each MCQ must be based on ONE of the items above.
   - Options should be 4 (A-D).
   - Keep question <= 240 chars (Telegram safe).
   - Each option <= 80 chars (Telegram safe).
   - Provide correct_index 0-3.
   - Explanation: 1-2 short lines, factual, no extra theory.

2) Pick the MOST MAINS-RELEVANT topic from the same items and write:
   - 1 GS-style Mains question
   - Model answer in ~160-220 words with Intro-Body-Conclusion

OUTPUT JSON SCHEMA (STRICT):
{{
  "mcqs": [
    {{
      "question": "string",
      "options": ["string","string","string","string"],
      "correct_index": 0,
      "explanation": "string",
      "source_ref": "short reference to item number like 'Item 12'"
    }}
  ],
  "mains": {{
    "question": "string",
    "answer": "string",
    "source_ref": "Item X"
  }}
}}

Remember: return ONLY JSON, no markdown, no extra keys.
""".strip()

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()

    # Responses API returns output in a nested structure; safest is to extract the first text block.
    # Typical path: data["output"][0]["content"][0]["text"]
    text = None
    try:
        text = data["output"][0]["content"][0]["text"]
    except Exception:
        # fallback: try search for any "text" field
        text = json.dumps(data)

    try:
        obj = json.loads(text)
    except Exception:
        raise RuntimeError(f"OpenAI did not return valid JSON. Raw: {text[:500]}")

    return obj


# =========================
# TELEGRAM API
# =========================
def tg_api(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing in env vars")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID missing in env vars")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, data=payload, timeout=HTTP_TIMEOUT)
    # Print Telegram error body for debugging
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram API error {r.status_code}: {r.text}")
    return r.json()


def tg_send_message(text: str, disable_preview: bool = True) -> None:
    # Telegram message length safe split
    chunks = []
    text = clean_text(text)
    if len(text) <= TG_MESSAGE_MAX:
        chunks = [text]
    else:
        chunks = textwrap.wrap(text, width=TG_MESSAGE_MAX, break_long_words=False, replace_whitespace=False)

    for ch in chunks:
        tg_api("sendMessage", {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": ch,
            "disable_web_page_preview": "true" if disable_preview else "false"
        })
        time.sleep(0.6)


def tg_send_quiz_poll(question: str, options: List[str], correct_index: int) -> Dict[str, Any]:
    q = safe_poll_question(question)
    opts = safe_poll_options(options)

    # fix correct index if out of range after truncation
    if correct_index < 0 or correct_index >= len(opts):
        correct_index = 0

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": q,
        "options": json.dumps(opts, ensure_ascii=False),
        "type": "quiz",
        "correct_option_id": str(correct_index),
        "is_anonymous": "false",
        "allows_multiple_answers": "false",
    }
    return tg_api("sendPoll", payload)


# =========================
# MAIN FLOW
# =========================
def main():
    date_str = now_ist_date_str()

    # 1) Fetch
    items = fetch_all_headlines()
    if not items:
        tg_send_message(f"⚠️ UPSC MCQ Bot ({date_str}): No items fetched today. Please check source access/URLs.")
        return

    # 2) Decide MCQ count
    mcq_count = max(MCQ_MIN, min(MCQ_MAX, 15))

    # 3) Generate via OpenAI
    generated = call_openai_generate(items, mcq_count=mcq_count)

    mcqs = generated.get("mcqs", [])
    mains = generated.get("mains", {})

    # 4) Header message
    tg_send_message(f"📌 UPSC Daily MCQs ({date_str}) — {DIFFICULTY.title()} (Polls)\n\nAnswer in polls. Explanation after each poll.")

    # 5) Send polls + explanation
    sent = 0
    for idx, q in enumerate(mcqs, 1):
        question = q.get("question", "")
        options = q.get("options", [])
        correct_index = int(q.get("correct_index", 0))
        explanation = q.get("explanation", "").strip()

        if not question or not options:
            continue

        # Send poll
        tg_send_quiz_poll(f"Q{idx}. {question}", options, correct_index)
        sent += 1

        # Explanation message
        # Keep short and clean
        exp_msg = f"✅ Q{idx} Explanation: {truncate(explanation, 700)}"
        tg_send_message(exp_msg)

        time.sleep(1.0)  # gentle rate limit

        if sent >= mcq_count:
            break

    # 6) Mains Q + Answer
    mains_q = mains.get("question", "").strip()
    mains_ans = mains.get("answer", "").strip()

    if mains_q and mains_ans:
        mains_post = (
            f"📝 UPSC Mains (Most Relevant)\n"
            f"Q: {truncate(mains_q, 700)}\n\n"
            f"Answer:\n{truncate(mains_ans, 2500)}"
        )
        tg_send_message(mains_post, disable_preview=True)
    else:
        tg_send_message("⚠️ Mains question/answer not generated today. (Model output missing).")

    tg_send_message("✅ Done for today.")


if __name__ == "__main__":
    main()
