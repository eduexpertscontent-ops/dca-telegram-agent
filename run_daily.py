# run_daily.py
# Posts EXACTLY 10 "Daily Current Affairs MCQ" Telegram QUIZ polls using ONLY today's (IST) news.
# Sources: PIB + GKToday + AffairsCloud (RSS)
# Requirements (pip): requests feedparser beautifulsoup4 python-dateutil openai

import os
import json
import time
import random
import hashlib
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple
from email.utils import parsedate_to_datetime

import requests
import feedparser
from bs4 import BeautifulSoup
from dateutil import tz

from openai import OpenAI


# =========================
# ENV / CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # e.g. "@UPPCSSUCCESS"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

IST = tz.gettz("Asia/Kolkata")
UTC = tz.gettz("UTC")

HISTORY_PATH = os.getenv("HISTORY_PATH", "history.json")

MCQS_TO_POST = 10
MAX_ARTICLES_TO_USE = 30           # today's RSS items used for context
MAX_ARTICLE_TEXT_CHARS = 3500      # cap per article text passed to LLM
REQUEST_TIMEOUT = 20
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DCA-Telegram-Agent/2.0"

# Telegram hard limits (keep safe)
TG_Q_MAX = 280
TG_OPT_MAX = 95
TG_EXPL_MAX = 180

# RSS feeds (add/remove as you want)
RSS_FEEDS = [
    "https://pib.gov.in/newsite/rssenglish.aspx",
    "https://www.gktoday.in/feed/",
    "https://affairscloud.com/feed/",
]


# =========================
# UTIL
# =========================
def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def now_ist() -> dt.datetime:
    return dt.datetime.now(tz=IST)

def today_ist_date() -> dt.date:
    return now_ist().date()

def date_key(d: Optional[dt.date] = None) -> str:
    if d is None:
        d = today_ist_date()
    return d.strftime("%Y-%m-%d")

def http_get(url: str) -> str:
    r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "header", "footer", "aside", "form", "nav"]):
        t.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text(" ", strip=True)
    return " ".join(text.split())

def parse_entry_dt_ist(entry: Any) -> Optional[dt.datetime]:
    raw = entry.get("published") or entry.get("updated")
    if not raw:
        return None
    try:
        d = parsedate_to_datetime(raw)
        if d.tzinfo is None:
            d = d.replace(tzinfo=UTC)
        return d.astimezone(IST)
    except Exception:
        return None

def safe_trim(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


# =========================
# TELEGRAM
# =========================
def tg_api(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data

def tg_send_message(text: str) -> None:
    tg_api("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": text})

def tg_send_quiz_poll(question: str, options: List[str], correct_option_id: int, explanation: str = "") -> None:
    payload: Dict[str, Any] = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": safe_trim(question, TG_Q_MAX),
        "options": [safe_trim(o, TG_OPT_MAX) for o in options],
        "type": "quiz",
        "correct_option_id": int(correct_option_id),
        "is_anonymous": True,
        "allows_multiple_answers": False,
    }
    explanation = (explanation or "").strip()
    if explanation:
        payload["explanation"] = safe_trim(explanation, TG_EXPL_MAX)
    tg_api("sendPoll", payload)


# =========================
# HISTORY (NO REPEATS)
# =========================
def load_history() -> Dict[str, Any]:
    if not os.path.exists(HISTORY_PATH):
        return {"days": {}}
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"days": {}}

def save_history(hist: Dict[str, Any]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def get_recent_hash_sets(hist: Dict[str, Any], lookback_days: int = 30) -> Tuple[set, set]:
    """Return used_question_hashes and used_fact_hashes from last N days."""
    used_q, used_f = set(), set()
    d0 = today_ist_date()
    for i in range(lookback_days):
        dk = date_key(d0 - dt.timedelta(days=i))
        day = hist.get("days", {}).get(dk)
        if not day:
            continue
        used_q.update(day.get("q_hashes", []))
        used_f.update(day.get("fact_hashes", []))
    return used_q, used_f


# =========================
# COLLECT TODAY'S NEWS ONLY
# =========================
def collect_today_rss_items() -> List[Dict[str, Any]]:
    today = today_ist_date()
    items: List[Dict[str, Any]] = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for e in feed.entries:
                pub = parse_entry_dt_ist(e)
                if not pub:
                    continue
                # STRICT: only today's IST
                if pub.date() != today:
                    continue

                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                if not title or not link:
                    continue

                items.append({
                    "title": title,
                    "link": link,
                    "published_ist": pub.isoformat(),
                    "feed": feed_url,
                })
        except Exception:
            continue

    # Dedup by link
    seen = set()
    uniq = []
    for it in items:
        if it["link"] in seen:
            continue
        seen.add(it["link"])
        uniq.append(it)

    uniq.sort(key=lambda x: x["published_ist"], reverse=True)
    return uniq[:MAX_ARTICLES_TO_USE]

def build_context(items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    contexts: List[Dict[str, str]] = []
    for it in items:
        try:
            html = http_get(it["link"])
            text = html_to_text(html)
            if not text or len(text) < 200:
                continue
            contexts.append({
                "title": it["title"],
                "url": it["link"],
                "published_ist": it["published_ist"],
                "text": text[:MAX_ARTICLE_TEXT_CHARS],
            })
        except Exception:
            continue
    return contexts


# =========================
# LLM: EXAM-STANDARD MCQs
# =========================
def llm_generate_mcqs(contexts: List[Dict[str, str]], used_q: set, used_f: set) -> List[Dict[str, Any]]:
    """
    Output format:
    [
      {
        "question": "...?",
        "options": ["...", "...", "...", "..."],
        "answer_index": 0-3,
        "explanation": "short",
        "source_url": "one of the provided urls",
        "fact_anchor": "short factual anchor phrase"
      }, ...
    ]
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    system = (
        "You are a competitive-exam MCQ setter.\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided contexts (no outside knowledge).\n"
        "2) Each question must be directly supported by exactly ONE context URL.\n"
        "3) No old news. Only today's content already provided.\n"
        "4) No vague questions. Must be exam-style, easy-to-moderate, crisp.\n"
        "5) Avoid repeating facts/questions.\n"
        "6) Options must be plausible and mutually exclusive.\n"
        "7) Keep each question under 180 characters if possible.\n"
        "Return valid JSON only."
    )

    prompt = {
        "today_ist": date_key(),
        "need_count": MCQS_TO_POST,
        "contexts": contexts,
        "avoid_question_hashes": list(used_q)[:300],
        "avoid_fact_hashes": list(used_f)[:300],
        "output_schema": {
            "question": "string",
            "options": ["string", "string", "string", "string"],
            "answer_index": "int (0-3)",
            "explanation": "string (<= 25 words)",
            "source_url": "string (must match a contexts.url)",
            "fact_anchor": "string (6-20 words, the exact key fact used)"
        }
    }

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content or "[]"
    try:
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
    except Exception:
        return []

    # Validate + dedupe
    allowed_urls = {c["url"] for c in contexts}
    out: List[Dict[str, Any]] = []

    for m in data:
        if not isinstance(m, dict):
            continue
        q = (m.get("question") or "").strip()
        opts = m.get("options")
        ans = m.get("answer_index")
        exp = (m.get("explanation") or "").strip()
        url = (m.get("source_url") or "").strip()
        anchor = (m.get("fact_anchor") or "").strip()

        if not q or not isinstance(opts, list) or len(opts) != 4:
            continue
        if not isinstance(ans, int) or ans < 0 or ans > 3:
            continue
        if url not in allowed_urls:
            continue
        if not anchor or len(anchor.split()) < 4:
            continue

        qh = sha(q.lower())
        fh = sha(anchor.lower())

        if qh in used_q or fh in used_f:
            continue

        # Telegram safety trims
        q = safe_trim(q, 220)
        opts = [safe_trim(str(o), TG_OPT_MAX) for o in opts]
        exp = safe_trim(exp, TG_EXPL_MAX)

        out.append({
            "question": q,
            "options": opts,
            "answer_index": ans,
            "explanation": exp,
            "source_url": url,
            "fact_anchor": anchor,
            "q_hash": qh,
            "fact_hash": fh,
        })
        used_q.add(qh)
        used_f.add(fh)

        if len(out) >= MCQS_TO_POST:
            break

    return out


# =========================
# MAIN PIPELINE
# =========================
def main() -> None:
    dk = date_key()
    hist = load_history()
    used_q, used_f = get_recent_hash_sets(hist, lookback_days=45)

    items = collect_today_rss_items()
    if not items:
        print(f"⚠️ No RSS items found for today (IST): {dk}. Nothing posted.")
        return

    contexts = build_context(items)
    if len(contexts) < 5:
        print(f"⚠️ Not enough readable article content today (IST): {dk}. Found {len(contexts)}. Nothing posted.")
        return

    mcqs = llm_generate_mcqs(contexts, used_q, used_f)

    # STRICT: do not post old/irrelevant filler. If <10, do not post at all.
    if len(mcqs) < MCQS_TO_POST:
        print(f"⚠️ Could not build {MCQS_TO_POST} high-quality MCQs from today's news. Got {len(mcqs)}. Nothing posted.")
        return

    # Telegram intro message (as you asked)
    tg_send_message("Daily Current Affairs MCQ")
    time.sleep(1)

    posted_q_hashes = []
    posted_fact_hashes = []
    posted_urls = []

    # Post 10 quiz polls
    for i, m in enumerate(mcqs[:MCQS_TO_POST], start=1):
        q = f"{i}. {m['question']}"
        options = m["options"]
        correct = int(m["answer_index"])

        # Randomize option order so answer isn't always A
        idxs = list(range(4))
        random.shuffle(idxs)
        new_options = [options[j] for j in idxs]
        new_correct = idxs.index(correct)

        tg_send_quiz_poll(
            question=q,
            options=new_options,
            correct_option_id=new_correct,
            explanation=m.get("explanation", "")
        )

        posted_q_hashes.append(m["q_hash"])
        posted_fact_hashes.append(m["fact_hash"])
        posted_urls.append(m["source_url"])

        time.sleep(1.2)

    # Save today's history
    hist.setdefault("days", {})
    hist["days"][dk] = {
        "posted_at_ist": now_ist().isoformat(),
        "count": MCQS_TO_POST,
        "q_hashes": posted_q_hashes,
        "fact_hashes": posted_fact_hashes,
        "urls": posted_urls,
    }
    save_history(hist)

    print(f"✅ Posted {MCQS_TO_POST} MCQs for {dk} successfully.")


if __name__ == "__main__":
    main()
