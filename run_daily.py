"""
Daily Current Affairs MCQ -> Telegram Poll Poster (UPSC Prelims Std)
Sources: TOI (Iran advisory), Economic Times (Iran), Reuters (BD T20 WC), TOI (BD), LiveLaw (SC UAPA bail), TOI (SC bail)

What it does:
1) Fetches text from the fixed source URLs
2) Asks OpenAI to generate EXACTLY 10 UPSC-prelims style MCQs ONLY from those sources
3) Validates + shuffles options (so answers won't all be A)
4) Posts to your Telegram channel as Polls with header "Daily Current Affairs MCQ"

Env vars required:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID     (e.g. "@UPPCSSUCCESS")
- OPENAI_API_KEY
Optional:
- OPENAI_MODEL (default: "gpt-4o-mini")

Run:
python run_daily.py
"""

import os
import re
import json
import time
import random
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# ---------------- CONFIG ----------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Fixed sources (as per your instruction)
SOURCE_URLS = [
    # Iran travel advisory
    "https://timesofindia.indiatimes.com/india/exercise-due-caution-india-issues-travel-advisory-for-iran-urges-nationals-to-avoid-areas-of-protests/articleshow/126355228.cms",
    "https://m.economictimes.com/news/india/avoid-non-essential-travel-to-iran-india/articleshow/126358992.cms",
    # Bangladesh T20 World Cup security concerns
    "https://www.reuters.com/sports/cricket/bangladesh-seeking-move-t20-world-cup-matches-india-report-2026-01-04/",
    "https://timesofindia.indiatimes.com/sports/cricket/news/safety-and-security-concerns-what-bangladesh-said-on-not-touring-india-for-t20-world-cup/articleshow/126334992.cms",
    # Supreme Court bail / UAPA
    "https://www.livelaw.in/top-stories/supreme-court-denies-bail-to-umar-khalid-sharjeel-imam-grants-bail-to-5-others-in-delhi-riots-larger-conspiracy-case-516860",
    "https://timesofindia.indiatimes.com/india/suprme-court-denies-umar-khalid-sharjeel-imam-bail-allows-it-for-five-co-accused/articleshow/126363870.cms",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT = 20
MAX_ARTICLE_CHARS = 8000  # keep prompt compact
POLL_SLEEP_SECONDS = 1.2  # avoid Telegram rate issues

client = OpenAI()


# ---------------- TELEGRAM ----------------
def tg(method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def post_header():
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "Daily Current Affairs MCQ"})


def post_poll(question: str, options: list[str], correct_index: int):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": question[:300],  # Telegram limit safety
        "options": options[:10],
        "is_anonymous": True,
        "type": "quiz",
        "correct_option_id": int(correct_index),
    }
    tg("sendPoll", payload)


# ---------------- EXTRACTION ----------------
def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_article_text(url: str) -> dict:
    """
    Lightweight extractor:
    - grabs title
    - grabs meta description if present
    - grabs visible <p> text
    Note: Some sites (e.g., Reuters) may block. We'll skip if too thin.
    """
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return {"url": url, "ok": False, "error": f"fetch_failed: {e}", "title": "", "text": ""}

    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)

    meta_desc = ""
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"):
        meta_desc = m["content"].strip()

    # Paragraphs
    ps = [clean_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
    ps = [p for p in ps if len(p) > 40]  # filter very short bits
    body = " ".join(ps)

    # Combine
    text = " ".join([t for t in [title, meta_desc, body] if t])
    text = clean_text(text)

    # trim
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS]

    # Heuristic for usable extraction
    ok = len(text) >= 800
    return {"url": url, "ok": ok, "error": "" if ok else "too_thin_or_blocked", "title": title, "text": text}


def collect_sources() -> list[dict]:
    items = []
    for u in SOURCE_URLS:
        art = extract_article_text(u)
        items.append(art)
    # keep only usable
    usable = [x for x in items if x["ok"]]
    return usable


# ---------------- MCQ GENERATION ----------------
def build_prompt(articles: list[dict]) -> str:
    # pack sources with ids
    packed = []
    for i, a in enumerate(articles, start=1):
        packed.append(
            f"SOURCE {i}\nURL: {a['url']}\nCONTENT:\n{a['text']}\n"
        )
    sources_blob = "\n\n".join(packed)

    return f"""
You are an exam-focused Current Affairs MCQ generator for UPSC Prelims.

STRICT REQUIREMENTS:
- Generate EXACTLY 10 UPSC Prelims-standard MCQs.
- You MUST use ONLY the facts present in the provided sources. Do NOT use outside knowledge.
- Do NOT invent dates, names, numbers, places, or claims not explicitly present in sources.
- Each MCQ must have exactly 4 options and only 1 correct option.
- Difficulty: Easy to Moderate; factual, elimination-friendly.
- Avoid multi-statement questions.
- No explanations.

OUTPUT JSON ONLY in this schema:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_option_index": 0,
      "source_url": "one of the provided URLs"
    }}
  ]
}}

Also:
- "source_url" must be exactly one of the provided URLs.
- Ensure "correct_option_index" is 0-3.
- Options must be unique and non-empty.

SOURCES:
{sources_blob}
""".strip()


def generate_mcqs_from_sources(articles: list[dict]) -> dict:
    prompt = build_prompt(articles)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    content = resp.choices[0].message.content.strip()

    # hard JSON parse
    try:
        data = json.loads(content)
    except Exception:
        # attempt to salvage JSON if model wrapped text
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise RuntimeError(f"Model did not return JSON. Got:\n{content[:800]}")
        data = json.loads(m.group(0))

    return data


# ---------------- VALIDATION + SHUFFLE ----------------
def normalize_mcq(mcq: dict) -> dict:
    q = clean_text(mcq.get("question", ""))
    opts = mcq.get("options", [])
    idx = mcq.get("correct_option_index", None)
    src = mcq.get("source_url", "")

    if not q or not isinstance(opts, list) or len(opts) != 4 or idx not in [0, 1, 2, 3]:
        raise ValueError("Invalid MCQ format")

    opts = [clean_text(o) for o in opts]
    if any(not o for o in opts) or len(set(opts)) != 4:
        raise ValueError("Options must be 4 unique non-empty strings")

    # Shuffle options to avoid all answers at same position
    correct_text = opts[idx]
    shuffled = opts[:]
    random.shuffle(shuffled)
    new_idx = shuffled.index(correct_text)

    return {
        "question": q,
        "options": shuffled,
        "correct_option_index": new_idx,
        "source_url": src,
    }


def validate_sources_in_mcqs(mcqs: list[dict], allowed_urls: set[str]) -> list[dict]:
    out = []
    for m in mcqs:
        nm = normalize_mcq(m)
        if nm["source_url"] not in allowed_urls:
            raise ValueError(f"MCQ source_url not in allowed list: {nm['source_url']}")
        out.append(nm)
    if len(out) != 10:
        raise ValueError(f"Expected 10 MCQs, got {len(out)}")
    return out


# ---------------- MAIN ----------------
def main():
    today = dt.datetime.now().strftime("%Y-%m-%d")

    articles = collect_sources()
    if not articles:
        # fail-safe: just post header + message
        post_header()
        tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "No verified exam-relevant current affairs available from the configured sources today."})
        return

    data = generate_mcqs_from_sources(articles)
    mcqs = data.get("mcqs", [])
    allowed = set(a["url"] for a in articles)

    mcqs = validate_sources_in_mcqs(mcqs, allowed)

    # Post
    post_header()
    for m in mcqs:
        post_poll(m["question"], m["options"], m["correct_option_index"])
        time.sleep(POLL_SLEEP_SECONDS)

    # Optional: post sources as one message (comment out if you don't want in channel)
    src_lines = []
    used_urls = []
    for m in mcqs:
        used_urls.append(m["source_url"])
    # keep unique, preserve order
    seen = set()
    for u in used_urls:
        if u not in seen:
            seen.add(u)
            src_lines.append(u)

    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "Sources:\n" + "\n".join(src_lines)})

    print(f"✅ Posted 10 MCQs for {today} successfully.")


if __name__ == "__main__":
    main()
