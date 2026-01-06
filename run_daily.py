"""
NEXTIAS -> Headlines of the Day + Daily Current Affairs -> UPSC Prelims MCQs -> Telegram Quiz Polls
(With header: "DCA MCQ - YYYY-MM-DD" and final Score Poll)

✅ Extracts from: https://www.nextias.com/daily-current-affairs
✅ Generates EXACTLY 10 MCQs (UPSC Prelims standard) from extracted content ONLY
✅ Posts:
   1) Header message: "DCA MCQ - DATE"
   2) 10 quiz polls (MCQs)
   3) 1 score poll at the end

--------------------
REQUIREMENTS:
pip install requests beautifulsoup4 openai

ENV VARS (Render / local):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID      (e.g. "@UPPCSSUCCESS")
- OPENAI_API_KEY
Optional:
- OPENAI_MODEL          (default "gpt-4o-mini")
- MAX_MCQS              (default 10)
- POLL_DELAY_SECONDS    (default 1.2)

Run:
python run_daily.py
"""

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

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_MCQS = int(os.getenv("MAX_MCQS", "10"))
POLL_DELAY_SECONDS = float(os.getenv("POLL_DELAY_SECONDS", "1.2"))

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT = 25
MAX_ARTICLE_CHARS = 7000

HISTORY_FILE = "history_nextias.json"

client = OpenAI()


# -------------------- TELEGRAM --------------------
def tg(method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API error: {data}")
    return data


def post_header_with_date(date_str: str):
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"DCA MCQ - {date_str}"})


def post_quiz_poll(question: str, options: list[str], correct_index: int):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": question[:300],
        "options": options[:4],
        "is_anonymous": True,
        "type": "quiz",
        "correct_option_id": int(correct_index),
    }
    tg("sendPoll", payload)


def post_score_poll(total_questions: int):
    # Telegram can't auto-total across multiple quiz polls; this is self-report.
    if total_questions == 10:
        options = ["0–2", "3–5", "6–8", "9–10"]
    else:
        options = ["0–20%", "21–50%", "51–80%", "81–100%"]

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": "Score Poll: How many did you get right? (Self-check)",
        "options": options,
        "is_anonymous": True,
        "type": "regular",
        "allows_multiple_answers": False,
    }
    tg("sendPoll", payload)


# -------------------- HELPERS --------------------
def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_history() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {"used_fact_hashes": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"used_fact_hashes": []}


def save_history(hist: dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def fetch_html(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


# -------------------- NEXTIAS EXTRACTION --------------------
def get_section_items_by_heading(soup: BeautifulSoup, heading_pattern: str) -> list[str]:
    """
    Finds a heading matching heading_pattern and returns bullet items (<li>) below it.
    Stops at next heading. Works even if site layout shifts slightly.
    """
    heading_regex = re.compile(heading_pattern, re.I)
    heading = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        if heading_regex.search(tag.get_text(" ", strip=True)):
            heading = tag
            break
    if not heading:
        return []

    items = []
    for sib in heading.find_all_next():
        if sib.name in ["h1", "h2", "h3", "h4"] and sib is not heading:
            break
        if sib.name == "li":
            txt = clean_text(sib.get_text(" ", strip=True))
            if len(txt) >= 15:
                items.append(txt)

    # de-dupe preserve order
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def extract_latest_ca_links(soup: BeautifulSoup, limit: int = 10) -> list[str]:
    """
    Extracts Daily Current Affairs article links from the hub page.
    Filters for /ca/current-affairs/ URLs.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/"):
            href = "https://www.nextias.com" + href
        if "nextias.com/ca/current-affairs" in href:
            links.append(href.split("?")[0])

    seen = set()
    uniq = []
    for l in links:
        if l not in seen:
            seen.add(l)
            uniq.append(l)

    return uniq[:limit]


def extract_article_text(url: str) -> dict:
    """
    Extracts visible paragraph/list text from a NextIAS CA article.
    """
    try:
        html = fetch_html(url)
    except Exception as e:
        return {"url": url, "ok": False, "title": "", "text": "", "error": f"fetch_failed: {e}"}

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    title = ""
    if soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

    chunks = []
    for el in soup.find_all(["p", "li"]):
        txt = clean_text(el.get_text(" ", strip=True))
        if len(txt) >= 40:
            chunks.append(txt)

    text = clean_text(" ".join(chunks))
    if len(text) > MAX_ARTICLE_CHARS:
        text = text[:MAX_ARTICLE_CHARS]

    ok = len(text) >= 700
    return {"url": url, "ok": ok, "title": title, "text": text, "error": "" if ok else "too_thin"}


def collect_nextias_content() -> dict:
    hub_html = fetch_html(BASE_URL)
    hub_soup = BeautifulSoup(hub_html, "html.parser")

    headlines = get_section_items_by_heading(hub_soup, r"Headlines\s*of\s*the\s*Day")
    ca_links = extract_latest_ca_links(hub_soup, limit=12)

    articles = []
    for link in ca_links:
        art = extract_article_text(link)
        if art["ok"]:
            articles.append(art)

    return {
        "hub_url": BASE_URL,
        "headlines_of_the_day": headlines,
        "daily_ca_articles": articles,
    }


# -------------------- MCQ GENERATION --------------------
def build_prompt(payload: dict, hist: dict) -> str:
    used_hashes = set(hist.get("used_fact_hashes", []))

    headlines = payload.get("headlines_of_the_day", [])
    headlines_block = "\n".join([f"- {h}" for h in headlines[:40]])

    article_blocks = []
    for a in payload.get("daily_ca_articles", [])[:10]:
        article_blocks.append(
            f"TITLE: {a['title']}\nURL: {a['url']}\nCONTENT:\n{a['text']}\n"
        )
    articles_block = "\n\n".join(article_blocks)

    used_hashes_block = "\n".join(list(used_hashes)[-300:])

    return f"""
You are a UPSC Prelims MCQ setter.

TASK:
Generate EXACTLY {MAX_MCQS} UPSC Prelims-standard Current Affairs MCQs using ONLY the provided NextIAS content.

CRITICAL EXAM RULES:
1) Use ONLY facts explicitly present in the content. Do NOT use outside knowledge.
2) UPSC framing: direct factual questions; avoid analytical framing.
3) ONE MCQ per distinct news topic (NO repetition of the same event/theme).
4) Prefer high-value items: first/largest, govt initiatives, indices/reports, major national/international developments.
5) Avoid routine advisories/court procedural trivia unless it is clearly a major, UPSC-relevant issue.
6) EXACTLY 4 options; only ONE correct; plausible same-category options.
7) No explanations.

ANTI-REPEAT:
Avoid reusing facts that match these historical hashes (best effort):
{used_hashes_block}

OUTPUT JSON ONLY in this schema:
{{
  "mcqs": [
    {{
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_option_index": 0,
      "source_url": "must be either the hub URL or one of the article URLs",
      "fact_fingerprint": "short unique phrase describing the key fact (e.g., 'LEDC formed under I&B')"
    }}
  ]
}}

INPUTS:

HUB URL:
{payload.get("hub_url")}

HEADLINES OF THE DAY:
{headlines_block}

DAILY CURRENT AFFAIRS ARTICLES:
{articles_block}
""".strip()


def generate_mcqs(payload: dict, hist: dict) -> dict:
    prompt = build_prompt(payload, hist)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Return JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )
    content = resp.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            raise RuntimeError(f"Model did not return JSON. Got:\n{content[:800]}")
        return json.loads(m.group(0))


def normalize_and_shuffle(mcq: dict, allowed_sources: set[str]) -> dict:
    q = clean_text(mcq.get("question", ""))
    opts = mcq.get("options", [])
    idx = mcq.get("correct_option_index", None)
    src = clean_text(mcq.get("source_url", ""))
    fp = clean_text(mcq.get("fact_fingerprint", ""))

    if not q or not isinstance(opts, list) or len(opts) != 4 or idx not in [0, 1, 2, 3]:
        raise ValueError("Invalid MCQ structure from model")

    opts = [clean_text(o) for o in opts]
    if len(set(opts)) != 4 or any(not o for o in opts):
        raise ValueError("Options must be 4 unique non-empty strings")

    if src not in allowed_sources:
        raise ValueError(f"source_url not allowed: {src}")

    correct_text = opts[idx]
    shuffled = opts[:]
    random.shuffle(shuffled)
    new_idx = shuffled.index(correct_text)

    return {
        "question": q,
        "options": shuffled,
        "correct_option_index": new_idx,
        "source_url": src,
        "fact_fingerprint": fp or q[:60],
    }


# -------------------- MAIN --------------------
def main():
    date_str = dt.datetime.now().strftime("%Y-%m-%d")
    hist = load_history()

    payload = collect_nextias_content()
    allowed_sources = {payload["hub_url"]} | {a["url"] for a in payload["daily_ca_articles"]}

    if not payload["headlines_of_the_day"] and not payload["daily_ca_articles"]:
        post_header_with_date(date_str)
        tg(
            "sendMessage",
            {"chat_id": TELEGRAM_CHAT_ID, "text": "No usable current affairs content could be extracted today."},
        )
        return

    data = generate_mcqs(payload, hist)
    mcqs_raw = data.get("mcqs", [])
    if not isinstance(mcqs_raw, list) or len(mcqs_raw) == 0:
        post_header_with_date(date_str)
        tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "No MCQs generated today (insufficient content)."})
        return

    # Normalize + validate + shuffle
    mcqs = []
    for m in mcqs_raw[:MAX_MCQS]:
        mcqs.append(normalize_and_shuffle(m, allowed_sources))

    if len(mcqs) < 3:
        post_header_with_date(date_str)
        tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "Insufficient high-value exam-relevant news today to generate MCQs."})
        return

    # Post sequence: header -> 10 quiz polls -> score poll
    post_header_with_date(date_str)

    for m in mcqs:
        post_quiz_poll(m["question"], m["options"], m["correct_option_index"])
        time.sleep(POLL_DELAY_SECONDS)

    post_score_poll(total_questions=len(mcqs))

    # Update history
    used = hist.get("used_fact_hashes", [])
    for m in mcqs:
        used.append(stable_hash(m["fact_fingerprint"].lower()))
    hist["used_fact_hashes"] = used[-800:]
    save_history(hist)

    print(f"✅ Posted {len(mcqs)} MCQs for {date_str} successfully.")


if __name__ == "__main__":
    main()
