import os
import re
import json
import asyncio
import datetime as dt
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from telegram import Bot, Poll
from openai import OpenAI

IST = ZoneInfo("Asia/Kolkata")

AFFAIRSCLOUD_URL = "https://affairscloud.com/current-affairs/"
ADDA247_URL = "https://currentaffairs.adda247.com/"

REQ_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
MCQ_COUNT = int(os.getenv("MCQ_COUNT", "10"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@UPPCSSUCCESS").strip()

client = OpenAI()


@dataclass
class MCQ:
    question: str
    options: List[str]
    correct_index: int
    explanation: str
    source: str


def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.text


def clean_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def shorten(text: str, n: int) -> str:
    text = clean_space(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def safe_q(q: str) -> str:
    return shorten(q, 290)     # poll question limit ~300


def safe_opt(o: str) -> str:
    return shorten(o, 95)


def safe_exp(e: str) -> str:
    return shorten(e, 190)     # quiz poll explanation is short


def dedup_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_links(domain: str, html: str, must_contain: str, limit: int = 7) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if href.startswith("http") and domain in href and must_contain in href:
            urls.append(href.split("#")[0])
    return dedup_preserve(urls)[:limit]


def fetch_article_text(url: str, max_chars: int = 900) -> Tuple[str, str]:
    html = http_get(url)
    soup = BeautifulSoup(html, "lxml")

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    main = soup.select_one(".entry-content") or soup.select_one("article") or soup.body
    text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)

    return shorten(title, 120), shorten(text, max_chars)


def build_digest() -> str:
    parts = []

    # AffairsCloud links often contain "current-affairs-"
    try:
        ac_html = http_get(AFFAIRSCLOUD_URL)
        ac_links = extract_links("affairscloud.com", ac_html, "current-affairs-", limit=6)
        ac_items = []
        for u in ac_links[:4]:
            t, body = fetch_article_text(u)
            ac_items.append(f"- Title: {t}\n  Text: {body}\n  Link: {u}")
        if ac_items:
            parts.append("SOURCE 1 (AffairsCloud):\n" + "\n".join(ac_items))
    except Exception:
        pass

    # Adda247 homepage contains many post links; we just take some and summarize
    try:
        adda_html = http_get(ADDA247_URL)
        # take any internal posts (best-effort)
        soup = BeautifulSoup(adda_html, "lxml")
        urls = []
        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if href.startswith("http") and "currentaffairs.adda247.com" in href:
                if href.rstrip("/") != ADDA247_URL.rstrip("/"):
                    urls.append(href.split("#")[0])
        urls = dedup_preserve(urls)[:6]

        adda_items = []
        for u in urls[:4]:
            t, body = fetch_article_text(u)
            adda_items.append(f"- Title: {t}\n  Text: {body}\n  Link: {u}")
        if adda_items:
            parts.append("SOURCE 2 (Adda247):\n" + "\n".join(adda_items))
    except Exception:
        pass

    return "\n\n".join(parts) if parts else "No source text fetched. Create general current affairs MCQs."


def extract_json_array(s: str) -> str:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return s
    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        return m.group(0)
    raise ValueError("Model did not return a JSON array.")


def generate_mcqs(digest: str) -> List[MCQ]:
    prompt = f"""
Create exactly {MCQ_COUNT} MCQs for UPPSC current affairs using ONLY the facts in SOURCES.

Return ONLY valid JSON array. No extra text.

Each item:
- question (max 250 chars)
- options (exactly 4)
- correct_index (0-3)
- explanation (max 180 chars, crisp)
- source ("AffairsCloud" or "Adda247")

SOURCES:
{digest}
""".strip()

    # OPTION 1: plain string input
    resp = client.responses.create(model=MODEL, input=prompt)
    raw = resp.output_text.strip()

    data = json.loads(extract_json_array(raw))
    out = []
    for it in data:
        opts = it["options"]
        ci = int(it["correct_index"])
        if not isinstance(opts, list) or len(opts) != 4 or not (0 <= ci <= 3):
            continue
        out.append(MCQ(
            question=safe_q(str(it["question"])),
            options=[safe_opt(str(x)) for x in opts],
            correct_index=ci,
            explanation=safe_exp(str(it["explanation"])),
            source=shorten(str(it.get("source", "Source")), 20)
        ))
    return out[:MCQ_COUNT]


async def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN env var")
    if not TELEGRAM_CHAT_ID:
        raise SystemExit("Missing TELEGRAM_CHAT_ID env var")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    digest = build_digest()
    mcqs = generate_mcqs(digest)

    if not mcqs:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="⚠️ Could not generate MCQs today.")
        return

    today = dt.datetime.now(IST).strftime("%d %b %Y")
    await bot.send_message(
        chat_id=TELEGRAM_CHAT_ID,
        text=f"📌 Daily Current Affairs Quiz — {today}\n✅ {len(mcqs)} Quiz Polls",
        disable_web_page_preview=True
    )

    for i, mcq in enumerate(mcqs, 1):
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=f"🧠 MCQ #{i} (Source: {mcq.source})"
        )
        await bot.send_poll(
            chat_id=TELEGRAM_CHAT_ID,
            question=mcq.question,
            options=mcq.options,
            type=Poll.QUIZ,
            correct_option_id=mcq.correct_index,
            explanation=mcq.explanation,
            is_anonymous=True,
            allows_multiple_answers=False
        )
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(main())
