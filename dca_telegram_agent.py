import os
import re
import json
import logging
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Optional
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from telegram import Update, Poll
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from openai import OpenAI


# ================== CONFIG ==================
IST = ZoneInfo("Asia/Kolkata")

AFFAIRSCLOUD_URL = "https://affairscloud.com/current-affairs/"
ADDA247_URL = "https://currentaffairs.adda247.com/"

# Public channel username (you set via env)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@UPPCSSUCCESS").strip()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# OpenAI key is read automatically from OPENAI_API_KEY env by SDK
client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2").strip()
MCQ_COUNT = int(os.getenv("MCQ_COUNT", "10"))

REQ_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("dca-quiz-poll-bot")


# ================== DATA ==================
@dataclass
class MCQ:
    question: str
    options: List[str]      # 4 options
    correct_index: int      # 0..3
    explanation: str        # short for Telegram quiz poll
    source: str             # short source label


# ================== HELPERS ==================
def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.text


def clean_space(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def shorten_chars(text: str, max_chars: int) -> str:
    text = clean_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def safe_poll_question(q: str) -> str:
    # Telegram poll question limit ~300 chars
    return shorten_chars(q, 290)


def safe_poll_option(o: str) -> str:
    return shorten_chars(o, 95)


def safe_poll_expl(e: str) -> str:
    # Telegram quiz poll explanation is short; keep well under 200 chars
    return shorten_chars(e, 190)


def dedup_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def is_probable_post_url(domain: str, url: str) -> bool:
    if not url.startswith("http"):
        return False
    if domain not in url:
        return False
    bad_parts = [
        "/category/", "/tag/", "/author/", "/page/", "#", "?amp", "/wp-json", "/feed"
    ]
    if any(bp in url for bp in bad_parts):
        return False
    return True


# ================== SCRAPING: GET LATEST POST LINKS ==================
def extract_post_links_affairscloud(html: str, limit: int = 10) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        # AffairsCloud post slugs often include "current-affairs-" (common pattern)
        if "affairscloud.com" in href and "current-affairs-" in href and is_probable_post_url("affairscloud.com", href):
            urls.append(href.split("#")[0])
    return dedup_preserve(urls)[:limit]


def extract_post_links_adda(html: str, limit: int = 10) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        # Keep actual post links, avoid category/tag pages
        if is_probable_post_url("currentaffairs.adda247.com", href):
            # many posts live directly under the domain with a slug
            # exclude the homepage itself
            if href.rstrip("/") == ADDA247_URL.rstrip("/"):
                continue
            urls.append(href.split("#")[0])
    return dedup_preserve(urls)[:limit]


def fetch_article_text(url: str, max_chars: int = 1400) -> Tuple[str, str]:
    """
    Returns: (title, summary_text)
    """
    html = http_get(url)
    soup = BeautifulSoup(html, "lxml")

    # title
    title = ""
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # main content best-effort
    main = soup.select_one(".entry-content") or soup.select_one("article") or soup.body
    text = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)

    text = clean_space(text)
    text = shorten_chars(text, max_chars)

    return shorten_chars(title, 120), text


def build_source_digest() -> str:
    """
    Fetches a handful of latest posts from both sites and builds a compact digest
    for the model to generate MCQs from.
    """
    digest_parts = []

    # AffairsCloud
    try:
        ac_html = http_get(AFFAIRSCLOUD_URL)
        ac_links = extract_post_links_affairscloud(ac_html, limit=7)
        ac_items = []
        for url in ac_links[:5]:
            try:
                t, body = fetch_article_text(url, max_chars=900)
                ac_items.append(f"- Title: {t}\n  Key text: {body}\n  Link: {url}")
            except Exception as e:
                log.warning("AffairsCloud article fetch failed: %s (%s)", url, e)
        if ac_items:
            digest_parts.append("SOURCE 1 (AffairsCloud current affairs):\n" + "\n".join(ac_items))
    except Exception as e:
        log.warning("AffairsCloud scrape failed: %s", e)

    # Adda247
    try:
        adda_html = http_get(ADDA247_URL)
        adda_links = extract_post_links_adda(adda_html, limit=10)
        adda_items = []
        for url in adda_links[:5]:
            try:
                t, body = fetch_article_text(url, max_chars=900)
                adda_items.append(f"- Title: {t}\n  Key text: {body}\n  Link: {url}")
            except Exception as e:
                log.warning("Adda article fetch failed: %s (%s)", url, e)
        if adda_items:
            digest_parts.append("SOURCE 2 (Adda247 current affairs):\n" + "\n".join(adda_items))
    except Exception as e:
        log.warning("Adda247 scrape failed: %s", e)

    if not digest_parts:
        # fallback minimal prompt if scraping fails
        return "No source text could be fetched right now. Create general current affairs MCQs."

    return "\n\n".join(digest_parts)


# ================== OPENAI: GENERATE MCQS (Option 1) ==================
def extract_json_array(text: str) -> str:
    """
    If model returns extra text, try to extract the first JSON array.
    """
    text = text.strip()
    # direct JSON array
    if text.startswith("[") and text.endswith("]"):
        return text
    # find first [...] block
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        return m.group(0)
    raise ValueError("Could not extract JSON array from model output.")


def generate_mcqs_from_digest(digest: str, n: int = 10) -> List[MCQ]:
    prompt = f"""
You are an exam content creator for UPPSC/competitive exams.

Using ONLY the factual information in the SOURCES below, create exactly {n} MCQs.

Output MUST be ONLY a valid JSON array (no markdown, no extra text).

Each JSON item must have:
- "question": string (max 250 chars)
- "options": array of exactly 4 strings
- "correct_index": integer 0-3
- "explanation": string (max 180 chars; crisp, factual)
- "source": string (either "AffairsCloud" or "Adda247")

Rules:
- No hallucinations. If a fact is not present in the sources, do not use it.
- Avoid ultra-specific numbers/dates unless clearly present in sources.
- Make options plausible, but only one correct.
- Explanations must be short.

SOURCES:
{digest}
""".strip()

    # OPTION 1: plain string input (no content[].type) — this avoids your earlier error
    resp = client.responses.create(
        model=MODEL,
        input=prompt
    )

    raw = resp.output_text.strip()
    json_text = extract_json_array(raw)
    data = json.loads(json_text)

    mcqs: List[MCQ] = []
    for item in data:
        q = safe_poll_question(str(item["question"]))
        opts = [safe_poll_option(x) for x in item["options"]]
        ci = int(item["correct_index"])
        expl = safe_poll_expl(str(item["explanation"]))
        src = shorten_chars(str(item.get("source", "")), 20) or "Source"

        if len(opts) != 4 or not (0 <= ci <= 3):
            continue

        mcqs.append(MCQ(question=q, options=opts, correct_index=ci, explanation=expl, source=src))

    return mcqs[:n]


# ================== TELEGRAM: POST QUIZ POLLS ==================
async def post_mcqs_to_channel(context: ContextTypes.DEFAULT_TYPE) -> None:
    digest = build_source_digest()
    mcqs = generate_mcqs_from_digest(digest, n=MCQ_COUNT)

    if not mcqs:
        await context.bot.send_message(chat_id=CHAT_ID, text="⚠️ Could not generate MCQs today.")
        return

    today = dt.datetime.now(IST).strftime("%d %b %Y")
    await context.bot.send_message(
        chat_id=CHAT_ID,
        text=f"📌 *Daily Current Affairs Quiz* — {today}\n✅ {len(mcqs)} Quiz Polls",
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True
    )

    import asyncio
    for i, mcq in enumerate(mcqs, 1):
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=f"🧠 *MCQ #{i}*  _(Source: {mcq.source})_",
            parse_mode=ParseMode.MARKDOWN
        )

        await context.bot.send_poll(
            chat_id=CHAT_ID,
            question=mcq.question,
            options=mcq.options,
            type=Poll.QUIZ,
            correct_option_id=mcq.correct_index,
            is_anonymous=True,
            allows_multiple_answers=False,
            explanation=mcq.explanation
        )

        await asyncio.sleep(1.0)  # pacing


# ================== COMMANDS ==================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ Bot is running.\n\n"
        "Note: Commands won’t work inside a channel.\n"
        "Use these in my private chat:\n"
        "/now  -> Post today’s 10 quiz polls now"
    )


async def now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Posting quiz polls to channel now…")
    await post_mcqs_to_channel(context)


# ================== MAIN ==================
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("ERROR: Set TELEGRAM_BOT_TOKEN env var.")
    if not CHAT_ID:
        raise SystemExit('ERROR: Set TELEGRAM_CHAT_ID env var (e.g., "@UPPCSSUCCESS").')

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("now", now_cmd))

    # Daily schedule at 10:00 AM IST
    app.job_queue.run_daily(
        post_mcqs_to_channel,
        time=dt.time(hour=10, minute=0, tzinfo=IST),
        name="daily_quiz_polls_10am_ist"
    )

    log.info("Bot started. Posting daily at 10:00 IST to %s", CHAT_ID)
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
