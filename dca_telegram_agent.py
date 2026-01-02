import os
import re
import sqlite3
import logging
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from telegram import Update, Poll
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

# ---------------- CONFIG ----------------
IST = ZoneInfo("Asia/Kolkata")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()   # public channel: "@YourChannelUsername"
MCQ_COUNT = int(os.getenv("MCQ_COUNT", "10"))

DB_PATH = os.getenv("DB_PATH", "dca_poll_bot.db")

ADDA_HUB = "https://currentaffairs.adda247.com/current-affairs-quiz/"
AFFAIRSCLOUD_HUB = "https://affairscloud.com/current-affairs-quiz/"

REQ_TIMEOUT = 25
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("dca-quiz-poll-bot")


# ---------------- DATA ----------------
@dataclass
class MCQ:
    question: str
    options: List[str]      # 4 options
    correct_index: int      # 0..3
    explanation: str        # short for quiz poll
    source_url: str


# ---------------- DB ----------------
def db_init() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS posted_urls (
                url TEXT PRIMARY KEY,
                posted_at TEXT NOT NULL
            )
        """)
        conn.commit()


def db_is_posted(url: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM posted_urls WHERE url = ? LIMIT 1", (url,))
        return cur.fetchone() is not None


def db_mark_posted(url: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO posted_urls(url, posted_at) VALUES(?, ?)",
            (url, dt.datetime.now(IST).isoformat())
        )
        conn.commit()


# ---------------- HELPERS ----------------
def get_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.text


def clean_space(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def shorten_chars(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def safe_question(text: str) -> str:
    # Poll question limit ~300 chars
    return shorten_chars(text, 290)


def safe_option(text: str) -> str:
    return shorten_chars(text, 95)


def safe_explanation(text: str) -> str:
    # Quiz poll explanation limit is small; keep below ~200 chars
    return shorten_chars(text, 190)


def dedup_preserve(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_links(html: str, must_contain_any: Tuple[str, ...]) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        if any(key in href for key in must_contain_any):
            urls.append(href.split("#")[0])
    return dedup_preserve(urls)


# ---------------- DISCOVERY ----------------
def find_adda_quiz_urls(limit: int = 6) -> List[str]:
    html = get_html(ADDA_HUB)
    # Adda uses multiple quiz slugs over time
    return extract_links(html, ("daily-current-affairs-quiz", "latest-daily-current-affairs-quiz"))[:limit]


def find_affairscloud_quiz_urls(limit: int = 6) -> List[str]:
    html = get_html(AFFAIRSCLOUD_HUB)
    return extract_links(html, ("current-affairs-quiz-",))[:limit]


# ---------------- PARSERS ----------------
def parse_adda_quiz_page(url: str) -> List[MCQ]:
    html = get_html(url)
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one(".entry-content") or soup.select_one("article") or soup
    text = clean_space(content.get_text("\n", strip=True))

    blocks = re.split(r"\n(?=Q\d+\.)", text)
    out: List[MCQ] = []

    for b in blocks:
        b = b.strip()
        if not re.match(r"^Q\d+\.", b):
            continue

        q = re.sub(r"^Q\d+\.\s*", "", b.splitlines()[0]).strip()

        opts = []
        for L in ["A", "B", "C", "D"]:
            m = re.search(rf"\(\s*{L}\s*\)\.?\s*(.+)", b)
            if m:
                opts.append(m.group(1).strip())
        if len(opts) != 4:
            continue

        ans_m = re.search(r"Answer:\s*([a-dA-D])", b)
        if not ans_m:
            continue
        correct_index = "ABCD".index(ans_m.group(1).upper())

        expl = ""
        expl_m = re.search(r"Explanation:\s*(.+)", b, flags=re.DOTALL)
        if expl_m:
            expl_raw = expl_m.group(1)
            expl_raw = re.split(r"\nInformation Booster|\nQ\d+\.", expl_raw)[0]
            expl = clean_space(expl_raw)

        if not expl:
            expl = "Based on the related current affairs update."

        out.append(MCQ(
            question=safe_question(q),
            options=[safe_option(x) for x in opts],
            correct_index=correct_index,
            explanation=safe_explanation(expl),
            source_url=url
        ))

    return out


def parse_affairscloud_quiz_page(url: str) -> List[MCQ]:
    html = get_html(url)
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one(".entry-content") or soup.select_one("article") or soup
    text = clean_space(content.get_text("\n", strip=True))

    # Normalize "a) option" -> "(A) option"
    text = re.sub(r"^\s*([a-dA-D])[\).\]]\s*", r"(\1) ", text, flags=re.MULTILINE)

    blocks = re.split(r"\n(?=Q\d+[\.\)])", text)
    out: List[MCQ] = []

    for b in blocks:
        b = b.strip()
        if not re.match(r"^Q\d+[\.\)]", b):
            continue

        first_line = b.splitlines()[0]
        q = re.sub(r"^Q\d+[\.\)]\s*", "", first_line).strip()

        opts_map = {}
        for L in ["A", "B", "C", "D"]:
            m = re.search(rf"\(\s*{L}\s*\)\s*(.+)", b)
            if m:
                opts_map[L] = m.group(1).strip()
        if len(opts_map) != 4:
            continue

        ans_m = re.search(r"(Ans|Answer)\s*[:\-]\s*([a-dA-D])", b)
        if not ans_m:
            continue
        correct_index = "ABCD".index(ans_m.group(2).upper())

        expl = ""
        expl_m = re.search(r"Explanation\s*[:\-]\s*(.+)", b, flags=re.DOTALL)
        if expl_m:
            expl = clean_space(expl_m.group(1))
            expl = re.split(r"\nQ\d+[\.\)]", expl)[0].strip()
        if not expl:
            expl = "Refer to the linked quiz page for full context."

        opts = [opts_map["A"], opts_map["B"], opts_map["C"], opts_map["D"]]
        out.append(MCQ(
            question=safe_question(q),
            options=[safe_option(x) for x in opts],
            correct_index=correct_index,
            explanation=safe_explanation(expl),
            source_url=url
        ))

    return out


# ---------------- COLLECTION ----------------
def collect_mcqs(target: int) -> Tuple[List[MCQ], List[str]]:
    mcqs: List[MCQ] = []
    used_sources: List[str] = []

    # 1) Adda247 first
    for url in find_adda_quiz_urls():
        if db_is_posted(url):
            continue
        try:
            items = parse_adda_quiz_page(url)
            if items:
                mcqs.extend(items)
                used_sources.append(url)
                db_mark_posted(url)
        except Exception as e:
            log.warning("Adda parse failed (%s): %s", url, e)
        if len(mcqs) >= target:
            return mcqs[:target], used_sources

    # 2) AffairsCloud fallback
    for url in find_affairscloud_quiz_urls():
        if db_is_posted(url):
            continue
        try:
            items = parse_affairscloud_quiz_page(url)
            if items:
                mcqs.extend(items)
                used_sources.append(url)
                db_mark_posted(url)
        except Exception as e:
            log.warning("AffairsCloud parse failed (%s): %s", url, e)
        if len(mcqs) >= target:
            break

    return mcqs[:target], used_sources


# ---------------- SENDER ----------------
async def post_daily_mcqs(context: ContextTypes.DEFAULT_TYPE) -> None:
    db_init()

    mcqs, sources = collect_mcqs(MCQ_COUNT)
    if not mcqs:
        await context.bot.send_message(chat_id=CHAT_ID, text="⚠️ Could not fetch MCQs today.")
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
        # Small label message (optional)
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=f"🧠 *MCQ #{i}*",
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

    # Send sources used (optional)
    if sources:
        msg = "🔗 *Sources used today:*\n" + "\n".join(sources[:5])
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=msg,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True
        )


# ---------------- COMMANDS ----------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✅ DCA Quiz Poll Bot is running.\n"
        "Commands:\n"
        "/now  -> post today's 10 quiz polls now\n"
    )


async def now_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Posting quiz polls now…")
    await post_daily_mcqs(context)


# ---------------- MAIN ----------------
def main() -> None:
    if not BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var.")
    if not CHAT_ID:
        raise SystemExit('Set TELEGRAM_CHAT_ID env var (e.g., "@YourPublicChannel").')

    db_init()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("now", now_cmd))

    # Daily schedule at 10:00 AM IST
    app.job_queue.run_daily(
        post_daily_mcqs,
        time=dt.time(hour=10, minute=0, tzinfo=IST),
        name="daily_quiz_polls_10am_ist"
    )

    log.info("Bot started for public channel %s. Scheduled 10:00 IST daily.", CHAT_ID)
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
