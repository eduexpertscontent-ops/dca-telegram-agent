import os
import re
import json
import html
import time
import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

# -------------------------
# CONFIG
# -------------------------
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "@DCAUPSC")

# 7:00 AM IST run via cron (recommended). This script assumes "run now".
IST = timezone(timedelta(hours=5, minutes=30))
today_ist = datetime.now(IST).date()
TODAY_STR_LONG = datetime.now(IST).strftime("%d %b %Y")  # e.g., "02 Jan 2026"
TODAY_STR_PW = datetime.now(IST).strftime("%d-%b-%Y")    # e.g., "02-Jan-2026"

# Sources
PIB_RSS = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"  # PIB Press Releases RSS
RBI_PR_LIST = "https://www.rbi.org.in/commonman/english/scripts/PressReleases.aspx"
PRS_BILLTRACK = "https://prsindia.org/billtrack"
PRS_REPORTS = "https://prsindia.org/policy/report-summaries"
INSIGHTS_CA = "https://www.insightsonindia.com/current-affairs-upsc/"
PW_DAILY = "https://pwonlyias.com/daily-current-affairs/"
PW_CA_HUB = "https://pwonlyias.com/current-affairs/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DCAUPSCBot/1.0; +https://telegram.org)"
}

# -------------------------
# TELEGRAM HELPERS
# -------------------------
def tg_api(method: str, payload: dict):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def tg_send_message(text: str):
    return tg_api("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    })

def tg_send_poll(question: str, options: list, correct_option_id: int):
    # Telegram poll question limits: question<=300, options<=100 each.
    question = question[:295]
    options = [opt[:95] for opt in options][:4]
    return tg_api("sendPoll", {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": question,
        "options": options,
        "type": "quiz",
        "correct_option_id": correct_option_id,
        "is_anonymous": False
    })

# -------------------------
# FETCHERS
# -------------------------
def fetch_pib_items(max_items=12):
    d = feedparser.parse(PIB_RSS)
    items = []
    for e in d.entries[:max_items]:
        published = getattr(e, "published", "") or getattr(e, "updated", "")
        items.append({
            "source": "PIB",
            "title": e.get("title", "").strip(),
            "url": e.get("link", "").strip(),
            "published": published,
            "summary": re.sub(r"\s+", " ", html.unescape(re.sub("<.*?>", " ", e.get("summary", "")))).strip()
        })
    return items

def fetch_rbi_press_releases(max_items=10):
    r = requests.get(RBI_PR_LIST, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # RBI page contains date blocks and links; we pick top links visible.
    links = soup.select("a")
    items = []
    for a in links:
        href = a.get("href") or ""
        text = a.get_text(" ", strip=True)
        if not text or "Image" in text:
            continue
        # heuristic: RBI press release detail pages often contain "Scripts/BS_PressReleaseDisplay.aspx" or similar
        if "PressRelease" in href or "BS_PressReleaseDisplay" in href or "pressreleases" in href.lower():
            url = href
            if url.startswith("/"):
                url = "https://www.rbi.org.in" + url
            if url.startswith("Scripts/") or url.startswith("commonman/"):
                url = "https://www.rbi.org.in/" + url

            items.append({
                "source": "RBI",
                "title": text,
                "url": url,
                "published": "",
                "summary": ""
            })
        if len(items) >= max_items:
            break
    # de-dup by url
    dedup = {}
    for it in items:
        dedup[it["url"]] = it
    return list(dedup.values())[:max_items]

def fetch_prs_items(max_items=10):
    items = []
    for page_url, label in [(PRS_BILLTRACK, "PRS BillTrack"), (PRS_REPORTS, "PRS Reports")]:
        r = requests.get(page_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        for a in soup.select("a"):
            href = a.get("href") or ""
            text = a.get_text(" ", strip=True)
            if not text or len(text) < 10:
                continue
            if href.startswith("/"):
                href = "https://prsindia.org" + href
            if "prsindia.org" not in href:
                continue
            # Keep only meaningful policy/bill links
            if any(x in href for x in ["/billtrack/", "/policy/"]):
                items.append({
                    "source": label,
                    "title": text,
                    "url": href,
                    "published": "",
                    "summary": ""
                })
            if len(items) >= max_items:
                break

    # de-dup
    dedup = {}
    for it in items:
        dedup[it["url"]] = it
    return list(dedup.values())[:max_items]

def fetch_insights_today_links(max_items=10):
    r = requests.get(INSIGHTS_CA, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Page lists day-wise links; we try to find link containing today's date in text.
    # Insights text format often: "UPSC Current Affairs – 2 January 2026"
    items = []
    date_patterns = [
        datetime.now(IST).strftime("%-d %B %Y"),  # may fail on windows-like systems; keep fallback
        datetime.now(IST).strftime("%d %B %Y").lstrip("0"),
        datetime.now(IST).strftime("%d %B %Y"),
        datetime.now(IST).strftime("%-d %B %Y") if hasattr(datetime.now(IST), "strftime") else ""
    ]
    date_patterns = [p for p in date_patterns if p]

    for a in soup.select("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        if "insightsonindia.com" not in href:
            continue
        if "upsc-current-affairs" in href and any(dp in text for dp in date_patterns):
            items.append({
                "source": "Insights",
                "title": text,
                "url": href,
                "published": "",
                "summary": ""
            })
        if len(items) >= max_items:
            break

    return items[:max_items]

def fetch_pw_today_links(max_items=10):
    r = requests.get(PW_DAILY, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # The page lists dates like "02-Jan-2026"; we find anchor containing that.
    items = []
    for a in soup.select("a"):
        text = a.get_text(" ", strip=True)
        href = a.get("href") or ""
        if TODAY_STR_PW in text or TODAY_STR_PW in href:
            if href.startswith("/"):
                href = "https://pwonlyias.com" + href
            if href.startswith("https://pwonlyias.com/"):
                items.append({
                    "source": "PWOnlyIAS",
                    "title": f"PW Daily Updates – {TODAY_STR_PW}",
                    "url": href,
                    "published": "",
                    "summary": ""
                })
        if len(items) >= 1:
            break
    return items[:max_items]

def fetch_page_text(url: str, max_chars=4000):
    """Fetch and extract readable text for grounding the agent."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts/styles
        for s in soup(["script", "style", "noscript"]):
            s.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""

# -------------------------
# OPENAI (Responses API)
# -------------------------
def openai_generate(payload_items):
    """
    Uses OpenAI Responses API.
    Requires: pip install openai
    """
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Attach short extracted text for each URL (grounding)
    grounded = []
    for it in payload_items:
        body = fetch_page_text(it["url"]) if it.get("url") else it.get("summary", "")
        grounded.append({
            **it,
            "content_excerpt": body or it.get("summary", "")
        })

    system = (
        "You are an UPSC current-affairs question setter. "
        "Create content ONLY from the provided items. "
        "If a fact is missing/uncertain, DO NOT invent. Drop that question. "
        "Difficulty: MODERATE. UPSC Prelims style."
    )

    user = {
        "today_date_ist": str(today_ist),
        "task": "Generate 10 MCQ polls + 2 mains questions from ONLY these items.",
        "output_format": {
            "mcqs": [
                {
                    "question": "string",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 0,
                    "explanation": "2-4 lines, purely from the items"
                }
            ],
            "mains": [
                {
                    "question": "UPSC mains style question",
                    "model_points": ["Intro", "Body points", "Conclusion"]
                }
            ],
            "dca_brief": ["5-10 bullets, each bullet tied to one of the items"]
        },
        "rules": [
            "STRICTLY use ONLY the provided items as source material.",
            "Do not use outside knowledge, even if you know it.",
            "No hallucinations. If not enough data for 10 MCQs, return fewer and explain in a field 'notes'.",
            "Keep MCQs varied across topics present in the items.",
            "Include scheme/act/institution name ONLY if present in items."
        ],
        "items": grounded
    }

    resp = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}
        ],
        # No web_search tool here because YOU asked: "strictly from those items".
        # We are already fetching the items ourselves.
    )

    text = resp.output_text
    # Ensure JSON parseable
    data = json.loads(text)
    return data

# -------------------------
# MAIN PIPELINE
# -------------------------
def run():
    items = []
    items += fetch_pib_items()
    items += fetch_rbi_press_releases()
    items += fetch_prs_items()
    items += fetch_insights_today_links()
    items += fetch_pw_today_links()

    # Keep only items that have at least title+url
    items = [it for it in items if it.get("title") and it.get("url")]

    # Hard cap to keep token cost stable
    items = items[:25]

    if not items:
        tg_send_message("⚠️ No items fetched today. Please check source availability.")
        return

    data = openai_generate(items)

    # 1) DCA Brief
    dca_lines = data.get("dca_brief", [])[:12]
    dca_msg = "🗓️ <b>DCA Brief</b> (" + TODAY_STR_LONG + ")\n\n" + "\n".join([f"• {html.escape(x)}" for x in dca_lines])
    tg_send_message(dca_msg)

    # 2) 10 MCQ Polls (Quiz)
    mcqs = data.get("mcqs", [])[:10]
    if not mcqs:
        tg_send_message("⚠️ Not enough grounded info to generate MCQ polls today (strict mode).")
        return

    tg_send_message("✅ <b>MCQ POLLS (Moderate)</b> — Answer in polls, explanation after each poll.")

    for i, q in enumerate(mcqs, 1):
        question = f"Q{i}. {q['question']}"
        options = q["options"]
        correct_index = int(q["correct_index"])

        poll_res = tg_send_poll(question, options, correct_index)

        # Follow-up explanation as message
        ans_letter = ["A", "B", "C", "D"][correct_index]
        exp = q.get("explanation", "").strip()
        explain_msg = f"✅ <b>Answer:</b> {ans_letter}\n🧠 <b>Explanation:</b> {html.escape(exp)}"
        tg_send_message(explain_msg)

        time.sleep(1.2)  # gentle pacing to avoid Telegram burst limits

    # 3) 2 Mains
    mains = data.get("mains", [])[:2]
    if mains:
        mains_text = f"📝 <b>Mains Practice (2 Qs)</b> — {TODAY_STR_LONG}\n"
        for idx, m in enumerate(mains, 1):
            mains_text += f"\n<b>Q{idx}.</b> {html.escape(m['question'])}\n"
            pts = m.get("model_points", [])
            for p in pts[:10]:
                mains_text += f"• {html.escape(p)}\n"
        tg_send_message(mains_text)

if __name__ == "__main__":
    run()
