# run_daily.py
# ============================================================
# DAILY TELEGRAM QUIZ POLLS (SSC/PCS Current Affairs) FROM YOUTUBE
# Channels:
# 1) NEXT EXAM  - https://www.youtube.com/@NEXTEXAM/videos
# 2) Crazy GkTrick - https://www.youtube.com/@CrazyGkTrick/videos
#
# How it works:
# - Fetches latest videos from channels via YouTube RSS (public)
# - Pulls transcript (Hindi/English) using youtube-transcript-api
# - Extracts current-affairs FACTS from transcript
# - Builds exam-like MCQs from those facts
# - Posts Telegram "quiz" polls at your cron time
# - Enforces NO repetition across days: video_id + fact hashes + question hashes
#
# Render cron for 3 PM IST (UTC 09:30): 30 9 * * *
# requirements.txt:
#   requests
#   openai
#   youtube-transcript-api
# ============================================================

import os
import re
import json
import time
import hashlib
import random
import datetime as dt
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple

import requests
from openai import OpenAI
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

# -------------------- REQUIRED ENV --------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # e.g., @UPPCSSUCCESS
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# -------------------- OPTIONAL ENV (TUNING) --------------------
MAX_POLLS_PER_DAY = int(os.getenv("MAX_POLLS_PER_DAY", "20"))  # you can increase, but Telegram rate-limits
MIN_POLLS_PER_DAY = int(os.getenv("MIN_POLLS_PER_DAY", "10"))  # minimum required; if less -> fail (no low quality)
MAX_VIDEOS_PER_CHANNEL = int(os.getenv("MAX_VIDEOS_PER_CHANNEL", "4"))
RECENT_WINDOW_DAYS = int(os.getenv("RECENT_WINDOW_DAYS", "2"))  # last 2 days (helps if videos posted late night)
TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "14000"))

SLEEP_BETWEEN_POLLS = float(os.getenv("SLEEP_BETWEEN_POLLS", "2"))
SLEEP_AFTER_QUIZ = float(os.getenv("SLEEP_AFTER_QUIZ", "2"))

HISTORY_FILE = os.getenv("HISTORY_FILE", "yt_mcq_history.json")
KEEP_LAST_DAYS = int(os.getenv("KEEP_LAST_DAYS", "30"))

HTTP_TIMEOUT = 25
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# IST helpers
IST_OFFSET = dt.timedelta(hours=5, minutes=30)

client = OpenAI()  # reads OPENAI_API_KEY

# -------------------- CHANNEL IDS (verified) --------------------
# NEXT EXAM
# Crazy GkTrick
CHANNELS = [
    {"name": "NEXT EXAM", "channel_id": "UC3-u8ek82h70ArrPO4HJypQ"},
    {"name": "Crazy GkTrick", "channel_id": "UCIl28Ab-H-3LYIPw4hhQexA"},
]

# -------------------- TIME HELPERS --------------------
def now_ist() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + IST_OFFSET

def ist_today_date() -> dt.date:
    return now_ist().date()

def ist_today_str() -> str:
    return ist_today_date().isoformat()

def parse_rfc3339(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

def to_ist_date(dt_utc: dt.datetime) -> dt.date:
    return (dt_utc.astimezone(dt.timezone.utc) + IST_OFFSET).date()

# -------------------- TELEGRAM --------------------
def tg(method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    last_err = None
    for _ in range(4):
        r = requests.post(url, json=payload, timeout=30)
        data = r.json()
        if data.get("ok"):
            return data
        last_err = data
        if data.get("error_code") == 429:
            retry_after = (data.get("parameters") or {}).get("retry_after", 5)
            print(f"⚠️ Telegram rate limit. Retrying after {retry_after}s...")
            time.sleep(int(retry_after) + 1)
            continue
        raise RuntimeError(f"Telegram error: {data}")
    raise RuntimeError(f"Telegram error after retries: {last_err}")

def post_header(date_str: str, total: int):
    tg(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🧠 Daily Current Affairs Quiz (SSC/PCS) — {date_str}\n\nTotal Questions: {total}\n\nQuiz polls below 👇",
            "disable_web_page_preview": True,
        },
    )

def post_sources_used(videos: List[Dict[str, Any]]):
    if not videos:
        return
    lines = ["📌 Sources used today (YouTube videos):"]
    for i, v in enumerate(videos, start=1):
        lines.append(f"{i}) {v['channel']} — {v['title']}\n{v['url']}")
    tg(
        "sendMessage",
        {"chat_id": TELEGRAM_CHAT_ID, "text": "\n\n".join(lines), "disable_web_page_preview": True},
    )

def post_closure():
    text = (
        "🏁 Today’s Practice Ends Here!\n\n"
        "Comment your score below 👇\n"
        "⏰ Back tomorrow at the same time."
    )
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True})

def post_score_poll(date_str: str, total: int):
    # Telegram max options = 10, so use bands for large totals
    if total <= 10:
        options = [f"{i}/{total}" for i in range(total, max(total - 9, -1), -1)]
        options = options[:10]
    else:
        options = [
            f"{total} ✅",
            f"{total-1}–{total-2}",
            f"{total-3}–{total-5}",
            f"{total-6}–{total-8}",
            f"{total-9}–{total-12}",
            f"{max(total-13,0)}–{max(total-18,0)}",
            f"{max(total-19,0)}–{max(total-25,0)}",
            "Below that 😅",
        ]
    tg(
        "sendPoll",
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": f"📊 Vote your score ({date_str}) ✅\nHow many were correct?",
            "options": options[:10],
            "is_anonymous": True,
            "allows_multiple_answers": False,
        },
    )

# -------------------- HISTORY (NO REPEAT) --------------------
def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()

def _h(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:24]

def load_history() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {"dates": {}, "video_ids": [], "fact_hashes": [], "q_hashes": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        hist.setdefault("dates", {})
        hist.setdefault("video_ids", [])
        hist.setdefault("fact_hashes", [])
        hist.setdefault("q_hashes", [])
        return hist
    except Exception:
        return {"dates": {}, "video_ids": [], "fact_hashes": [], "q_hashes": []}

def save_history(hist: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def update_history(
    hist: dict,
    date_str: str,
    used_video_ids: List[str],
    used_fact_hashes: List[str],
    used_q_hashes: List[str],
) -> dict:
    hist.setdefault("dates", {})
    hist["dates"][date_str] = {
        "video_ids": used_video_ids,
        "fact_hashes": used_fact_hashes,
        "q_hashes": used_q_hashes,
    }

    # prune old days
    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > KEEP_LAST_DAYS:
        for d in all_dates[:-KEEP_LAST_DAYS]:
            hist["dates"].pop(d, None)

    vids, facts, qs = [], [], []
    for d in sorted(hist["dates"].keys()):
        vids.extend(hist["dates"][d].get("video_ids", []))
        facts.extend(hist["dates"][d].get("fact_hashes", []))
        qs.extend(hist["dates"][d].get("q_hashes", []))

    hist["video_ids"] = vids[-2000:]
    hist["fact_hashes"] = facts[-6000:]
    hist["q_hashes"] = qs[-6000:]
    return hist

# -------------------- YOUTUBE RSS --------------------
def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text

def yt_rss_url(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

def parse_video_id(link: str) -> Optional[str]:
    m = re.search(r"v=([a-zA-Z0-9_-]{8,})", link or "")
    return m.group(1) if m else None

def fetch_recent_videos_from_channel(channel_id: str, channel_name: str, window_days: int) -> List[Dict[str, Any]]:
    """
    Returns videos within last `window_days` by IST date.
    """
    xml_text = http_get(yt_rss_url(channel_id))
    root = ET.fromstring(xml_text)

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
    }

    out = []
    today = ist_today_date()
    earliest = today - dt.timedelta(days=max(window_days, 1) - 1)

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("atom:link", ns)
        link = link_el.attrib.get("href") if link_el is not None else ""
        published = entry.findtext("atom:published", default="", namespaces=ns)
        vid = entry.findtext("yt:videoId", default="", namespaces=ns) or parse_video_id(link)
        if not vid or not published:
            continue

        pub_dt = parse_rfc3339(published)
        pub_ist_date = to_ist_date(pub_dt)
        if not (earliest <= pub_ist_date <= today):
            continue

        out.append(
            {
                "channel": channel_name,
                "video_id": vid,
                "title": title[:200],
                "url": link,
                "published": published,
            }
        )

    out.sort(key=lambda x: x["published"], reverse=True)
    return out[:MAX_VIDEOS_PER_CHANNEL]

# -------------------- TRANSCRIPT --------------------
def get_transcript_text(video_id: str) -> Optional[str]:
    """
    Try Hindi then English. Returns None if not available.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en"])
        text = " ".join([t.get("text", "") for t in transcript]).strip()
        if text:
            return text[:TRANSCRIPT_MAX_CHARS]
        return None
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None
    except Exception:
        return None

# -------------------- JSON SCHEMAS --------------------
FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "minItems": 8,
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "event_key": {"type": "string", "minLength": 3, "maxLength": 80},
                    "fact": {"type": "string", "minLength": 12, "maxLength": 240},
                },
                "required": ["event_key", "fact"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}

MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "event_key": {"type": "string", "minLength": 3, "maxLength": 80},
        "question": {"type": "string", "minLength": 10, "maxLength": 260},
        "options": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
        },
        "correct_option_id": {"type": "integer", "minimum": 0, "maximum": 3},
        "correct_answer": {"type": "string", "minLength": 1, "maxLength": 80},
        "explanation": {"type": "string", "maxLength": 200},
        "source_title": {"type": "string", "minLength": 3, "maxLength": 200},
        "source_url": {"type": "string", "minLength": 5, "maxLength": 400},
        "source": {"type": "string", "minLength": 2, "maxLength": 80},
    },
    "required": [
        "event_key",
        "question",
        "options",
        "correct_option_id",
        "correct_answer",
        "explanation",
        "source_title",
        "source_url",
        "source",
    ],
    "additionalProperties": False,
}

# -------------------- QUALITY --------------------
BAD_OPTIONS = {"all of the above", "none of the above", "all above", "none"}
STATIC_GK_PHRASES = [
    "capital of", "currency of", "national animal", "largest planet", "highest mountain"
]

def is_exam_style(q: str) -> bool:
    q = (q or "").strip()
    low = q.lower()
    if not q.endswith("?"):
        return False
    if not low.startswith(("which", "what", "who", "where", "when")):
        return False
    if low.startswith(("with reference to", "consider the following statements")):
        return False
    if "section" in low or "category" in low:
        return False
    if any(p in low for p in STATIC_GK_PHRASES):
        return False
    return True

def mcq_ok(mcq: Dict[str, Any]) -> bool:
    if not is_exam_style(mcq.get("question", "")):
        return False
    opts = mcq.get("options") or []
    if len(opts) != 4:
        return False
    if len(set(_norm_text(o) for o in opts)) != 4:
        return False
    if any(_norm_text(o) in BAD_OPTIONS for o in opts):
        return False
    ca = mcq.get("correct_answer", "")
    if ca not in opts:
        return False
    mcq["correct_option_id"] = opts.index(ca)
    return True

def shuffle_options(mcq: Dict[str, Any]) -> Dict[str, Any]:
    options = mcq["options"]
    correct_text = options[mcq["correct_option_id"]]
    rnd = random.Random(_h(mcq["question"] + correct_text))
    shuffled = options[:]
    rnd.shuffle(shuffled)
    mcq["options"] = shuffled
    mcq["correct_option_id"] = shuffled.index(correct_text)
    mcq["correct_answer"] = correct_text
    return mcq

# -------------------- LLM: FACTS + MCQ --------------------
def extract_facts_from_transcript(
    video: Dict[str, Any],
    transcript_text: str,
) -> List[Dict[str, str]]:
    system = (
        "Extract ONLY current-affairs facts from the transcript.\n"
        "Facts must be useful for SSC/PCS current affairs.\n"
        "Rules:\n"
        "- Each fact must be a single, testable statement from the transcript.\n"
        "- Focus on: appointments, reports/indices, launches, summits, awards, locations, international events, government decisions.\n"
        "- Avoid static GK, definitions, and generic advice.\n"
        "- Return 8 to 40 facts.\n"
        "- event_key should be 3–8 words and represent the news topic.\n"
    )
    user = {
        "channel": video["channel"],
        "video_title": video["title"],
        "video_url": video["url"],
        "transcript": transcript_text[:TRANSCRIPT_MAX_CHARS],
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "facts_schema",
                "strict": True,
                "schema": FACTS_SCHEMA,
            }
        },
    )
    data = json.loads(resp.output_text)
    return data.get("facts", [])

def generate_mcq_from_fact(video: Dict[str, Any], fact_item: Dict[str, str]) -> Optional[Dict[str, Any]]:
    system = (
        "Create ONE SSC/PCS-standard EASY exam-like MCQ from the given FACT.\n"
        "STRICT RULES:\n"
        "- Use ONLY the FACT. Do not add outside info.\n"
        "- Question must start with Which/What/Who/Where/When and end with '?'.\n"
        "- Options must be same-category and plausible.\n"
        "- No 'All/None of the above'.\n"
        "- Explanation <= 200 characters.\n"
        "- correct_answer must exactly match one option.\n"
    )
    user = {
        "channel": video["channel"],
        "video_title": video["title"],
        "video_url": video["url"],
        "event_key": fact_item.get("event_key", ""),
        "fact": fact_item.get("fact", ""),
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "mcq_schema",
                "strict": True,
                "schema": MCQ_SCHEMA,
            }
        },
    )
    mcq = json.loads(resp.output_text)

    mcq["source"] = f"YouTube: {video['channel']}"
    mcq["source_title"] = video["title"][:200]
    mcq["source_url"] = video["url"][:400]

    if not mcq_ok(mcq):
        return None
    return mcq

# -------------------- BUILD DAILY SET --------------------
def build_daily_mcqs() -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[str]]:
    date_str = ist_today_str()
    hist = load_history()

    used_video_ids = set(hist.get("video_ids", [])[-2000:])
    used_fact_hashes = set(hist.get("fact_hashes", [])[-6000:])
    used_q_hashes = set(hist.get("q_hashes", [])[-6000:])

    # 1) Fetch recent videos
    all_videos: List[Dict[str, Any]] = []
    for ch in CHANNELS:
        vids = fetch_recent_videos_from_channel(ch["channel_id"], ch["name"], RECENT_WINDOW_DAYS)
        print(f"✅ {ch['name']}: {len(vids)} recent videos (window={RECENT_WINDOW_DAYS} days)")
        all_videos.extend(vids)

    # dedup + skip already used
    uniq = []
    seen = set()
    for v in sorted(all_videos, key=lambda x: x["published"], reverse=True):
        if v["video_id"] in seen:
            continue
        seen.add(v["video_id"])
        if v["video_id"] in used_video_ids:
            continue
        uniq.append(v)

    if not uniq:
        raise RuntimeError("No NEW videos found in the recent window, or all already used.")

    # 2) Transcripts
    videos_with_text: List[Tuple[Dict[str, Any], str]] = []
    for v in uniq:
        t = get_transcript_text(v["video_id"])
        if not t:
            print(f"⚠️ Transcript not available: {v['channel']} | {v['title']} | {v['url']}")
            continue
        videos_with_text.append((v, t))

    if not videos_with_text:
        raise RuntimeError("No transcripts available from the selected videos.")

    # 3) Facts -> MCQs
    mcqs: List[Dict[str, Any]] = []
    day_used_vids: List[str] = []
    day_used_fact_hashes: List[str] = []
    day_used_q_hashes: List[str] = []
    videos_actually_used: List[Dict[str, Any]] = []

    for v, t in videos_with_text:
        if len(mcqs) >= MAX_POLLS_PER_DAY:
            break

        facts = extract_facts_from_transcript(v, t)
        if not facts:
            continue

        # mark video as used (only if we extracted facts)
        if v["video_id"] not in day_used_vids:
            day_used_vids.append(v["video_id"])
            used_video_ids.add(v["video_id"])
            videos_actually_used.append(v)

        for f in facts:
            if len(mcqs) >= MAX_POLLS_PER_DAY:
                break

            fact_text = (f.get("fact") or "").strip()
            if len(fact_text) < 12:
                continue

            fh = _h(_norm_text(fact_text))
            if fh in used_fact_hashes:
                continue

            mcq = generate_mcq_from_fact(v, f)
            if not mcq:
                continue

            qh = _h(_norm_text(mcq["question"]))
            if qh in used_q_hashes:
                continue

            used_fact_hashes.add(fh)
            used_q_hashes.add(qh)
            day_used_fact_hashes.append(fh)
            day_used_q_hashes.append(qh)

            mcqs.append(mcq)

    if len(mcqs) < MIN_POLLS_PER_DAY:
        raise RuntimeError(f"Too few MCQs generated today. Got {len(mcqs)}; need at least {MIN_POLLS_PER_DAY}.")

    return date_str, mcqs, videos_actually_used, day_used_vids, day_used_fact_hashes, day_used_q_hashes

# -------------------- POST --------------------
def post_mcqs(date_str: str, mcqs: List[Dict[str, Any]], videos_used: List[Dict[str, Any]]):
    post_sources_used(videos_used)
    post_header(date_str, len(mcqs))

    for i, q in enumerate(mcqs, start=1):
        q = shuffle_options(q)
        tg(
            "sendPoll",
            {
                "chat_id": TELEGRAM_CHAT_ID,
                "question": f"Q{i}. {q['question']}",
                "options": q["options"],
                "type": "quiz",
                "correct_option_id": q["correct_option_id"],
                "explanation": q["explanation"],
                "is_anonymous": True,
            },
        )
        time.sleep(SLEEP_BETWEEN_POLLS)

    time.sleep(SLEEP_AFTER_QUIZ)
    post_closure()
    time.sleep(2)
    post_score_poll(date_str, len(mcqs))

# -------------------- MAIN --------------------
def main():
    date_str, mcqs, videos_used, used_vids, used_facts, used_qs = build_daily_mcqs()
    post_mcqs(date_str, mcqs, videos_used)

    hist = load_history()
    hist = update_history(hist, date_str, used_vids, used_facts, used_qs)
    save_history(hist)

    print(f"✅ Done. Posted {len(mcqs)} MCQs for {date_str}. Videos used: {len(videos_used)}")

if __name__ == "__main__":
    main()
