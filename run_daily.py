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
from bs4 import BeautifulSoup
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

# -------------------- OPTIONAL ENV --------------------
MAX_POLLS_PER_DAY = int(os.getenv("MAX_POLLS_PER_DAY", "20"))
MIN_POLLS_PER_DAY = int(os.getenv("MIN_POLLS_PER_DAY", "10"))
MAX_VIDEOS_PER_CHANNEL = int(os.getenv("MAX_VIDEOS_PER_CHANNEL", "6"))

# We will try 2 days first, then expand to 4 if transcripts/descriptions are thin
RECENT_WINDOW_DAYS_PRIMARY = int(os.getenv("RECENT_WINDOW_DAYS_PRIMARY", "2"))
RECENT_WINDOW_DAYS_FALLBACK = int(os.getenv("RECENT_WINDOW_DAYS_FALLBACK", "4"))

TRANSCRIPT_MAX_CHARS = int(os.getenv("TRANSCRIPT_MAX_CHARS", "14000"))
TEXT_MAX_CHARS = int(os.getenv("TEXT_MAX_CHARS", "14000"))

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

IST_OFFSET = dt.timedelta(hours=5, minutes=30)

client = OpenAI()  # reads OPENAI_API_KEY

# -------------------- CHANNELS --------------------
CHANNELS = [
    {"name": "NEXT EXAM", "channel_id": "UC3-u8ek82h70ArrPO4HJypQ"},
    {"name": "Crazy GkTrick", "channel_id": "UCIl28Ab-H-3LYIPw4hhQexA"},
]

# -------------------- TIME --------------------
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

def post_sources_used(videos: List[Dict[str, Any]]):
    if not videos:
        return
    lines = ["📌 Sources used today (YouTube videos):"]
    for i, v in enumerate(videos, start=1):
        lines.append(f"{i}) {v['channel']} — {v['title']}\n{v['url']}")
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "\n\n".join(lines), "disable_web_page_preview": True})

def post_header(date_str: str, total: int):
    tg("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🧠 Daily Current Affairs Quiz (SSC/PCS) — {date_str}\n\nTotal Questions: {total}\n\nQuiz polls below 👇",
        "disable_web_page_preview": True
    })

def post_closure():
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": "🏁 Today’s Practice Ends Here!\n\nComment your score below 👇\n⏰ Back tomorrow at the same time."})

def post_score_poll(date_str: str, total: int):
    if total <= 10:
        options = [f"{i}/{total}" for i in range(total, max(total - 9, -1), -1)][:10]
    else:
        options = [
            f"{total} ✅", f"{total-1}–{total-2}", f"{total-3}–{total-5}",
            f"{total-6}–{total-8}", f"{total-9}–{total-12}",
            f"{max(total-13,0)}–{max(total-18,0)}",
            f"{max(total-19,0)}–{max(total-25,0)}",
            "Below that 😅"
        ][:10]
    tg("sendPoll", {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": f"📊 Vote your score ({date_str}) ✅\nHow many were correct?",
        "options": options,
        "is_anonymous": True,
        "allows_multiple_answers": False
    })

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

def save_history(hist: dict):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def update_history(hist: dict, date_str: str, used_video_ids: List[str], used_fact_hashes: List[str], used_q_hashes: List[str]) -> dict:
    hist.setdefault("dates", {})
    hist["dates"][date_str] = {
        "video_ids": used_video_ids,
        "fact_hashes": used_fact_hashes,
        "q_hashes": used_q_hashes,
    }

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

# -------------------- YOUTUBE FETCH --------------------
def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text

def yt_rss_url(channel_id: str) -> str:
    return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

def parse_video_id(link: str) -> Optional[str]:
    m = re.search(r"v=([a-zA-Z0-9_-]{8,})", link or "")
    return m.group(1) if m else None

def fetch_recent_videos(channel_id: str, channel_name: str, window_days: int) -> List[Dict[str, Any]]:
    xml_text = http_get(yt_rss_url(channel_id))
    root = ET.fromstring(xml_text)

    ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}

    today = ist_today_date()
    earliest = today - dt.timedelta(days=max(window_days, 1) - 1)
    out = []

    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        link_el = entry.find("atom:link", ns)
        link = link_el.attrib.get("href") if link_el is not None else ""
        published = entry.findtext("atom:published", default="", namespaces=ns)
        vid = entry.findtext("yt:videoId", default="", namespaces=ns) or parse_video_id(link)
        if not vid or not published:
            continue

        pub_dt = parse_rfc3339(published)
        pub_ist = to_ist_date(pub_dt)
        if not (earliest <= pub_ist <= today):
            continue

        out.append({"channel": channel_name, "video_id": vid, "title": title[:200], "url": link, "published": published})

    out.sort(key=lambda x: x["published"], reverse=True)
    return out[:MAX_VIDEOS_PER_CHANNEL]

# -------------------- TRANSCRIPT + DESCRIPTION FALLBACK --------------------
def get_transcript_text(video_id: str) -> Optional[str]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en"])
        text = " ".join([t.get("text", "") for t in transcript]).strip()
        return text[:TRANSCRIPT_MAX_CHARS] if text else None
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return None
    except Exception:
        return None

def get_video_description(video_id: str) -> Optional[str]:
    """
    Fallback when transcript is missing.
    Uses watch page HTML (public) and extracts meta description/og:description.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        html = http_get(url)
        soup = BeautifulSoup(html, "html.parser")

        # Try meta description
        m1 = soup.find("meta", attrs={"name": "description"})
        if m1 and m1.get("content"):
            desc = m1["content"].strip()
            if desc:
                return desc[:TEXT_MAX_CHARS]

        # Try og description
        m2 = soup.find("meta", attrs={"property": "og:description"})
        if m2 and m2.get("content"):
            desc = m2["content"].strip()
            if desc:
                return desc[:TEXT_MAX_CHARS]

        return None
    except Exception:
        return None

# -------------------- LLM SCHEMAS --------------------
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
    "required": ["event_key", "question", "options", "correct_option_id", "correct_answer", "explanation", "source_title", "source_url", "source"],
    "additionalProperties": False,
}

# -------------------- QUALITY --------------------
BAD_OPTIONS = {"all of the above", "none of the above", "all above", "none"}
STATIC_GK_PHRASES = ["capital of", "currency of", "national animal", "largest planet", "highest mountain"]

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

# -------------------- LLM PIPELINE --------------------
def extract_facts(video: Dict[str, Any], text_blob: str) -> List[Dict[str, str]]:
    system = (
        "Extract ONLY current-affairs facts from the provided content.\n"
        "Rules:\n"
        "- Facts must be directly supported by the content.\n"
        "- Avoid static GK and generic statements.\n"
        "- Output 8 to 40 facts.\n"
        "- event_key: 3–8 words describing the topic.\n"
    )
    user = {
        "channel": video["channel"],
        "video_title": video["title"],
        "video_url": video["url"],
        "content": text_blob[:TEXT_MAX_CHARS],
    }
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        text={"format": {"type": "json_schema", "name": "facts_schema", "strict": True, "schema": FACTS_SCHEMA}},
    )
    data = json.loads(resp.output_text)
    return data.get("facts", [])

def build_mcq(video: Dict[str, Any], fact_item: Dict[str, str]) -> Optional[Dict[str, Any]]:
    system = (
        "Create ONE SSC/PCS-standard easy exam-like MCQ from the FACT.\n"
        "Rules:\n"
        "- Use ONLY the fact. Do not add outside info.\n"
        "- Question starts with Which/What/Who/Where/When and ends with '?'.\n"
        "- 4 options, same-category, plausible.\n"
        "- No all/none of the above.\n"
        "- Explanation <= 200 characters.\n"
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
        input=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
        text={"format": {"type": "json_schema", "name": "mcq_schema", "strict": True, "schema": MCQ_SCHEMA}},
    )
    mcq = json.loads(resp.output_text)

    mcq["source"] = f"YouTube: {video['channel']}"
    mcq["source_title"] = video["title"][:200]
    mcq["source_url"] = video["url"][:400]

    return mcq if mcq_ok(mcq) else None

# -------------------- BUILD DAILY --------------------
def build_daily_mcqs_with_window(window_days: int) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[str]]:
    date_str = ist_today_str()
    hist = load_history()

    used_video_ids = set(hist.get("video_ids", [])[-2000:])
    used_fact_hashes = set(hist.get("fact_hashes", [])[-6000:])
    used_q_hashes = set(hist.get("q_hashes", [])[-6000:])

    # videos
    all_videos: List[Dict[str, Any]] = []
    for ch in CHANNELS:
        vids = fetch_recent_videos(ch["channel_id"], ch["name"], window_days)
        print(f"✅ {ch['name']}: {len(vids)} videos found (window={window_days} days)")
        all_videos.extend(vids)

    # dedup + skip used
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
        raise RuntimeError("No NEW videos found in the selected window.")

    videos_used: List[Dict[str, Any]] = []
    day_used_vids: List[str] = []
    day_used_fact_hashes: List[str] = []
    day_used_q_hashes: List[str] = []

    mcqs: List[Dict[str, Any]] = []

    for v in uniq:
        if len(mcqs) >= MAX_POLLS_PER_DAY:
            break

        # transcript -> else description
        text_blob = get_transcript_text(v["video_id"])
        if text_blob:
            print(f"✅ Transcript OK: {v['video_id']} ({v['channel']})")
        else:
            print(f"⚠️ No transcript for video: {v['video_id']} ({v['channel']})")
            text_blob = get_video_description(v["video_id"])
            if text_blob:
                print(f"✅ Using description fallback: {v['video_id']} ({v['channel']})")
            else:
                continue

        facts = extract_facts(v, text_blob)
        if not facts:
            continue

        if v["video_id"] not in day_used_vids:
            day_used_vids.append(v["video_id"])
            used_video_ids.add(v["video_id"])
            videos_used.append(v)

        for f in facts:
            if len(mcqs) >= MAX_POLLS_PER_DAY:
                break

            fact_text = (f.get("fact") or "").strip()
            if len(fact_text) < 12:
                continue

            fh = _h(_norm_text(fact_text))
            if fh in used_fact_hashes:
                continue

            mcq = build_mcq(v, f)
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

    return date_str, mcqs, videos_used, day_used_vids, day_used_fact_hashes, day_used_q_hashes

def build_daily_mcqs() -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[str]]:
    # Try primary window
    date_str, mcqs, videos_used, used_vids, used_facts, used_qs = build_daily_mcqs_with_window(RECENT_WINDOW_DAYS_PRIMARY)

    if len(mcqs) >= MIN_POLLS_PER_DAY:
        return date_str, mcqs, videos_used, used_vids, used_facts, used_qs

    print(f"⚠️ Only {len(mcqs)} MCQs in {RECENT_WINDOW_DAYS_PRIMARY}-day window. Expanding window...")
    date_str2, mcqs2, videos_used2, used_vids2, used_facts2, used_qs2 = build_daily_mcqs_with_window(RECENT_WINDOW_DAYS_FALLBACK)

    if len(mcqs2) < MIN_POLLS_PER_DAY:
        raise RuntimeError(f"Too few MCQs. Got {len(mcqs2)}; need at least {MIN_POLLS_PER_DAY}. (Transcripts/descriptions too thin)")

    return date_str2, mcqs2, videos_used2, used_vids2, used_facts2, used_qs2

# -------------------- POST --------------------
def post_mcqs(date_str: str, mcqs: List[Dict[str, Any]], videos_used: List[Dict[str, Any]]):
    post_sources_used(videos_used)
    post_header(date_str, len(mcqs))

    for i, q in enumerate(mcqs, start=1):
        q = shuffle_options(q)
        tg("sendPoll", {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": f"Q{i}. {q['question']}",
            "options": q["options"],
            "type": "quiz",
            "correct_option_id": q["correct_option_id"],
            "explanation": q["explanation"],
            "is_anonymous": True,
        })
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
