import os
import re
import json
import time
import hashlib
import random
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import requests
from openai import OpenAI

# -------------------- CONFIG --------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # e.g., @UPPCSSUCCESS
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Keep last N days to avoid repeats across days
KEEP_LAST_DAYS = int(os.getenv("KEEP_LAST_DAYS", "15"))

# Telegram pacing (helps avoid 429)
SLEEP_BETWEEN_POLLS = float(os.getenv("SLEEP_BETWEEN_POLLS", "2"))
SLEEP_AFTER_QUIZ = float(os.getenv("SLEEP_AFTER_QUIZ", "2"))

# Timezone: Asia/Kolkata (IST)
IST_OFFSET = dt.timedelta(hours=5, minutes=30)

HISTORY_FILE = "mcq_history.json"
HTTP_TIMEOUT = 25
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

client = OpenAI()  # reads OPENAI_API_KEY from environment


# -------------------- TELEGRAM HELPERS --------------------
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


# -------------------- JSON SCHEMA --------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "mcqs": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "event_key": {"type": "string", "minLength": 3, "maxLength": 80},
                    "question": {"type": "string", "minLength": 1, "maxLength": 320},
                    "options": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                    "correct_option_id": {"type": "integer", "minimum": 0, "maximum": 3},
                    "correct_answer": {"type": "string", "minLength": 1, "maxLength": 120},
                    "explanation": {"type": "string", "maxLength": 220},
                    "source_url": {"type": "string", "minLength": 5, "maxLength": 400},
                    "source_title": {"type": "string", "minLength": 3, "maxLength": 200},
                },
                "required": [
                    "event_key",
                    "question",
                    "options",
                    "correct_option_id",
                    "correct_answer",
                    "explanation",
                    "source_url",
                    "source_title",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["date", "mcqs"],
    "additionalProperties": False,
}


# -------------------- HISTORY (NO-REPEAT ACROSS DAYS) --------------------
def load_history() -> dict:
    """
    {
      "dates": {
        "YYYY-MM-DD": {
          "event_keys":[...],
          "question_norms":[...],
          "url_hashes":[...]
        }
      },
      "event_keys":[...],
      "question_norms":[...],
      "url_hashes":[...]
    }
    """
    if not os.path.exists(HISTORY_FILE):
        return {"dates": {}, "event_keys": [], "question_norms": [], "url_hashes": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        hist.setdefault("dates", {})
        hist.setdefault("event_keys", [])
        hist.setdefault("question_norms", [])
        hist.setdefault("url_hashes", [])
        return hist
    except Exception:
        return {"dates": {}, "event_keys": [], "question_norms": [], "url_hashes": []}


def save_history(hist: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def _url_hash(url: str) -> str:
    return hashlib.sha256((url or "").strip().encode("utf-8")).hexdigest()[:24]


def update_history(hist: dict, mcq_set: dict, keep_last_days: int) -> dict:
    today = mcq_set["date"]
    hist.setdefault("dates", {})

    day_event_keys = [q["event_key"] for q in mcq_set["mcqs"]]
    day_qnorms = [_norm_text(q["question"]) for q in mcq_set["mcqs"]]
    day_uhashes = [_url_hash(q["source_url"]) for q in mcq_set["mcqs"]]

    hist["dates"][today] = {
        "event_keys": day_event_keys,
        "question_norms": day_qnorms,
        "url_hashes": day_uhashes,
    }

    # prune old dates
    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > keep_last_days:
        for d in all_dates[:-keep_last_days]:
            hist["dates"].pop(d, None)

    # rebuild flattened lists (bounded)
    eks, qns, uhs = [], [], []
    for d in sorted(hist["dates"].keys()):
        eks.extend(hist["dates"][d].get("event_keys", []))
        qns.extend(hist["dates"][d].get("question_norms", []))
        uhs.extend(hist["dates"][d].get("url_hashes", []))

    hist["event_keys"] = eks[-400:]
    hist["question_norms"] = qns[-400:]
    hist["url_hashes"] = uhs[-600:]
    return hist


# -------------------- LIGHT HTML PARSING (NO RSS) --------------------
def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def strip_html(html: str) -> str:
    # very basic (works without bs4)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;|&apos;", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_meta_date(html: str) -> Optional[str]:
    """
    Tries to extract ISO-like date from meta tags or common patterns.
    Returns YYYY-MM-DD if found.
    """
    # common meta tags
    candidates = []

    # meta property="article:published_time"
    m = re.search(r'property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']', html, re.I)
    if m:
        candidates.append(m.group(1))

    # meta name="publish-date" / "date" / "parsely-pub-date"
    for name in ["publish-date", "date", "parsely-pub-date", "pubdate", "DC.date.issued"]:
        m = re.search(rf'name=["\']{re.escape(name)}["\']\s+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            candidates.append(m.group(1))

    # time datetime="..."
    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, re.I)
    if m:
        candidates.append(m.group(1))

    for c in candidates:
        iso = normalize_date_to_yyyy_mm_dd(c)
        if iso:
            return iso

    # fallback: try "January 5, 2026" pattern
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})",
        html,
        re.I,
    )
    if m:
        month = m.group(1).lower()
        day = int(m.group(2))
        year = int(m.group(3))
        month_num = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }[month]
        return f"{year:04d}-{month_num:02d}-{day:02d}"

    return None


def normalize_date_to_yyyy_mm_dd(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None

    # ISO with time
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # 05-01-2026 or 05/01/2026 (assume D-M-Y)
    m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


def now_ist_date() -> dt.date:
    # timezone-aware UTC -> IST date
    return (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).date()



def ist_today_str() -> str:
    return now_ist_date().isoformat()


def ist_yesterday_str() -> str:
    return (now_ist_date() - dt.timedelta(days=1)).isoformat()


# -------------------- SOURCES (AUTHENTIC CA SITES) --------------------
def collect_affairscloud(today_iso: str) -> List[Dict[str, str]]:
    """
    AffairsCloud has date-wise pages. We scrape the 'Current Affairs Today' listing
    and pick pages whose title/url contains today's date components.
    """
    base = "https://affairscloud.com/current-affairs-ca/current-affairs-today/"
    html = http_get(base)

    # Find article links (date-wise posts)
    links = re.findall(r'href=["\'](https://affairscloud\.com/current-affairs-[^"\']+)["\']', html, re.I)
    links = list(dict.fromkeys(links))[:40]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page) or "AffairsCloud Current Affairs"
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            items.append({"source": "AffairsCloud", "url": url, "title": title, "date": pub})
        except Exception:
            continue

    # Keep only today
    return [it for it in items if it.get("date") == today_iso]


def collect_gktoday(today_iso: str) -> List[Dict[str, str]]:
    base = "https://www.gktoday.in/current-affairs/"
    html = http_get(base)

    links = re.findall(r'href=["\'](https://www\.gktoday\.in/[^"\']+)["\']', html, re.I)
    # reduce junk
    links = [u for u in links if "/current-affairs/" not in u and "wp-" not in u]
    links = list(dict.fromkeys(links))[:60]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            # GKToday home sometimes shows older; we enforce exact today only
            if pub == today_iso:
                items.append({"source": "GKToday", "url": url, "title": title, "date": pub})
        except Exception:
            continue
    return items


def collect_adda247(today_iso: str) -> List[Dict[str, str]]:
    base = "https://currentaffairs.adda247.com/"
    html = http_get(base)

    # Posts are under currentaffairs.adda247.com/<slug>/
    links = re.findall(r'href=["\'](https://currentaffairs\.adda247\.com/[^"\']+/)["\']', html, re.I)
    links = [u for u in links if "category" not in u and "tag" not in u and "page" not in u]
    links = list(dict.fromkeys(links))[:80]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == today_iso:
                items.append({"source": "Adda247", "url": url, "title": title, "date": pub})
        except Exception:
            continue
    return items


def collect_jagranjosh(today_iso: str) -> List[Dict[str, str]]:
    base = "https://www.jagranjosh.com/current-affairs"
    html = http_get(base)

    # Jagran links look like https://www.jagranjosh.com/general-knowledge/<slug>-<digits>
    links = re.findall(r'href=["\'](https://www\.jagranjosh\.com/[^"\']+)["\']', html, re.I)
    links = [u for u in links if "/general-knowledge/" in u or "/current-affairs/" in u]
    links = list(dict.fromkeys(links))[:80]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == today_iso:
                items.append({"source": "JagranJosh", "url": url, "title": title, "date": pub})
        except Exception:
            continue
    return items


def extract_title(html: str) -> Optional[str]:
    m = re.search(r"(?is)<title>\s*(.*?)\s*</title>", html)
    if not m:
        return None
    t = strip_html(m.group(1))
    # clean common suffixes
    t = re.sub(r"\s*\|\s*.*$", "", t).strip()
    return t[:180] if t else None


def guess_date_from_url(url: str) -> Optional[str]:
    """
    If URL contains /5-january-2026/ or /2026/01/05/ etc.
    """
    u = url.lower()

    # /2026/01/05/
    m = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # /5-january-2026/
    m = re.search(
        r"/(\d{1,2})-(january|february|march|april|may|june|july|august|september|october|november|december)-(\d{4})/?",
        u,
    )
    if m:
        day = int(m.group(1))
        mon = m.group(2)
        year = int(m.group(3))
        mon_num = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
        }[mon]
        return f"{year:04d}-{mon_num:02d}-{day:02d}"

    return None


def fetch_article_snippet(url: str, max_chars: int = 900) -> str:
    html = http_get(url)
    text = strip_html(html)
    # keep it short for token control
    return text[:max_chars]


def collect_fresh_items() -> Tuple[str, List[Dict[str, str]]]:
    """
    Collects same-day items from all sources.
    If not enough, allows yesterday as fallback (still recent).
    """
    today = ist_today_str()
    yesterday = ist_yesterday_str()

    sources = [collect_affairscloud, collect_gktoday, collect_adda247, collect_jagranjosh]

    all_today = []
    for fn in sources:
        try:
            got = fn(today)
            print(f"✅ {fn.__name__}: {len(got)} items for {today}")
            all_today.extend(got)
        except Exception as e:
            print(f"⚠️ {fn.__name__} failed: {e}")

    # Deduplicate by URL
    seen = set()
    dedup_today = []
    for it in all_today:
        h = _url_hash(it["url"])
        if h not in seen:
            seen.add(h)
            dedup_today.append(it)

    if len(dedup_today) >= 14:
        return today, dedup_today

    # fallback: add yesterday items (still not "old 2025 stuff")
    all_yday = []
    for fn in sources:
        try:
            got = fn(yesterday)
            all_yday.extend(got)
        except Exception:
            continue

    for it in all_yday:
        h = _url_hash(it["url"])
        if h not in seen:
            seen.add(h)
            it = dict(it)
            it["date"] = yesterday
            dedup_today.append(it)

    # date label still "today" for posting; but content may include yesterday if needed
    return today, dedup_today


# -------------------- MCQ GENERATION --------------------
def _schema_for_n(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "date": {"type": "string"},
            "mcqs": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": SCHEMA["properties"]["mcqs"]["items"],
            },
        },
        "required": ["date", "mcqs"],
        "additionalProperties": False,
    }


def generate_mcqs() -> dict:
    """
    1) Collect fresh items from authentic CA sites (same-day; yday fallback only if needed).
    2) Remove anything repeated via history (URL hash + title norms).
    3) Ask model to select most exam-relevant 10 and write UPSC-level MCQs.
    """
    today_label, fresh_items = collect_fresh_items()
    if len(fresh_items) < 10:
        raise RuntimeError(
            f"Not enough fresh items from authentic CA sources for today. Found {len(fresh_items)}."
        )

    hist = load_history()
    used_url_hashes = set(hist.get("url_hashes", [])[-600:])
    used_event_keys = set(_norm_text(x) for x in (hist.get("event_keys", [])[-400:]))
    used_qnorms = set(hist.get("question_norms", [])[-400:])

    # Filter out previously used URLs and very similar titles
    filtered = []
    seen_title = set()
    for it in fresh_items:
        uh = _url_hash(it["url"])
        if uh in used_url_hashes:
            continue
        tn = _norm_text(it["title"])
        if tn in seen_title:
            continue
        seen_title.add(tn)
        filtered.append(it)

    if len(filtered) < 12:
        # If history is too strict, relax URL only (still prevents repeats strongly)
        filtered = [it for it in fresh_items if _url_hash(it["url"]) not in used_url_hashes]

    # Build an evidence pack for the model (title+url+short snippet)
    # Keep to a manageable size (top 22 items)
    pack = []
    for it in filtered[:22]:
        try:
            snippet = fetch_article_snippet(it["url"], max_chars=800)
        except Exception:
            snippet = ""
        pack.append(
            {
                "source": it["source"],
                "date": it["date"],
                "title": it["title"],
                "url": it["url"],
                "snippet": snippet,
            }
        )

    system = (
        "You are an expert UPSC/State PCS question setter.\n"
        "You MUST create EXACTLY 10 UPSC-level current affairs MCQs.\n"
        "Use ONLY the provided evidence pack (title/url/snippet). Do NOT introduce outside facts.\n"
        "TARGET EXAMS: UPSC, State PCS, SSC (CGL), Railways, and general govt exams.\n\n"
        "STRICT CONSTRAINTS:\n"
        "- SAME-DAY FOCUS: Prefer items dated today; if pack includes yesterday, use it only if needed.\n"
        "- NO BANKING/FINANCE QUESTIONS: Avoid RBI circular trivia, bank appointments, IBPS-type banking CA.\n"
        "  (Economy is allowed: GDP, inflation, fiscal policy, govt schemes, indices, major reforms, trade, etc.)\n"
        "- Framing must look like real exam MCQs: 'With reference to...', 'Consider the following statements...',\n"
        "  'Which of the following is/are correct?', 'Match the following', etc.\n"
        "- Each MCQ must be from a DIFFERENT event/topic (no duplicates).\n"
        "- event_key must be unique (3–10 words), should identify the event.\n"
        "- correct_answer must EXACTLY match one option.\n"
        "- correct_option_id must match correct_answer index.\n"
        "- Options must be plausible and distinct.\n"
        "- Explanation <= 220 characters and should justify WHY the correct option is correct.\n"
        "- Include source_url and source_title from the evidence pack for each MCQ.\n"
    )

    avoid_block = {
        "used_event_keys": sorted(list(used_event_keys))[:120],
        "used_question_norms": sorted(list(used_qnorms))[:120],
    }

    user = {
        "posting_date_label": today_label,
        "evidence_pack": pack,
        "avoid": avoid_block,
        "output_rules": "Return JSON strictly as per schema.",
    }

    schema = _schema_for_n(10)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "daily_upsc_level_ca_mcqs",
                "strict": True,
                "schema": schema,
            }
        },
    )

    mcq_set = json.loads(resp.output_text)
    mcq_set["date"] = today_label  # ensure label is today for your channel
    return mcq_set


# -------------------- QUALITY GUARDS --------------------
def enforce_no_bank_terms(q: Dict[str, Any]) -> bool:
    """
    Hard guard to reduce accidental banking CA.
    """
    bad = [
        "bank", "ibps", "sbi", "rrb", "nbfc", "repo rate", "mclr",
        "basel", "npa", "crr", "slr", "rtgs", "neft", "upi limits",
        "scheduled commercial bank",
    ]
    text = (q.get("question", "") + " " + " ".join(q.get("options", []))).lower()
    return not any(b in text for b in bad)


def normalize_and_dedupe_mcqs(mcq_set: dict) -> dict:
    seen_q = set()
    seen_ek = set()
    out = []
    for q in mcq_set["mcqs"]:
        qn = _norm_text(q["question"])
        ek = _norm_text(q["event_key"])
        if qn in seen_q or ek in seen_ek:
            continue
        if not enforce_no_bank_terms(q):
            continue
        # ensure answer mapping is correct
        if q["correct_answer"] not in q["options"]:
            continue
        if q["options"].index(q["correct_answer"]) != q["correct_option_id"]:
            q["correct_option_id"] = q["options"].index(q["correct_answer"])

        # unique options
        if len(set(_norm_text(x) for x in q["options"])) != 4:
            continue

        seen_q.add(qn)
        seen_ek.add(ek)
        out.append(q)

    # If less than 10 after guards, keep best effort (but try one regeneration)
    if len(out) < 10:
        mcq_set["mcqs"] = out
        return mcq_set

    mcq_set["mcqs"] = out[:10]
    return mcq_set


# -------------------- QUIZ POSTING --------------------
def shuffle_options_and_fix_answer(q: dict) -> dict:
    options = q["options"]
    correct_text = options[q["correct_option_id"]]

    seed = q["question"] + correct_text
    rnd = random.Random(seed)
    shuffled = options[:]
    rnd.shuffle(shuffled)

    q["options"] = shuffled
    q["correct_option_id"] = shuffled.index(correct_text)
    q["correct_answer"] = correct_text
    return q


def post_competitive_closure_message():
    text = (
        "🏁 Today’s Challenge Ends Here!\n\n"
        "Comment your score below 👇\n"
        "Let’s see how many 8+ scorers we have today 🔥\n\n"
        "⏰ Back tomorrow at the same time."
    )
    tg(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
    )


def post_score_poll(date_str: str):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": f"📊 Vote your score ({date_str}) ✅\nHow many were correct out of 10?",
        "options": [
            "10/10 🏆",
            "9/10 🔥",
            "8/10 ✅",
            "7/10 👍",
            "6/10 🙂",
            "5/10 📘",
            "4/10 🧠",
            "3 or less 😅",
        ],
        "is_anonymous": True,
        "allows_multiple_answers": False,
    }
    tg("sendPoll", payload)


def post_to_channel(mcq_set: dict):
    tg(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 UPSC-level MCQ polls below 👇",
        },
    )

    for i, q in enumerate(mcq_set["mcqs"], start=1):
        q = shuffle_options_and_fix_answer(q)

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": f"Q{i}. {q['question']}",
            "options": q["options"],
            "type": "quiz",
            "correct_option_id": q["correct_option_id"],
            "explanation": q["explanation"],
            "is_anonymous": True,
        }
        tg("sendPoll", payload)
        time.sleep(SLEEP_BETWEEN_POLLS)

    time.sleep(SLEEP_AFTER_QUIZ)

    try:
        post_competitive_closure_message()
    except Exception as e:
        print("❌ Failed to post closure message:", e)

    time.sleep(2)

    try:
        post_score_poll(mcq_set["date"])
    except Exception as e:
        print("❌ Failed to post score poll:", e)


# -------------------- MAIN --------------------
def main():
    mcq_set = generate_mcqs()
    mcq_set = normalize_and_dedupe_mcqs(mcq_set)

    # If guards reduced count, try one more regeneration to fill
    if len(mcq_set["mcqs"]) < 10:
        print(f"⚠️ After quality guards only {len(mcq_set['mcqs'])} MCQs. Regenerating once...")
        mcq_set2 = generate_mcqs()
        mcq_set2 = normalize_and_dedupe_mcqs(mcq_set2)
        merged = (mcq_set["mcqs"] + mcq_set2["mcqs"])
        # final unique merge
        temp = {"date": mcq_set["date"], "mcqs": merged}
        temp = normalize_and_dedupe_mcqs(temp)
        mcq_set["mcqs"] = temp["mcqs"][:10]

    if len(mcq_set["mcqs"]) < 10:
        raise RuntimeError(f"Could not reach 10 high-quality UPSC-level MCQs. Got {len(mcq_set['mcqs'])}.")

    post_to_channel(mcq_set)

    # Save history AFTER success
    hist = load_history()
    hist = update_history(hist, mcq_set, keep_last_days=KEEP_LAST_DAYS)
    save_history(hist)

    print("✅ Posted 10 quiz polls + message + score poll successfully.")


if __name__ == "__main__":
    main()
