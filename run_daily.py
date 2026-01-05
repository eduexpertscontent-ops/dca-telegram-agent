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

KEEP_LAST_DAYS = int(os.getenv("KEEP_LAST_DAYS", "15"))

SLEEP_BETWEEN_POLLS = float(os.getenv("SLEEP_BETWEEN_POLLS", "2"))
SLEEP_AFTER_QUIZ = float(os.getenv("SLEEP_AFTER_QUIZ", "2"))

HISTORY_FILE = "mcq_history.json"
HTTP_TIMEOUT = 25
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
client = OpenAI()  # reads OPENAI_API_KEY from environment


# -------------------- TIME HELPERS --------------------
def now_ist_date() -> dt.date:
    # timezone-aware UTC -> IST date (fixes utcnow warning)
    return (dt.datetime.now(dt.timezone.utc) + IST_OFFSET).date()


def ist_today_str() -> str:
    return now_ist_date().isoformat()


def ist_yesterday_str() -> str:
    return (now_ist_date() - dt.timedelta(days=1)).isoformat()


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
        "YYYY-MM-DD": {"event_keys":[...], "question_norms":[...], "url_hashes":[...]}
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

    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > keep_last_days:
        for d in all_dates[:-keep_last_days]:
            hist["dates"].pop(d, None)

    eks, qns, uhs = [], [], []
    for d in sorted(hist["dates"].keys()):
        eks.extend(hist["dates"][d].get("event_keys", []))
        qns.extend(hist["dates"][d].get("question_norms", []))
        uhs.extend(hist["dates"][d].get("url_hashes", []))

    hist["event_keys"] = eks[-500:]
    hist["question_norms"] = qns[-500:]
    hist["url_hashes"] = uhs[-800:]
    return hist


# -------------------- HTTP + LIGHT HTML PARSING (NO RSS) --------------------
def http_get(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def strip_html(html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;|&apos;", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_title(html: str) -> Optional[str]:
    m = re.search(r"(?is)<title>\s*(.*?)\s*</title>", html)
    if not m:
        return None
    t = strip_html(m.group(1))
    t = re.sub(r"\s*\|\s*.*$", "", t).strip()
    return t[:180] if t else None


def normalize_date_to_yyyy_mm_dd(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None

    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"

    return None


def extract_meta_date(html: str) -> Optional[str]:
    candidates = []

    m = re.search(r'property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']', html, re.I)
    if m:
        candidates.append(m.group(1))

    for name in ["publish-date", "date", "parsely-pub-date", "pubdate", "DC.date.issued"]:
        m = re.search(rf'name=["\']{re.escape(name)}["\']\s+content=["\']([^"\']+)["\']', html, re.I)
        if m:
            candidates.append(m.group(1))

    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, re.I)
    if m:
        candidates.append(m.group(1))

    for c in candidates:
        iso = normalize_date_to_yyyy_mm_dd(c)
        if iso:
            return iso

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


def guess_date_from_url(url: str) -> Optional[str]:
    u = (url or "").lower()

    m = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

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
    return text[:max_chars]


# -------------------- SOURCES (YOUR ALLOWED LIST) --------------------
def collect_affairscloud(date_iso: str) -> List[Dict[str, str]]:
    base = "https://affairscloud.com/current-affairs-ca/current-affairs-today/"
    html = http_get(base)

    links = re.findall(r'href=["\'](https://affairscloud\.com/current-affairs-[^"\']+)["\']', html, re.I)
    links = list(dict.fromkeys(links))[:50]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page) or "AffairsCloud Current Affairs"
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == date_iso:
                items.append({"source": "AffairsCloud", "url": url, "title": title, "date": pub})
        except Exception:
            continue

    return items


def collect_gktoday(date_iso: str) -> List[Dict[str, str]]:
    base = "https://www.gktoday.in/current-affairs/"
    html = http_get(base)

    links = re.findall(r'href=["\'](https://www\.gktoday\.in/[^"\']+)["\']', html, re.I)
    links = [u for u in links if "wp-content" not in u]
    links = list(dict.fromkeys(links))[:80]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == date_iso:
                items.append({"source": "GKToday", "url": url, "title": title, "date": pub})
        except Exception:
            continue

    return items


def collect_adda247(date_iso: str) -> List[Dict[str, str]]:
    base = "https://currentaffairs.adda247.com/"
    html = http_get(base)

    links = re.findall(r'href=["\'](https://currentaffairs\.adda247\.com/[^"\']+/)["\']', html, re.I)
    links = [u for u in links if "category" not in u and "tag" not in u and "/page/" not in u]
    links = list(dict.fromkeys(links))[:120]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == date_iso:
                items.append({"source": "Adda247", "url": url, "title": title, "date": pub})
        except Exception:
            continue

    return items


def collect_jagranjosh(date_iso: str) -> List[Dict[str, str]]:
    base = "https://www.jagranjosh.com/current-affairs"
    html = http_get(base)

    links = re.findall(r'href=["\'](https://www\.jagranjosh\.com/[^"\']+)["\']', html, re.I)
    links = [u for u in links if "/general-knowledge/" in u or "/current-affairs/" in u]
    links = list(dict.fromkeys(links))[:120]

    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == date_iso:
                items.append({"source": "JagranJosh", "url": url, "title": title, "date": pub})
        except Exception:
            continue

    return items


def collect_fresh_items() -> Tuple[str, List[Dict[str, str]]]:
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
    dedup = []
    for it in all_today:
        h = _url_hash(it["url"])
        if h not in seen:
            seen.add(h)
            dedup.append(it)

    # If not enough today, add yesterday (still recent, avoids 2025 old junk)
    if len(dedup) < 14:
        all_yday = []
        for fn in sources:
            try:
                all_yday.extend(fn(yesterday))
            except Exception:
                continue

        for it in all_yday:
            h = _url_hash(it["url"])
            if h not in seen:
                seen.add(h)
                it = dict(it)
                it["date"] = yesterday
                dedup.append(it)

    return today, dedup


# -------------------- BAN BANKING/FINANCE (ALLOW ECONOMY) --------------------
# Banking/Finance terms to exclude (including payments/rates/markets)
BANK_FIN_BLOCK = [
    "rbi", "repo", "reverse repo", "crr", "slr", "mclr", "mpc",
    "sebi", "irdai", "pfrda", "nbfc",
    "bank ", "banks", "banking", "loan", "credit", "debit",
    "upi", "rtgs", "neft", "imps", "payments", "payment",
    "mutual fund", "ipo", "share market", "stock", "stocks", "bond", "g-sec", "treasury bill", "forex",
]

# Economy allowed themes (macro/policy) – these are OK even if “economy”
ECON_ALLOW = [
    "gdp", "inflation", "cpi", "wpi", "fiscal", "budget", "tax", "gst",
    "subsidy", "trade", "export", "import", "current account", "cad",
    "poverty", "employment", "unemployment", "agriculture", "msp", "food security",
]

def is_banking_finance_related(title: str, snippet: str) -> bool:
    txt = f"{title} {snippet}".lower()
    hit_bank = any(k in txt for k in BANK_FIN_BLOCK)
    hit_econ = any(k in txt for k in ECON_ALLOW)
    # If it looks banking/finance and NOT clearly macro economy, exclude
    return hit_bank and not hit_econ


# -------------------- GENERATION: GK/EXAM-STYLE DIRECT MCQs --------------------
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
    today_label, items = collect_fresh_items()
    if len(items) < 10:
        raise RuntimeError(f"Not enough fresh items from CA sources. Found {len(items)}.")

    hist = load_history()
    used_url_hashes = set(hist.get("url_hashes", [])[-800:])
    used_event_keys = set(_norm_text(x) for x in (hist.get("event_keys", [])[-500:]))
    used_qnorms = set(hist.get("question_norms", [])[-500:])

    # Build evidence pack (filter repeats + banking/finance)
    pack = []
    seen_title = set()

    # Try more candidates because some will be filtered out
    for it in items[:80]:
        uh = _url_hash(it["url"])
        if uh in used_url_hashes:
            continue

        tn = _norm_text(it["title"])
        if tn in seen_title:
            continue
        seen_title.add(tn)

        try:
            snippet = fetch_article_snippet(it["url"], max_chars=900)
        except Exception:
            snippet = ""

        # Exclude banking/finance (economy allowed)
        if is_banking_finance_related(it["title"], snippet):
            continue

        pack.append(
            {
                "source": it["source"],
                "date": it["date"],
                "title": it["title"],
                "url": it["url"],
                "snippet": snippet,
            }
        )

        if len(pack) >= 28:
            break

    if len(pack) < 14:
        raise RuntimeError(
            f"Not enough usable items after filters (banking/finance removed). Found {len(pack)}."
        )

    # System prompt tuned to match YOUR SAMPLE (direct factual daily quiz)
    system = (
        "You are a daily Current Affairs MCQ setter for SSC/State PCS/Railways/General Govt exams.\n"
        "You must write questions in the SAME style as popular daily quiz formats:\n"
        "- Direct factual questions (Who/What/Which/Where/When).\n"
        "- NOT statement-based UPSC format.\n"
        "- One event = one question.\n"
        "\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided evidence pack (title/url/snippet). Do NOT add outside facts.\n"
        "2) Freshness: Prefer items dated today; if yesterday items exist, use only if needed.\n"
        "3) HARD BAN: Do NOT create Banking/Finance questions (RBI/SEBI/IRDAI/PFRDA, repo/CRR/SLR/MCLR,\n"
        "   UPI/RTGS/NEFT, banks/NBFCs, stock market/IPO/mutual funds).\n"
        "   Economy is allowed ONLY at macro/policy level (GDP, inflation, budget, GST, trade, poverty, employment, agriculture).\n"
        "4) Questions must look like real exam MCQs: concise, clear, no casual tone.\n"
        "5) Options must be plausible and same-category (states vs states, ministries vs ministries, organisations vs organisations).\n"
        "6) correct_answer must EXACTLY match one option; correct_option_id must match index.\n"
        "7) explanation <= 220 chars, short and exam-friendly.\n"
        "8) Ensure 10 questions are from 10 different topics (no duplicates).\n"
    )

    user = {
        "posting_date_label": today_label,
        "evidence_pack": pack,
        "avoid": {
            "used_event_keys": sorted(list(used_event_keys))[:200],
            "used_question_norms": sorted(list(used_qnorms))[:200],
        },
        "output_rules": "Return JSON strictly as per schema.",
        "note": "Do not force fixed section-wise quotas; pick most exam-relevant items available."
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
                "name": "daily_exam_style_ca_mcqs",
                "strict": True,
                "schema": schema,
            }
        },
    )

    mcq_set = json.loads(resp.output_text)
    mcq_set["date"] = today_label
    return mcq_set


# -------------------- QUALITY GUARDS (MATCH YOUR SAMPLE STYLE) --------------------
def is_exam_style_question(q: Dict[str, Any]) -> bool:
    txt = (q.get("question") or "").strip().lower()
    # Must be direct style; reject statement-based UPSC format
    bad_starts = ("consider the following statements", "with reference to", "which of the statements")
    if txt.startswith(bad_starts):
        return False

    good_starts = (
        "which", "what", "who", "where", "when", "name the", "the name of", "recently",
        "skilling",  # allows a few titles converted into direct
    )
    if not txt.startswith(good_starts):
        # still allow if it ends with '?', and short
        if not txt.endswith("?"):
            return False

    if len(txt) > 320:
        return False
    return True


def enforce_no_banking_finance(q: Dict[str, Any]) -> bool:
    text = (q.get("question", "") + " " + " ".join(q.get("options", [])) + " " + q.get("explanation","")).lower()
    # if any blocked term appears, reject (even if model slips)
    return not any(k in text for k in BANK_FIN_BLOCK)


def normalize_and_dedupe_mcqs(mcq_set: dict) -> dict:
    seen_q = set()
    seen_ek = set()
    seen_url = set()

    out = []
    for q in mcq_set.get("mcqs", []):
        # basic checks
        if not is_exam_style_question(q):
            continue
        if not enforce_no_banking_finance(q):
            continue

        # mapping checks
        opts = q.get("options") or []
        if len(opts) != 4:
            continue
        if len(set(_norm_text(o) for o in opts)) != 4:
            continue

        ca = q.get("correct_answer", "")
        if ca not in opts:
            continue
        q["correct_option_id"] = opts.index(ca)

        qn = _norm_text(q.get("question", ""))
        ek = _norm_text(q.get("event_key", ""))
        uh = _url_hash(q.get("source_url", ""))

        if not ek or ek in seen_ek:
            continue
        if not qn or qn in seen_q:
            continue
        if uh in seen_url:
            continue

        seen_q.add(qn)
        seen_ek.add(ek)
        seen_url.add(uh)

        # trim
        q["explanation"] = (q.get("explanation") or "")[:220]
        q["source_title"] = (q.get("source_title") or "")[:200]
        q["source_url"] = (q.get("source_url") or "")[:400]

        out.append(q)

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
            "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇",
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

    # If less than 10 after guards, regenerate once and merge
    if len(mcq_set["mcqs"]) < 10:
        print(f"⚠️ Only {len(mcq_set['mcqs'])} MCQs after quality guards. Regenerating once...")
        mcq_set2 = generate_mcqs()
        mcq_set2 = normalize_and_dedupe_mcqs(mcq_set2)

        merged = {"date": mcq_set["date"], "mcqs": mcq_set["mcqs"] + mcq_set2["mcqs"]}
        merged = normalize_and_dedupe_mcqs(merged)
        mcq_set["mcqs"] = merged["mcqs"][:10]

    if len(mcq_set["mcqs"]) < 10:
        raise RuntimeError(f"Could not reach 10 good exam-style MCQs. Got {len(mcq_set['mcqs'])}.")

    post_to_channel(mcq_set)

    hist = load_history()
    hist = update_history(hist, mcq_set, keep_last_days=KEEP_LAST_DAYS)
    save_history(hist)

    print("✅ Posted 10 quiz polls + message + score poll successfully.")


if __name__ == "__main__":
    main()
