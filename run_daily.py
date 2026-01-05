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

# ==================== CONFIG ====================
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # e.g., @UPPCSSUCCESS
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# No repetition window
KEEP_LAST_DAYS = int(os.getenv("KEEP_LAST_DAYS", "30"))

# Posting volume
MAX_POLLS_PER_DAY = int(os.getenv("MAX_POLLS_PER_DAY", "25"))  # can go beyond 10
MIN_POLLS_TO_POST = int(os.getenv("MIN_POLLS_TO_POST", "10"))  # if less, still post what we have
MAX_ARTICLES_PER_SOURCE = int(os.getenv("MAX_ARTICLES_PER_SOURCE", "40"))

SLEEP_BETWEEN_POLLS = float(os.getenv("SLEEP_BETWEEN_POLLS", "2"))
SLEEP_AFTER_QUIZ = float(os.getenv("SLEEP_AFTER_QUIZ", "2"))

HISTORY_FILE = os.getenv("HISTORY_FILE", "mcq_history.json")
HTTP_TIMEOUT = 25

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

IST_OFFSET = dt.timedelta(hours=5, minutes=30)
client = OpenAI()  # reads OPENAI_API_KEY from environment


# ==================== TIME HELPERS ====================
def now_ist() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + IST_OFFSET

def ist_today_date() -> dt.date:
    return now_ist().date()

def ist_today_str() -> str:
    return ist_today_date().isoformat()

def ist_yesterday_date() -> dt.date:
    return ist_today_date() - dt.timedelta(days=1)

def ist_yesterday_str() -> str:
    return ist_yesterday_date().isoformat()

def month_name_lower(m: int) -> str:
    names = [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december"
    ]
    return names[m-1]


# ==================== TELEGRAM HELPERS ====================
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

def post_info(text: str):
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True})

def post_header(date_str: str, total: int):
    tg("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🧠 Daily Current Affairs Quiz (SSC/PCS) — {date_str}\n\nTotal Questions: {total}\n\nQuiz polls below 👇",
        "disable_web_page_preview": True
    })

def post_closure():
    tg("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "🏁 Today’s Practice Ends Here!\n\nComment your score below 👇\n⏰ Back tomorrow at the same time.",
        "disable_web_page_preview": True
    })

def post_score_poll(date_str: str, total: int):
    # Telegram allows max 10 options in poll
    if total <= 10:
        opts = [f"{i}/{total}" for i in range(total, max(total-9, -1), -1)][:10]
    else:
        opts = [
            f"{total} ✅",
            f"{total-1}–{total-2}",
            f"{total-3}–{total-5}",
            f"{total-6}–{total-8}",
            f"{total-9}–{total-12}",
            f"{max(total-13,0)}–{max(total-18,0)}",
            f"{max(total-19,0)}–{max(total-25,0)}",
            "Below that 😅",
        ][:10]

    tg("sendPoll", {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": f"📊 Vote your score ({date_str}) ✅\nHow many were correct?",
        "options": opts,
        "is_anonymous": True,
        "allows_multiple_answers": False,
    })


# ==================== HISTORY (NO REPEAT) ====================
def load_history() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {"dates": {}, "url_hashes": [], "fact_hashes": [], "q_hashes": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        hist.setdefault("dates", {})
        hist.setdefault("url_hashes", [])
        hist.setdefault("fact_hashes", [])
        hist.setdefault("q_hashes", [])
        return hist
    except Exception:
        return {"dates": {}, "url_hashes": [], "fact_hashes": [], "q_hashes": []}

def save_history(hist: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)

def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()

def _h(s: str) -> str:
    return hashlib.sha256((s or "").strip().encode("utf-8")).hexdigest()[:24]

def update_history(hist: dict, date_str: str, used_urls: List[str], used_fact_hashes: List[str], used_q_hashes: List[str]) -> dict:
    hist.setdefault("dates", {})
    hist["dates"][date_str] = {
        "url_hashes": [_h(u) for u in used_urls],
        "fact_hashes": used_fact_hashes,
        "q_hashes": used_q_hashes,
    }

    # trim old dates
    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > KEEP_LAST_DAYS:
        for d in all_dates[:-KEEP_LAST_DAYS]:
            hist["dates"].pop(d, None)

    # rebuild flattened caches
    uhs, fhs, qhs = [], [], []
    for d in sorted(hist["dates"].keys()):
        uhs.extend(hist["dates"][d].get("url_hashes", []))
        fhs.extend(hist["dates"][d].get("fact_hashes", []))
        qhs.extend(hist["dates"][d].get("q_hashes", []))

    hist["url_hashes"] = uhs[-6000:]
    hist["fact_hashes"] = fhs[-20000:]
    hist["q_hashes"] = qhs[-20000:]
    return hist


# ==================== HTTP + TEXT CLEAN ====================
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
    return None

def extract_meta_date(html: str) -> Optional[str]:
    candidates = []
    m = re.search(r'property=["\']article:published_time["\']\s+content=["\']([^"\']+)["\']', html, re.I)
    if m:
        candidates.append(m.group(1))
    m = re.search(r'<time[^>]+datetime=["\']([^"\']+)["\']', html, re.I)
    if m:
        candidates.append(m.group(1))
    for c in candidates:
        iso = normalize_date_to_yyyy_mm_dd(c)
        if iso:
            return iso
    return None

def guess_date_from_url(url: str) -> Optional[str]:
    u = (url or "").lower()
    m = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"/(20\d{2})-(\d{2})-(\d{2})/", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"(\d{1,2})-(\w+)-(20\d{2})", u)
    if m:
        # not reliable; ignore
        return None
    m = re.search(r"(\d{2})-(\d{2})-(20\d{2})", u)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


# ==================== SOURCES ====================
def pw_url_for_date(d: dt.date) -> str:
    # example: https://www.pw.live/ssc/exams/daily-current-affairs-today-5-january-2026
    return f"https://www.pw.live/ssc/exams/daily-current-affairs-today-{d.day}-{month_name_lower(d.month)}-{d.year}"

def collect_pw(date_iso: str) -> List[Dict[str, str]]:
    d = dt.date.fromisoformat(date_iso)
    url = pw_url_for_date(d)
    try:
        html = http_get(url)
    except Exception:
        return []
    title = extract_title(html) or f"PW Daily Current Affairs {date_iso}"
    text = strip_html(html)

    # keep a focused chunk to reduce noise
    low = text.lower()
    anchors = [
        "most important current affairs",
        "daily current affairs",
        "today current affairs",
        "current affairs today"
    ]
    for a in anchors:
        idx = low.find(a)
        if idx != -1:
            text = text[idx: idx + 9000]
            break
    else:
        text = text[:9000]

    return [{
        "source": "PW",
        "url": url,
        "title": title,
        "date": date_iso,
        "text": text
    }]

def collect_drishti(date_iso: str) -> List[Dict[str, str]]:
    base = "https://www.drishtiias.com/current-affairs-news-analysis-editorials"
    try:
        html = http_get(base)
    except Exception:
        return []

    # typical links include /current-affairs-news-analysis-editorials/...
    links = re.findall(
        r'href=["\'](https://www\.drishtiias\.com/current-affairs-news-analysis-editorials/[^"\']+)["\']',
        html,
        re.I
    )
    links = [u for u in links if "amp" not in u]
    links = list(dict.fromkeys(links))[:MAX_ARTICLES_PER_SOURCE]

    out = []
    for url in links:
        try:
            page = http_get(url)
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            # strict: only today
            if pub and pub != date_iso:
                continue
            # if pub missing, still accept (Drishti sometimes hides it)
            title = extract_title(page) or "DrishtiIAS Current Affairs"
            text = strip_html(page)[:9000]
            out.append({"source": "DrishtiIAS", "url": url, "title": title, "date": date_iso, "text": text})
        except Exception:
            continue

    # IMPORTANT: to avoid old content from Drishti list pages, we keep only those
    # whose text includes the target date string at least once (weak check)
    # If your Drishti pages don't show date, this filter can over-reject.
    # We'll keep it soft: only apply if we already have many.
    if len(out) >= 6:
        out2 = []
        for a in out:
            if date_iso in a["text"]:
                out2.append(a)
        if len(out2) >= 2:
            out = out2

    return out[:MAX_ARTICLES_PER_SOURCE]

def collect_currentaffairsworld(date_iso: str) -> List[Dict[str, str]]:
    base = "https://currentaffairsworld.in/category/current-affairs/"
    try:
        html = http_get(base)
    except Exception:
        return []

    links = re.findall(r'href=["\'](https://currentaffairsworld\.in/[^"\']+)["\']', html, re.I)
    links = [u for u in links if "wp-content" not in u]
    links = list(dict.fromkeys(links))[:MAX_ARTICLES_PER_SOURCE]

    out = []
    for url in links:
        try:
            page = http_get(url)
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub and pub != date_iso:
                continue
            title = extract_title(page) or "CurrentAffairsWorld"
            text = strip_html(page)[:9000]
            out.append({"source": "CurrentAffairsWorld", "url": url, "title": title, "date": date_iso, "text": text})
        except Exception:
            continue
    return out[:MAX_ARTICLES_PER_SOURCE]

def collect_fresh_articles() -> Tuple[str, List[Dict[str, str]]]:
    today = ist_today_str()
    yday = ist_yesterday_str()

    # today first
    today_articles = []
    today_articles.extend(collect_pw(today))
    today_articles.extend(collect_drishti(today))
    today_articles.extend(collect_currentaffairsworld(today))

    print(f"✅ PW: {len([a for a in today_articles if a['source']=='PW'])} items for {today}")
    print(f"✅ DrishtiIAS: {len([a for a in today_articles if a['source']=='DrishtiIAS'])} items for {today}")
    print(f"✅ CurrentAffairsWorld: {len([a for a in today_articles if a['source']=='CurrentAffairsWorld'])} items for {today}")

    # fallback yesterday ONLY if today is empty/useless
    if len(today_articles) < 1:
        yday_articles = []
        yday_articles.extend(collect_pw(yday))
        yday_articles.extend(collect_drishti(yday))
        yday_articles.extend(collect_currentaffairsworld(yday))
        if yday_articles:
            return yday, yday_articles

    return today, today_articles


# ==================== QUALITY FILTERS ====================
BAD_OPTIONS = {"all of the above", "none of the above", "all above", "none"}

STATIC_GK_PHRASES = [
    "capital of", "currency of", "largest planet", "highest mountain",
    "national animal", "national anthem", "trick", "mnemonic",
    "one-liner", "static gk"
]

META_PHRASES = [
    "youtube", "video", "channel", "subscribe", "like", "share", "comment",
    "shorts", "watch", "playlist", "hosted on", "follow us", "telegram"
]

# Keep it SSC/PCS friendly (avoid deep banking/market terms)
BANKING_BLOCK = [
    "rbi", "sebi", "irdai", "pfrda", "repo", "reverse repo", "crr", "slr",
    "upi", "neft", "rtgs", "imps", "ipo", "stock", "share market",
    "mutual fund", "cryptocurrency", "bitcoin"
]
ECON_ALLOW = ["gdp", "cpi", "inflation", "budget", "gst", "fiscal", "export", "import", "trade"]

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
    if any(p in low for p in META_PHRASES):
        return False
    return True

def banking_blocked(text: str) -> bool:
    low = (text or "").lower()
    if any(k in low for k in BANKING_BLOCK):
        # allow macro words if NOT the finance regulator/payment infra words
        if any(x in low for x in ECON_ALLOW) and not any(y in low for y in ["rbi","sebi","upi","neft","rtgs","repo","crr","slr"]):
            return False
        return True
    return False


# ==================== LLM: FACTS + MCQs ====================
FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "minItems": 10,
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": {
                    "event_key": {"type": "string", "minLength": 3, "maxLength": 80},
                    "fact": {"type": "string", "minLength": 12, "maxLength": 260},
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

def extract_facts(article: Dict[str, str]) -> List[Dict[str, str]]:
    system = (
        "You are an expert SSC/PCS Current Affairs setter.\n"
        "Extract ONLY short, verifiable CURRENT AFFAIRS facts from the provided text.\n"
        "Hard Rules:\n"
        "- Use ONLY given text. No outside info.\n"
        "- Facts must be specific (names, places, awards, summits, reports, indices, appointments, schemes, sports results).\n"
        "- Avoid: video/channel/subscribe, tricks/mnemonics, static GK, history background, generic explanations.\n"
        "- Avoid deep banking/markets; macro economy (GDP/CPI/Budget) is allowed.\n"
        "- Return 10–50 facts.\n"
        "- event_key: 3–8 words summarizing the fact topic.\n"
    )
    user = {
        "source": article["source"],
        "title": article["title"],
        "url": article["url"],
        "date": article["date"],
        "text": article["text"][:9000],
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "facts_schema", "strict": True, "schema": FACTS_SCHEMA}},
    )
    data = json.loads(resp.output_text)
    facts = data.get("facts", []) or []

    # remove meta/video/trick/static gk/banking-ish facts
    cleaned = []
    for f in facts:
        ft = (f.get("fact") or "").lower()
        ek = (f.get("event_key") or "").lower()
        if any(p in ft for p in META_PHRASES) or any(p in ek for p in META_PHRASES):
            continue
        if any(p in ft for p in STATIC_GK_PHRASES):
            continue
        if banking_blocked(ft):
            continue
        if len((f.get("fact") or "").strip()) < 12:
            continue
        cleaned.append(f)

    return cleaned

def mcq_ok(mcq: Dict[str, Any]) -> bool:
    qtext = (mcq.get("question", "") or "").strip()
    qlow = qtext.lower()

    if any(p in qlow for p in META_PHRASES):
        return False
    if any(p in qlow for p in STATIC_GK_PHRASES):
        return False
    if banking_blocked(qtext + " " + " ".join(mcq.get("options", []) or []) + " " + (mcq.get("explanation","") or "")):
        return False

    if not is_exam_style(qtext):
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

def build_mcq_from_fact(article: Dict[str, str], fact_item: Dict[str, str]) -> Optional[Dict[str, Any]]:
    system = (
        "You are a professional SSC/PCS exam MCQ setter.\n"
        "Create ONE easy-to-moderate exam-style Current Affairs MCQ from the FACT.\n"
        "Hard Rules:\n"
        "- Use ONLY the given fact. Do NOT add outside knowledge.\n"
        "- Question must start with Which/What/Who/Where/When and end with '?'.\n"
        "- Make it practical and exam-relevant (not vague).\n"
        "- 4 options: same category, plausible, short.\n"
        "- No 'All of the above' or 'None of the above'.\n"
        "- Explanation <= 200 characters, directly justifying correct answer.\n"
        "- Avoid: video/channel/subscribe/tricks/static GK.\n"
    )
    user = {
        "event_key": fact_item.get("event_key", ""),
        "fact": fact_item.get("fact", ""),
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "mcq_schema", "strict": True, "schema": MCQ_SCHEMA}},
    )
    mcq = json.loads(resp.output_text)

    mcq["source"] = article["source"]
    mcq["source_title"] = article["title"][:200]
    mcq["source_url"] = article["url"][:400]

    return mcq if mcq_ok(mcq) else None


# ==================== BUILD DAILY MCQs ====================
def build_daily_mcqs() -> Tuple[str, List[Dict[str, Any]], List[str], List[str], List[str]]:
    date_str, articles = collect_fresh_articles()
    if not articles:
        return date_str, [], [], [], []

    hist = load_history()
    used_url_hashes = set(hist.get("url_hashes", [])[-6000:])
    used_fact_hashes = set(hist.get("fact_hashes", [])[-20000:])
    used_q_hashes = set(hist.get("q_hashes", [])[-20000:])

    # de-dup articles by url
    unique_articles = []
    seen_urls = set()
    for a in articles:
        if a["url"] in seen_urls:
            continue
        seen_urls.add(a["url"])

        # absolute "no repeat news pages"
        if _h(a["url"]) in used_url_hashes:
            continue

        unique_articles.append(a)

    # deterministic shuffle
    rnd = random.Random(date_str)
    rnd.shuffle(unique_articles)

    mcqs: List[Dict[str, Any]] = []
    used_urls_today: List[str] = []
    used_fact_hashes_today: List[str] = []
    used_q_hashes_today: List[str] = []

    for art in unique_articles:
        if len(mcqs) >= MAX_POLLS_PER_DAY:
            break

        try:
            facts = extract_facts(art)
        except Exception as e:
            print(f"⚠️ Facts extraction failed: {art['source']} | {art['url']} | {e}")
            continue

        if not facts:
            continue

        used_urls_today.append(art["url"])

        # shuffle facts for variety
        rnd2 = random.Random(_h(art["url"] + date_str))
        facts2 = facts[:]
        rnd2.shuffle(facts2)

        for f in facts2:
            if len(mcqs) >= MAX_POLLS_PER_DAY:
                break

            fact_text = (f.get("fact") or "").strip()
            if len(fact_text) < 12:
                continue

            # No repetition by fact
            fh = _h(_norm_text(fact_text))
            if fh in used_fact_hashes:
                continue

            # build mcq
            try:
                mcq = build_mcq_from_fact(art, f)
            except Exception as e:
                print(f"⚠️ MCQ gen failed: {art['source']} | {e}")
                continue

            if not mcq:
                continue

            qh = _h(_norm_text(mcq["question"]))
            if qh in used_q_hashes:
                continue

            used_fact_hashes.add(fh)
            used_q_hashes.add(qh)
            used_fact_hashes_today.append(fh)
            used_q_hashes_today.append(qh)

            mcqs.append(mcq)

    return date_str, mcqs, used_urls_today, used_fact_hashes_today, used_q_hashes_today


# ==================== POST ====================
def post_mcqs(date_str: str, mcqs: List[Dict[str, Any]]):
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


# ==================== MAIN ====================
def main():
    date_str, mcqs, used_urls, used_fact_hashes, used_q_hashes = build_daily_mcqs()

    if not mcqs:
        msg = (
            f"⚠️ Could not generate MCQs for {date_str}.\n"
            "Reason: sources might be unavailable/changed today.\n"
            "I will try again tomorrow automatically."
        )
        print(msg)
        post_info(msg)
        return

    # post whatever we got; don't crash if less than MIN_POLLS_TO_POST
    post_mcqs(date_str, mcqs)

    # update history
    hist = load_history()
    hist = update_history(hist, date_str, used_urls, used_fact_hashes, used_q_hashes)
    save_history(hist)

    print(f"✅ Posted {len(mcqs)} MCQs for {date_str} successfully.")

if __name__ == "__main__":
    main()
