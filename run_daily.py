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
MCQ_ITEM_SCHEMA = {
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
}

SET_SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "mcqs": {"type": "array", "minItems": 10, "maxItems": 10, "items": MCQ_ITEM_SCHEMA},
    },
    "required": ["date", "mcqs"],
    "additionalProperties": False,
}


# -------------------- HISTORY (NO-REPEAT ACROSS DAYS) --------------------
def load_history() -> dict:
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

    hist["event_keys"] = eks[-600:]
    hist["question_norms"] = qns[-600:]
    hist["url_hashes"] = uhs[-1000:]
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
    return None

def guess_date_from_url(url: str) -> Optional[str]:
    u = (url or "").lower()
    m = re.search(r"/(20\d{2})/(\d{2})/(\d{2})/", u)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None

def fetch_article_snippet(url: str, max_chars: int = 1000) -> str:
    html = http_get(url)
    return strip_html(html)[:max_chars]


# -------------------- SOURCES (ALLOWED) --------------------
def collect_affairscloud(date_iso: str) -> List[Dict[str, str]]:
    base = "https://affairscloud.com/current-affairs-ca/current-affairs-today/"
    html = http_get(base)
    links = re.findall(r'href=["\'](https://affairscloud\.com/current-affairs-[^"\']+)["\']', html, re.I)
    links = list(dict.fromkeys(links))[:70]

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
                items.append({"source": "GKToday", "url": url, "title": title, "date": pub})
        except Exception:
            continue
    return items

def collect_adda247(date_iso: str) -> List[Dict[str, str]]:
    base = "https://currentaffairs.adda247.com/"
    html = http_get(base)
    links = re.findall(r'href=["\'](https://currentaffairs\.adda247\.com/[^"\']+/)["\']', html, re.I)
    links = [u for u in links if "category" not in u and "tag" not in u and "/page/" not in u]
    links = list(dict.fromkeys(links))[:160]

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
    links = list(dict.fromkeys(links))[:160]

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

    # Dedup by URL
    seen = set()
    dedup = []
    for it in all_today:
        h = _url_hash(it["url"])
        if h not in seen:
            seen.add(h)
            dedup.append(it)

    # Fallback yesterday (only if not enough)
    if len(dedup) < 18:
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


# -------------------- BAN BANKING/FINANCE (ALLOW ECONOMY MACRO) --------------------
BANK_FIN_BLOCK = [
    "rbi", "repo", "reverse repo", "crr", "slr", "mclr", "mpc",
    "sebi", "irdai", "pfrda", "nbfc",
    "bank", "banking", "loan", "credit", "debit",
    "upi", "rtgs", "neft", "imps", "payment",
    "mutual fund", "ipo", "share market", "stock", "stocks", "bond", "g-sec", "treasury bill", "forex",
    "microfinance", "fintech", "insurance premium",
]

ECON_ALLOW = [
    "gdp", "inflation", "cpi", "wpi", "fiscal", "budget", "tax", "gst",
    "subsidy", "trade", "export", "import", "current account", "cad",
    "poverty", "employment", "unemployment", "agriculture", "msp",
]

def is_banking_finance_related(title: str, snippet: str) -> bool:
    txt = f"{title} {snippet}".lower()
    hit_bank = any(k in txt for k in BANK_FIN_BLOCK)
    hit_econ = any(k in txt for k in ECON_ALLOW)
    return hit_bank and not hit_econ


# -------------------- QUALITY GUARDS (STRICT) --------------------
VAGUE_PHRASES = [
    "recently concluded", "significant conference", "took place recently",
    "focusing on", "important conference", "major conference", "recent defence conference",
]

STATIC_GK_TRIGGERS = [
    "what does january", "commemorate", "republic day", "independence day",
    "labour day", "national youth day", "when is republic day",
]

BAD_OPTIONS = ["all of the above", "none of the above", "all above", "none"]

def extract_title_entities(title: str) -> List[str]:
    """
    Extract entities from title to force anchoring.
    Examples: 'Integrated Security Hub', 'Kalai-II', 'SOAR', 'CQB', 'PM Modi'
    """
    t = title or ""
    entities = set()

    # Acronyms / All-caps words length>=2
    for m in re.findall(r"\b[A-Z]{2,}\b", t):
        entities.add(m)

    # Hyphenated / with digits like Kalai-II, 72nd, etc.
    for m in re.findall(r"\b[A-Za-z]+[-–][A-Za-z0-9]+\b", t):
        entities.add(m)

    # Title-case sequences (2-4 words) like Integrated Security Hub
    for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", t):
        entities.add(m)

    # Single capitalized proper nouns (fallback)
    for m in re.findall(r"\b[A-Z][a-z]{3,}\b", t):
        entities.add(m)

    # remove very generic
    stop = {"India", "Indian", "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December", "Monday", "Tuesday",
            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Government", "Ministry"}
    entities = [e for e in entities if e not in stop and len(e) >= 4]
    # prefer longer entities first
    entities.sort(key=len, reverse=True)
    return entities[:8]

def contains_any_entity(text: str, entities: List[str]) -> bool:
    txt = (text or "").lower()
    for e in entities:
        if e.lower() in txt:
            return True
    return False

def is_exam_style_direct(qtext: str) -> bool:
    t = (qtext or "").strip().lower()
    # Reject UPSC-style statements
    if t.startswith(("with reference to", "consider the following statements", "which of the statements")):
        return False
    # Must be direct question (like daily quiz)
    if not t.endswith("?"):
        return False
    if not t.startswith(("which", "what", "who", "where", "when", "name", "the name")):
        # allow some direct forms if they still look like MCQ prompt
        return False
    return True

def passes_hard_filters(mcq: Dict[str, Any], title_entities: List[str]) -> bool:
    q = (mcq.get("question") or "")
    opts = mcq.get("options") or []
    exp = mcq.get("explanation") or ""

    # Must be anchored: include at least one entity from title in question
    if not contains_any_entity(q, title_entities):
        return False

    # No vague phrasing
    low = q.lower()
    if any(v in low for v in VAGUE_PHRASES):
        return False

    # No static GK triggers
    if any(s in low for s in STATIC_GK_TRIGGERS):
        return False

    # No all/none options
    if any((_norm_text(o) in [_norm_text(x) for x in BAD_OPTIONS]) for o in opts):
        return False

    # Direct exam style
    if not is_exam_style_direct(q):
        return False

    # Banking/finance ban (question/options/explanation)
    joined = (q + " " + " ".join(opts) + " " + exp).lower()
    if any(k in joined for k in BANK_FIN_BLOCK):
        # allow macro economy ONLY if economy keywords exist AND banking terms are not the core
        if not any(k in joined for k in ECON_ALLOW):
            return False
        # still reject if it's clearly RBI/SEBI/UPI etc.
        if any(k in joined for k in ["rbi", "sebi", "upi", "rtgs", "neft", "repo", "crr", "slr"]):
            return False

    # Options distinct
    if len(opts) != 4:
        return False
    if len(set(_norm_text(o) for o in opts)) != 4:
        return False

    # correct_answer mapping
    ca = mcq.get("correct_answer", "")
    if ca not in opts:
        return False
    if opts.index(ca) != mcq.get("correct_option_id", -1):
        # auto-fix later; still acceptable
        pass

    return True


# -------------------- MCQ GENERATION (ONE ARTICLE -> ONE MCQ) --------------------
def generate_one_mcq_from_article(article: Dict[str, str], avoid_event_keys: List[str], avoid_qnorms: List[str]) -> Optional[Dict[str, Any]]:
    title = article["title"]
    url = article["url"]
    snippet = article["snippet"]
    entities = extract_title_entities(title)

    # Model must use evidence only + must include entity
    system = (
        "You are a daily Current Affairs MCQ setter for SSC/State PCS/Railways/General Govt exams.\n"
        "Write ONE direct factual MCQ (daily-quiz style) from the given article evidence.\n"
        "STRICT:\n"
        "- Use ONLY the provided title+snippet. Do NOT add outside facts.\n"
        "- The question MUST include at least one key entity from the title (proper noun/term).\n"
        "- Question must start with: Which/What/Who/Where/When/Name and end with '?'.\n"
        "- Do NOT use: 'With reference to', 'Consider the following statements'.\n"
        "- Do NOT use 'All of the above' or 'None of the above'.\n"
        "- Do NOT create Banking/Finance CA (RBI/SEBI/UPI/repo/banks/markets). Economy macro allowed.\n"
        "- Options must be plausible and same-category.\n"
        "- correct_answer must exactly match one option; correct_option_id must match it.\n"
        "- Explanation <= 220 chars.\n"
        "- event_key must be a short unique identifier (3–10 words).\n"
    )

    user = {
        "source_title": title,
        "source_url": url,
        "snippet": snippet,
        "must_include_one_of_entities": entities[:6],
        "avoid_event_keys_norm": avoid_event_keys[:200],
        "avoid_question_norms": avoid_qnorms[:200],
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
                "name": "single_mcq",
                "strict": True,
                "schema": MCQ_ITEM_SCHEMA,
            }
        },
    )

    mcq = json.loads(resp.output_text)

    # Force source fields to match
    mcq["source_title"] = title[:200]
    mcq["source_url"] = url[:400]

    # Fix answer mapping if needed
    opts = mcq.get("options") or []
    ca = mcq.get("correct_answer", "")
    if ca in opts:
        mcq["correct_option_id"] = opts.index(ca)

    # Hard filter check
    if not passes_hard_filters(mcq, entities):
        return None

    return mcq


def generate_mcq_set() -> Dict[str, Any]:
    today_label, items = collect_fresh_items()
    if len(items) < 10:
        raise RuntimeError(f"Not enough fresh items from CA sources. Found {len(items)}.")

    hist = load_history()
    used_url_hashes = set(hist.get("url_hashes", [])[-1000:])
    used_event_keys = [_norm_text(x) for x in (hist.get("event_keys", [])[-600:])]
    used_qnorms = hist.get("question_norms", [])[-600:]

    # Build article pool with snippets + filters
    pool = []
    seen_title = set()
    for it in items[:120]:
        uh = _url_hash(it["url"])
        if uh in used_url_hashes:
            continue

        tn = _norm_text(it["title"])
        if tn in seen_title:
            continue
        seen_title.add(tn)

        try:
            snippet = fetch_article_snippet(it["url"], max_chars=1100)
        except Exception:
            snippet = ""

        # filter banking/finance at article level
        if is_banking_finance_related(it["title"], snippet):
            continue

        pool.append({**it, "snippet": snippet})

    if len(pool) < 14:
        raise RuntimeError(f"Not enough usable articles after banking/finance filter. Found {len(pool)}.")

    # Create 10 MCQs by iterating articles and regenerating when needed
    out: List[Dict[str, Any]] = []
    seen_event = set(_norm_text(x) for x in used_event_keys)
    seen_q = set(_norm_text(x) for x in used_qnorms)
    seen_url = set(used_url_hashes)

    max_attempts = 60
    attempts = 0

    # shuffle pool so you don't always pick same order
    rnd = random.Random(today_label)
    rnd.shuffle(pool)

    i = 0
    while len(out) < 10 and attempts < max_attempts and i < len(pool):
        art = pool[i]
        i += 1
        attempts += 1

        # Skip if URL already used (extra safety)
        if _url_hash(art["url"]) in seen_url:
            continue

        mcq = generate_one_mcq_from_article(
            art,
            avoid_event_keys=list(seen_event),
            avoid_qnorms=list(seen_q),
        )
        if not mcq:
            continue

        ek = _norm_text(mcq["event_key"])
        qn = _norm_text(mcq["question"])
        uh = _url_hash(mcq["source_url"])

        if ek in seen_event or qn in seen_q or uh in seen_url:
            continue

        # accept
        seen_event.add(ek)
        seen_q.add(qn)
        seen_url.add(uh)
        out.append(mcq)

    if len(out) < 10:
        raise RuntimeError(f"Could not build 10 high-quality anchored MCQs. Got {len(out)}.")

    return {"date": today_label, "mcqs": out}


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
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True})

def post_score_poll(date_str: str):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "question": f"📊 Vote your score ({date_str}) ✅\nHow many were correct out of 10?",
        "options": ["10/10 🏆", "9/10 🔥", "8/10 ✅", "7/10 👍", "6/10 🙂", "5/10 📘", "4/10 🧠", "3 or less 😅"],
        "is_anonymous": True,
        "allows_multiple_answers": False,
    }
    tg("sendPoll", payload)

def post_to_channel(mcq_set: dict):
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇"})
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
    mcq_set = generate_mcq_set()
    post_to_channel(mcq_set)

    hist = load_history()
    hist = update_history(hist, mcq_set, keep_last_days=KEEP_LAST_DAYS)
    save_history(hist)

    print("✅ Posted 10 quiz polls + message + score poll successfully.")

if __name__ == "__main__":
    main()
