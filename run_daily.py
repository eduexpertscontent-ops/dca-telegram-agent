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

# ============================================================
# DAILY CURRENT AFFAIRS MCQ BOT (SSC + PCS) — 10 QUIZ POLLS
# - Uses only: NextIAS + GKToday + DrishtiIAS
# - Strictly current: TODAY only (fallback to yesterday only if too few)
# - No repetition of topics across last N days (by URL + event_key + question)
# - Easy exam-like framing (SSC/PCS style)
# - Soft international target (won't crash if intl is low)
# ============================================================

# -------------------- CONFIG --------------------
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # e.g., @UPPCSSUCCESS
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

KEEP_LAST_DAYS = int(os.getenv("KEEP_LAST_DAYS", "15"))

SLEEP_BETWEEN_POLLS = float(os.getenv("SLEEP_BETWEEN_POLLS", "2"))
SLEEP_AFTER_QUIZ = float(os.getenv("SLEEP_AFTER_QUIZ", "2"))

# HARD CURRENTNESS CONTROLS
MIN_TODAY_ITEMS = int(os.getenv("MIN_TODAY_ITEMS", "18"))   # if < this, pull yesterday too
ALLOW_YESTERDAY_FALLBACK = os.getenv("ALLOW_YESTERDAY_FALLBACK", "1") == "1"

# SOFT CONTENT MIX
MIN_INTL = int(os.getenv("MIN_INTL", "2"))  # soft target only

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
        "question": {"type": "string", "minLength": 1, "maxLength": 260},
        "options": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
        },
        "correct_option_id": {"type": "integer", "minimum": 0, "maximum": 3},
        "correct_answer": {"type": "string", "minLength": 1, "maxLength": 80},
        "explanation": {"type": "string", "maxLength": 200},
        "source_url": {"type": "string", "minLength": 5, "maxLength": 400},
        "source_title": {"type": "string", "minLength": 3, "maxLength": 200},
        "source": {"type": "string", "minLength": 2, "maxLength": 40},
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
        "source",
    ],
    "additionalProperties": False,
}

# -------------------- HISTORY --------------------
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

    hist["event_keys"] = eks[-900:]
    hist["question_norms"] = qns[-900:]
    hist["url_hashes"] = uhs[-1500:]
    return hist

# -------------------- HTTP + PARSING --------------------
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
    return None

def fetch_article_snippet(url: str, max_chars: int = 1400) -> Tuple[str, Optional[str]]:
    """
    Returns: (snippet_text, published_date_iso_if_found)
    """
    html = http_get(url)
    pub = extract_meta_date(html) or guess_date_from_url(url)
    return strip_html(html)[:max_chars], pub

# -------------------- BLOCK META/LISTS --------------------
BLOCK_TITLE_PHRASES = [
    "important days", "appointments", "current affairs pdf", "weekly current affairs",
    "monthly current affairs", "capsule", "compilation", "quiz", "mcq", "practice questions",
    "questions and answers", "obituary current affairs", "year 2026",
]
VAGUE_PHRASES = [
    "recently concluded", "significant conference", "took place recently",
    "what kind of events", "what is the focus of", "section cover", "section covers",
]
STATIC_GK_PHRASES = [
    "queen of the hills", "national anthem", "national animal", "largest planet",
    "capital of", "currency of", "highest mountain"
]
BAD_OPTIONS = ["all of the above", "none of the above", "all above", "none"]

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

def enforce_no_banking_finance_text(text: str) -> bool:
    low = (text or "").lower()
    if any(k in low for k in BANK_FIN_BLOCK):
        # allow macro economy keywords, but still block RBI/SEBI/UPI etc.
        if any(k in low for k in ECON_ALLOW) and not any(x in low for x in ["rbi", "sebi", "upi", "rtgs", "neft", "repo", "crr", "slr"]):
            return True
        return False
    return True

# -------------------- INTERNATIONAL DETECTION (ROBUST) --------------------
INDIA_MARKERS = [
    "india", "indian", "bharat", "new delhi", "delhi", "parliament",
    "uttar pradesh", "bihar", "madhya pradesh", "rajasthan", "gujarat", "maharashtra",
    "tamil nadu", "karnataka", "kerala", "west bengal", "odisha", "telangana", "andhra pradesh",
    "punjab", "haryana", "jharkhand", "chhattisgarh", "assam", "sikkim", "goa", "ladakh",
    "jammu", "kashmir", "lakshadweep", "andaman", "nicobar",
    "supreme court", "isro", "drdo", "ministry of", "cabinet", "pm modi"
]
INTERNATIONAL_MARKERS = [
    "united nations", "un ", "who", "unesco", "imf", "world bank", "wto", "nato",
    "g20", "brics", "european union", "eu ", "asean", "quad", "opec", "cop", "summit", "treaty"
]
COUNTRY_MARKERS = [
    "usa", "u.s.", "united states", "uk", "britain", "russia", "china", "japan", "france",
    "germany", "canada", "mexico", "brazil", "australia", "south korea", "north korea",
    "iran", "israel", "saudi", "uae", "qatar", "turkey", "pakistan", "bangladesh",
    "sri lanka", "nepal", "bhutan", "thailand", "vietnam", "indonesia", "singapore",
    "egypt", "south africa", "nigeria", "kenya"
]

def is_international_item(title: str, snippet: str) -> bool:
    """
    Score-based classifier:
    - If India isn't mentioned, any intl/country marker => international.
    - If India is mentioned, require stronger intl signal so India+World events still count.
    """
    txt = f"{title} {snippet}".lower()

    india_hits = sum(1 for k in INDIA_MARKERS if k in txt)
    intl_hits = sum(1 for k in INTERNATIONAL_MARKERS if k in txt)
    country_hits = sum(1 for k in COUNTRY_MARKERS if k in txt)

    strong = (intl_hits + country_hits) >= 1

    if india_hits == 0:
        return strong

    # India mentioned: need at least 2 global signals
    return (intl_hits + country_hits) >= 2

# -------------------- SOURCES (ONLY YOUR 3) --------------------
def collect_nextias(date_iso: str) -> List[Dict[str, str]]:
    base = "https://www.nextias.com/daily-current-affairs"
    html = http_get(base)
    links = re.findall(r'href=["\'](https://www\.nextias\.com/daily-current-affairs/[^"\']+)["\']', html, re.I)
    links = [u for u in links if u.rstrip("/") != base.rstrip("/")]
    links = list(dict.fromkeys(links))[:180]
    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            # accept today only here (unknown allowed); strictness happens later using page pub
            if pub == date_iso or pub == "":
                items.append({"source": "NextIAS", "url": url, "title": title, "date": pub or date_iso})
        except Exception:
            continue
    return items

def collect_gktoday(date_iso: str) -> List[Dict[str, str]]:
    base = "https://www.gktoday.in/current-affairs/"
    html = http_get(base)
    links = re.findall(r'href=["\'](https://www\.gktoday\.in/[^"\']+)["\']', html, re.I)
    links = [u for u in links if "wp-content" not in u]
    links = list(dict.fromkeys(links))[:240]
    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == date_iso or pub == "":
                items.append({"source": "GKToday", "url": url, "title": title, "date": pub or date_iso})
        except Exception:
            continue
    return items

def collect_drishtiias(date_iso: str) -> List[Dict[str, str]]:
    base = "https://www.drishtiias.com/current-affairs-news-analysis-editorials"
    html = http_get(base)
    links = re.findall(
        r'href=["\'](https://www\.drishtiias\.com/current-affairs-news-analysis-editorials/[^"\']+)["\']',
        html, re.I
    )
    links = [u for u in links if u.rstrip("/") != base.rstrip("/")]
    links = list(dict.fromkeys(links))[:260]
    items = []
    for url in links:
        try:
            page = http_get(url)
            title = extract_title(page)
            if not title:
                continue
            pub = extract_meta_date(page) or guess_date_from_url(url) or ""
            if pub == date_iso or pub == "":
                items.append({"source": "DrishtiIAS", "url": url, "title": title, "date": pub or date_iso})
        except Exception:
            continue
    return items

def collect_fresh_items() -> Tuple[str, List[Dict[str, str]]]:
    """
    Strictly prefers TODAY. Only uses yesterday if too few today items AND fallback enabled.
    """
    today = ist_today_str()
    yesterday = ist_yesterday_str()
    sources = [collect_nextias, collect_gktoday, collect_drishtiias]

    all_today = []
    for fn in sources:
        try:
            got = fn(today)
            print(f"✅ {fn.__name__}: {len(got)} items (today/unknown accepted) for {today}")
            all_today.extend(got)
        except Exception as e:
            print(f"⚠️ {fn.__name__} failed: {e}")

    # Dedup by URL
    seen = set()
    dedup_today = []
    for it in all_today:
        h = _url_hash(it["url"])
        if h not in seen:
            seen.add(h)
            dedup_today.append(it)

    # Only fallback if too few
    dedup = list(dedup_today)
    if ALLOW_YESTERDAY_FALLBACK and len(dedup_today) < MIN_TODAY_ITEMS:
        print(f"⚠️ Only {len(dedup_today)} items for today. Trying yesterday fallback...")
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

# -------------------- EXAM-LIKE EASY FRAMING CHECKS --------------------
def is_exam_style_easy(qtext: str) -> bool:
    t = (qtext or "").strip()
    low = t.lower()

    if not t.endswith("?"):
        return False

    starters = ("which", "what", "who", "where", "when")
    if not low.startswith(starters):
        return False

    if low.startswith(("with reference to", "consider the following statements")):
        return False

    if "section" in low or "category" in low:
        return False
    if any(p in low for p in VAGUE_PHRASES):
        return False

    return True

def extract_title_entities(title: str) -> List[str]:
    t = title or ""
    entities = set()

    for m in re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", t):
        entities.add(m)

    for m in re.findall(r"\b[A-Z]{2,}\b", t):
        entities.add(m)

    for m in re.findall(r"\b[A-Z][a-z]{3,}\b", t):
        entities.add(m)

    stop = {"India", "Indian", "Government", "Ministry"}
    entities = [e for e in entities if e not in stop and len(e) >= 4]
    entities.sort(key=len, reverse=True)
    return entities[:8]

def contains_entity_relaxed(mcq: Dict[str, Any], entities: List[str]) -> bool:
    if not entities:
        return True
    joined = (
        (mcq.get("question") or "") + " " +
        " ".join(mcq.get("options") or []) + " " +
        (mcq.get("explanation") or "")
    ).lower()
    return any(e.lower() in joined for e in entities)

def passes_hard_filters(mcq: Dict[str, Any], title_entities: List[str]) -> bool:
    q = (mcq.get("question") or "")
    opts = mcq.get("options") or []
    exp = mcq.get("explanation") or ""
    title = (mcq.get("source_title") or "").lower()

    qlow = q.lower()

    if any(p in title for p in BLOCK_TITLE_PHRASES):
        return False

    if any(v in qlow for v in VAGUE_PHRASES):
        return False

    if any(p in qlow for p in STATIC_GK_PHRASES):
        return False

    if any(_norm_text(o) in [_norm_text(x) for x in BAD_OPTIONS] for o in opts):
        return False

    if not is_exam_style_easy(q):
        return False

    joined = (q + " " + " ".join(opts) + " " + exp)
    if not enforce_no_banking_finance_text(joined):
        return False

    if len(opts) != 4:
        return False
    if len(set(_norm_text(o) for o in opts)) != 4:
        return False

    ca = mcq.get("correct_answer", "")
    if ca not in opts:
        return False

    if not contains_entity_relaxed(mcq, title_entities):
        return False

    return True

# -------------------- MCQ GENERATION --------------------
def generate_one_mcq_from_article(
    article: Dict[str, Any],
    avoid_event_keys: List[str],
    avoid_qnorms: List[str],
) -> Optional[Dict[str, Any]]:
    title = article["title"]
    url = article["url"]
    snippet = article["snippet"]
    entities = extract_title_entities(title)

    system = (
        "You create EASY, exam-like Current Affairs MCQs for SSC/PCS.\n"
        "STRICT RULES:\n"
        "- Use ONLY the given title + snippet. Do NOT add outside facts.\n"
        "- Question MUST be direct like real exams.\n"
        "- Start with Which/What/Who/Where/When and end with '?'.\n"
        "- Do NOT use: 'With reference to' or 'Consider the following statements'.\n"
        "- Do NOT ask meta questions about sections/categories.\n"
        "- Do NOT use 'All of the above'/'None of the above'.\n"
        "- Avoid Banking/Finance (RBI/SEBI/UPI/banks/markets). Economy macro is allowed.\n"
        "- Options must be short, factual, and same-category.\n"
        "- correct_answer must exactly match one option; correct_option_id must match.\n"
        "- Explanation <= 200 characters.\n"
        "- event_key must be a short unique identifier (3–10 words).\n"
    )

    user = {
        "source": article["source"],
        "source_title": title,
        "source_url": url,
        "snippet": snippet,
        "anchor_entities_hint": entities[:6],
        "avoid_event_keys_norm": avoid_event_keys[:350],
        "avoid_question_norms": avoid_qnorms[:350],
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
                "name": "single_easy_exam_mcq",
                "strict": True,
                "schema": MCQ_ITEM_SCHEMA,
            }
        },
    )

    mcq = json.loads(resp.output_text)

    mcq["source"] = article["source"]
    mcq["source_title"] = title[:200]
    mcq["source_url"] = url[:400]

    opts = mcq.get("options") or []
    ca = mcq.get("correct_answer", "")
    if ca in opts:
        mcq["correct_option_id"] = opts.index(ca)

    if not passes_hard_filters(mcq, entities):
        return None

    return mcq

def generate_mcq_set() -> Dict[str, Any]:
    today_label, items = collect_fresh_items()
    if len(items) < 10:
        raise RuntimeError(f"Not enough items from sources. Found {len(items)}.")

    hist = load_history()
    used_url_hashes = set(hist.get("url_hashes", [])[-1500:])
    used_event_keys = [_norm_text(x) for x in (hist.get("event_keys", [])[-900:])]
    used_qnorms = hist.get("question_norms", [])[-900:]

    pool = []
    seen_title = set()

    for it in items[:260]:
        uh = _url_hash(it["url"])
        if uh in used_url_hashes:
            continue

        tn = _norm_text(it["title"])
        if tn in seen_title:
            continue
        seen_title.add(tn)

        title_low = (it["title"] or "").lower()
        if any(p in title_low for p in BLOCK_TITLE_PHRASES):
            continue

        # Fetch snippet + discover actual publish date
        try:
            snippet, pub = fetch_article_snippet(it["url"], max_chars=1400)
        except Exception:
            snippet, pub = "", None

        # STRICT CURRENTNESS:
        # If pub exists and it's neither today nor yesterday (only if fallback enabled), reject.
        if pub:
            if pub != today_label:
                if not (ALLOW_YESTERDAY_FALLBACK and pub == ist_yesterday_str()):
                    continue

        joined_low = (it["title"] + " " + snippet).lower()

        # block banking/finance articles (allow macro economy)
        if any(k in joined_low for k in BANK_FIN_BLOCK) and not any(k in joined_low for k in ECON_ALLOW):
            continue

        pool.append({
            **it,
            "snippet": snippet,
            "pub": pub or it.get("date") or today_label,
            "is_international": is_international_item(it["title"], snippet),
        })

    if len(pool) < 10:
        raise RuntimeError(f"Not enough usable articles after strict currentness + filters. Found {len(pool)}.")

    print("🧾 Pool size:", len(pool))
    print("🌍 Intl candidates in pool:", sum(1 for a in pool if a.get("is_international")))

    rnd = random.Random(today_label)
    rnd.shuffle(pool)

    out: List[Dict[str, Any]] = []
    seen_event = set(_norm_text(x) for x in used_event_keys)
    seen_q = set(_norm_text(x) for x in used_qnorms)
    seen_url = set(used_url_hashes)

    intl_count = 0
    max_attempts = 260
    attempts = 0

    # 1) Try to pick international first (soft target)
    for art in pool:
        if intl_count >= MIN_INTL or len(out) >= 10:
            break
        if not art.get("is_international", False):
            continue

        attempts += 1
        if attempts > max_attempts:
            break

        mcq = generate_one_mcq_from_article(art, list(seen_event), list(seen_q))
        if not mcq:
            continue

        ek = _norm_text(mcq["event_key"])
        qn = _norm_text(mcq["question"])
        uh2 = _url_hash(mcq["source_url"])
        if ek in seen_event or qn in seen_q or uh2 in seen_url:
            continue

        seen_event.add(ek); seen_q.add(qn); seen_url.add(uh2)
        out.append(mcq)
        intl_count += 1

    # 2) Fill remaining with any
    for art in pool:
        if len(out) >= 10:
            break

        attempts += 1
        if attempts > max_attempts:
            break

        mcq = generate_one_mcq_from_article(art, list(seen_event), list(seen_q))
        if not mcq:
            continue

        ek = _norm_text(mcq["event_key"])
        qn = _norm_text(mcq["question"])
        uh2 = _url_hash(mcq["source_url"])
        if ek in seen_event or qn in seen_q or uh2 in seen_url:
            continue

        seen_event.add(ek); seen_q.add(qn); seen_url.add(uh2)
        out.append(mcq)
        if art.get("is_international", False):
            intl_count += 1

    # Ensure we produce 10 daily (or fail clearly)
    if len(out) < 10:
        raise RuntimeError(
            f"Could not build 10 fresh, current SSC/PCS MCQs today. Got {len(out)}. "
            f"Try increasing MIN_TODAY_ITEMS fallback, or allow yesterday fallback."
        )

    # Soft intl warning only (no crash)
    if intl_count < MIN_INTL:
        print(f"⚠️ Intl shortfall: needed {MIN_INTL}, got {intl_count}. Proceeding anyway.")

    return {"date": today_label, "mcqs": out[:10]}

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
        "🏁 Today’s Practice Ends Here!\n\n"
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
    tg("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧠 Daily Current Affairs Practice (SSC/PCS) ({mcq_set['date']})\n\nMCQ polls below 👇"})
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

    print("✅ Posted 10 quiz polls + closure message + score poll successfully.")

if __name__ == "__main__":
    main()
