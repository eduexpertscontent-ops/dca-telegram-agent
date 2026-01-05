import os
import json
import time
import datetime as dt
import requests
import random
import re
import hashlib
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
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

NEWS_TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "25"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "260"))
CONTEXT_ITEMS = int(os.getenv("CONTEXT_ITEMS", "80"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "140"))
MIN_TODAY_ITEMS_REQUIRED = int(os.getenv("MIN_TODAY_ITEMS_REQUIRED", "18"))

HISTORY_FILE = "mcq_history.json"

client = OpenAI()  # reads OPENAI_API_KEY from environment
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


# -------------------- TELEGRAM HELPERS --------------------
def tg(method: str, payload: dict) -> dict:
    """
    Telegram API wrapper with basic retry on rate limits.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    last_err = None

    for _ in range(4):  # retry up to 4 times
        r = requests.post(url, json=payload, timeout=30)
        data = r.json()

        if data.get("ok"):
            return data

        last_err = data

        # Rate limit / flood control
        if data.get("error_code") == 429:
            retry_after = (data.get("parameters") or {}).get("retry_after", 5)
            print(f"⚠️ Telegram rate limit. Retrying after {retry_after}s...")
            time.sleep(int(retry_after) + 1)
            continue

        # Other Telegram errors: raise
        raise RuntimeError(f"Telegram error: {data}")

    raise RuntimeError(f"Telegram error after retries: {last_err}")


# -------------------- JSON SCHEMA (UPSC-level) --------------------
# Note: Adds mcq_type, difficulty, source_hint to enforce quality; these are NOT posted to Telegram.
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
                    "event_key": {"type": "string", "minLength": 3, "maxLength": 60},
                    "mcq_type": {"type": "string", "minLength": 3, "maxLength": 30},
                    "difficulty": {"type": "string", "enum": ["Easy", "Moderate", "Tough"]},
                    "question": {"type": "string", "minLength": 1, "maxLength": 300},
                    "options": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                    "correct_option_id": {"type": "integer", "minimum": 0, "maximum": 3},
                    "correct_answer": {"type": "string", "minLength": 1, "maxLength": 120},
                    "explanation": {"type": "string", "maxLength": 220},
                    "source_hint": {"type": "string", "maxLength": 80},
                },
                "required": [
                    "event_key",
                    "mcq_type",
                    "difficulty",
                    "question",
                    "options",
                    "correct_option_id",
                    "correct_answer",
                    "explanation",
                    "source_hint",
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
        "YYYY-MM-DD": {"questions":[...], "event_keys":[...], "title_norms":[...], "url_hashes":[...]}
      },
      "questions":[...],
      "event_keys":[...],
      "title_norms":[...],
      "url_hashes":[...]
    }
    """
    if not os.path.exists(HISTORY_FILE):
        return {"dates": {}, "questions": [], "event_keys": [], "title_norms": [], "url_hashes": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        hist.setdefault("dates", {})
        hist.setdefault("questions", [])
        hist.setdefault("event_keys", [])
        hist.setdefault("title_norms", [])
        hist.setdefault("url_hashes", [])
        return hist
    except Exception:
        return {"dates": {}, "questions": [], "event_keys": [], "title_norms": [], "url_hashes": []}


def save_history(hist: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def update_history_with_set(hist: dict, mcq_set: dict, keep_last_days: int, used_title_norms, used_url_hashes) -> dict:
    today = mcq_set["date"]
    hist.setdefault("dates", {})
    hist["dates"][today] = {
        "questions": [q["question"] for q in mcq_set["mcqs"]],
        "event_keys": [q["event_key"] for q in mcq_set["mcqs"]],
        "title_norms": list(used_title_norms),
        "url_hashes": list(used_url_hashes),
    }

    # prune old dates
    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > keep_last_days:
        for d in all_dates[:-keep_last_days]:
            hist["dates"].pop(d, None)

    # rebuild flattened lists (bounded)
    qs, eks, tns, uhs = [], [], [], []
    for d in sorted(hist["dates"].keys()):
        qs.extend(hist["dates"][d].get("questions", []))
        eks.extend(hist["dates"][d].get("event_keys", []))
        tns.extend(hist["dates"][d].get("title_norms", []))
        uhs.extend(hist["dates"][d].get("url_hashes", []))

    hist["questions"] = qs[-700:]
    hist["event_keys"] = eks[-700:]
    hist["title_norms"] = tns[-1200:]
    hist["url_hashes"] = uhs[-1200:]
    return hist


# -------------------- NORMALIZATION / DEDUPE --------------------
def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()


def _hash_url(url: str) -> str:
    return hashlib.sha1((url or "").encode("utf-8")).hexdigest()[:16]


def _dedupe_keep_order_by_key(items, key_fn):
    seen = set()
    out = []
    for it in items:
        k = key_fn(it)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _norm_q(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


# -------------------- RSS FETCH --------------------
def _get(url: str) -> str:
    r = requests.get(url, timeout=NEWS_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def _safe_parse_date(pub: str):
    if not pub:
        return None
    try:
        d = parsedate_to_datetime(pub)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(IST)
    except Exception:
        return None


def _parse_rss_items(xml_text: str):
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        desc = (item.findtext("description") or "").strip()

        pub_dt = _safe_parse_date(pub)

        items.append(
            {
                "title": title,
                "link": link,
                "published_ist": pub_dt,
                "description": re.sub(r"<.*?>", "", desc)[:280],
            }
        )
    return items


# -------------------- SOURCES (UPSC + GOVT EXAM PREP) --------------------
def fetch_news_pool():
    """
    Mix of Tier-1 UPSC-friendly + govt-exam-prep CA sites (as requested).
    """
    feeds = [
        # Tier-1 (UPSC-friendly)
        ("PIB", "https://pib.gov.in/RssMain.aspx?mod=0&ln=1"),
        ("PRS", "https://prsindia.org/rss.xml"),
        ("IndianExpress-Explained", "https://indianexpress.com/section/explained/feed/"),
        ("DownToEarth", "https://www.downtoearth.org.in/rss"),

        # Govt-exam-prep CA (your allowed list)
        ("AffairsCloud", "https://affairscloud.com/feed/"),
        ("GKToday", "https://www.gktoday.in/feed/"),
        ("JagranJosh-CA", "https://www.jagranjosh.com/rss/current-affairs.xml"),
        ("Adda247-CA", "https://www.adda247.com/jobs/feed/"),
    ]

    all_items = []
    for source, url in feeds:
        try:
            xml_text = _get(url)
            items = _parse_rss_items(xml_text)
            for it in items:
                it["source"] = source
            all_items.extend(items)
        except Exception as e:
            print(f"⚠️ Failed to fetch {source}: {e}")

    all_items = [x for x in all_items if x.get("title") and x.get("link")]
    all_items = all_items[:MAX_NEWS_ITEMS]
    all_items = _dedupe_keep_order_by_key(all_items, lambda x: (_norm_text(x["title"]), _hash_url(x["link"])))
    return all_items


# -------------------- FRESHNESS FILTER (IST, SAME DAY) --------------------
def filter_news_today_ist(items, avoid_title_norms, avoid_url_hashes):
    now_ist = dt.datetime.now(IST)
    today = now_ist.date()

    def is_today(it):
        return it.get("published_ist") and it["published_ist"].date() == today

    today_items = [it for it in items if is_today(it)]

    # If too few today items, allow last 24h
    if len(today_items) < MIN_TODAY_ITEMS_REQUIRED:
        cutoff = now_ist - dt.timedelta(hours=24)
        pool = [it for it in items if it.get("published_ist") and it["published_ist"] >= cutoff]
    else:
        pool = today_items

    out = []
    for it in pool:
        tn = _norm_text(it["title"])
        uh = _hash_url(it["link"])
        if tn in avoid_title_norms:
            continue
        if uh in avoid_url_hashes:
            continue
        out.append(it)

    out = _dedupe_keep_order_by_key(out, lambda x: _norm_text(x["title"]))
    return out


# -------------------- EXCLUDE BANKING/FINANCE (ALLOW ECONOMY) --------------------
BANK_FIN_EXCLUDE = [
    "rbi", "repo", "reverse repo", "crr", "slr", "mclr", "mpc",
    "nbfc", "banking", "bank ", "banks", "loan", "credit", "debit",
    "sebi", "irdai", "pfrda", "mutual fund", "ipo", "share market",
    "bond", "g-sec", "treasury bill", "forex", "rupee hits", "stocks",
    "upi", "payments", "digital payment", "fintech",
]

ECON_ALLOWED = [
    "gdp", "inflation", "cpi", "wpi", "fiscal", "budget", "tax",
    "gst", "subsidy", "poverty", "employment", "unemployment",
    "agriculture", "msp", "food security", "trade", "export", "import",
    "current account", "cad", "imf", "world bank", "economic survey",
]

def is_banking_finance_item(item: dict) -> bool:
    txt = (item.get("title","") + " " + item.get("description","")).lower()
    bank_hit = any(k in txt for k in BANK_FIN_EXCLUDE)
    econ_hit = any(k in txt for k in ECON_ALLOWED)
    return bank_hit and not econ_hit


# -------------------- UPSC-WORTHINESS FILTER --------------------
UPSC_CORE_KEYWORDS = [
    # Polity/Governance/Law
    "bill", "act", "amendment", "ordinance", "parliament", "supreme court", "high court",
    "committee", "commission", "tribunal", "authority", "constitution",
    # Governance/programmes
    "scheme", "yojana", "mission", "policy", "guidelines", "framework",
    # Environment/Geo
    "biodiversity", "wildlife", "tiger reserve", "national park", "wetland", "climate", "pollution",
    "glacier", "river", "basin", "earthquake", "cyclone",
    # IR/Summits
    "g20", "brics", "sco", "cop", "asean", "quad", "un ", "who", "wto",
    # Economy (non-banking)
    "gdp", "inflation", "cpi", "fiscal", "budget", "gst", "subsidy", "trade",
    # Science/Tech (UPSC)
    "isro", "satellite", "space", "semiconductor", "quantum", "biotech", "genome",
]

SHALLOW_PATTERNS = [
    "brand ambassador", "celebrated", "anniversary", "wins", "won the match",
    "trailer", "box office", "viral", "launched a new app", "opened a new outlet",
]

def is_upsc_worthy(item: dict) -> bool:
    txt = (item.get("title","") + " " + item.get("description","")).lower()
    if any(p in txt for p in SHALLOW_PATTERNS):
        return False
    return any(k in txt for k in UPSC_CORE_KEYWORDS)


# -------------------- EXAM RELEVANCY SCORING (POSITIVE ONLY) --------------------
POSITIVE_KEYWORDS = [
    "parliament", "bill", "act", "amendment", "committee", "commission", "tribunal",
    "scheme", "mission", "policy", "guidelines", "framework",
    "biodiversity", "wildlife", "wetland", "climate", "pollution",
    "isro", "satellite", "space", "quantum", "semiconductor", "biotech",
    "g20", "brics", "sco", "cop", "un", "who", "wto",
    "gdp", "inflation", "cpi", "fiscal", "budget", "gst", "trade",
]

def exam_relevancy_score(item: dict) -> int:
    title = _norm_text(item.get("title", ""))
    desc = _norm_text(item.get("description", ""))
    text = f"{title} {desc}"

    score = 0

    src = (item.get("source") or "").lower()
    if "pib" in src:
        score += 12
    elif "prs" in src:
        score += 12
    elif "indianexpress-explained" in src:
        score += 10
    elif "downtoearth" in src:
        score += 10
    elif "affairscloud" in src:
        score += 5
    elif "gktoday" in src:
        score += 5
    elif "jagranjosh" in src:
        score += 3
    elif "adda247" in src:
        score += 3

    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            score += 2

    if "india" in text or "indian" in text:
        score += 1

    if len(title) < 25:
        score -= 1

    return score


def rank_news(items):
    scored = []
    for it in items:
        it = dict(it)
        it["exam_score"] = exam_relevancy_score(it)
        scored.append(it)

    scored.sort(
        key=lambda x: (x.get("exam_score", 0), x.get("published_ist") or dt.datetime.min.replace(tzinfo=IST)),
        reverse=True,
    )
    return scored


def build_news_context(items, max_items: int):
    picked = items[:max_items]
    lines = []
    for i, it in enumerate(picked, start=1):
        pub = it.get("published_ist")
        pub_s = pub.strftime("%Y-%m-%d %H:%M IST") if pub else "unknown time"
        lines.append(
            f"{i}. [score={it.get('exam_score',0)}] [{it['source']}] {pub_s} | {it['title']} | {it['link']} | {it.get('description','')}"
        )
    return "\n".join(lines)


# -------------------- OPENAI: UPSC PRELIMS STYLE (NO BANKING/FINANCE) --------------------
UPSC_STYLE_RULES = """
You are a UPSC CSE Prelims (GS Paper 1) question setter.

Core philosophy:
- Test conceptual understanding linked with current affairs.
- Avoid one-line trivia.
- Prefer statement-based MCQs.

Mandatory distribution:
- At least 7 out of 10 questions MUST be statement-based (2 or 3 statements).
- Remaining can be: correctly matched pairs, institutions & functions, environment & geography, reports/indices (concept + purpose).

Hard exclusions:
- NO banking/finance MCQs: RBI/SEBI/IRDAI/PFRDA, repo/CRR/SLR, UPI/payments, banks/NBFCs, stock/IPO/mutual funds.
- NO sports/entertainment/trivial awards.
- NO trivial appointments (unless constitutional/major international office).

Statement option format (use exactly this when using 2 statements):
(a) 1 only
(b) 2 only
(c) Both 1 and 2
(d) Neither 1 nor 2

Other hard rules:
1) Use ONLY the provided news items. Do not invent facts.
2) Every MCQ must be based on a DIFFERENT news story.
3) Make options close/plausible (UPSC-like distractors).
4) Keep explanation <= 220 chars and include:
   - Current fact (what happened)
   - Static concept link (what is it/role/definition)
5) Use formal UPSC tone (no quiz language).
"""

def generate_mcqs_upsc(today_str: str, news_context: str, avoid_questions: list, avoid_event_keys: list):
    system = (
        "You write UPSC Prelims-quality MCQs in ENGLISH.\n"
        "Use ONLY the provided news list.\n"
        + UPSC_STYLE_RULES +
        "\nReturn STRICT JSON as per schema."
    )

    avoid_block = "\n".join([f"- {a}" for a in (avoid_questions or [])[:200]])
    avoid_keys_block = "\n".join([f"- {a}" for a in (avoid_event_keys or [])[:200]])

    user = (
        f"DATE (IST): {today_str}\n\n"
        "NEWS ITEMS (use ONLY these):\n"
        f"{news_context}\n\n"
        "Create EXACTLY 10 UPSC Prelims-quality MCQs.\n"
        "- Pick the MOST UPSC-relevant items.\n"
        "- No fixed quotas by topic.\n"
        "- Ensure India coverage is present naturally if today's items support it.\n\n"
        "AVOID repeating ANY of these prior questions/topics:\n"
        f"{avoid_block}\n\n"
        "AVOID these prior event_keys/topics:\n"
        f"{avoid_keys_block}\n\n"
        "For each MCQ also include:\n"
        "- mcq_type (e.g., Statement-based / Correct match / Institution / Environment / Report)\n"
        "- difficulty (Easy/Moderate/Tough)\n"
        "- source_hint (just source label like PIB/PRS/GKToday etc.)\n"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text={"format": {"type": "json_schema", "name": "daily_upsc_mcqs", "strict": True, "schema": SCHEMA}},
    )
    return json.loads(resp.output_text)


# -------------------- QUALITY CHECKS --------------------
def looks_upsc_style(q: dict) -> bool:
    text = (q.get("question") or "").strip().lower()
    # UPSC style often begins with these
    starters = (
        "consider the following statements",
        "with reference to",
        "which of the following",
        "in the context of",
        "recently",
    )
    if not text.startswith(starters):
        return False

    opts = q.get("options") or []
    if len(opts) != 4:
        return False
    if len(set(o.strip() for o in opts)) != 4:
        return False

    exp = (q.get("explanation") or "").strip()
    if len(exp) < 55:
        return False

    # Avoid obvious banking terms even if slipped
    lower_all = (q.get("question","") + " " + " ".join(opts) + " " + exp).lower()
    if any(k in lower_all for k in ["repo", "crr", "slr", "mclr", "upi", "sebi", "irdai", "pfrda", "nbfc", "ipo"]):
        return False

    return True


def fix_mapping_and_keys(mcq_set: dict) -> dict:
    fixed = {"date": mcq_set["date"], "mcqs": []}
    seen_event_keys = set()

    for q in mcq_set["mcqs"]:
        # Ensure distinct options
        opts = [o.strip() for o in q["options"]]
        uniq, seen = [], set()
        for o in opts:
            oo = o if o else "—"
            if oo in seen:
                oo = oo + " "
            seen.add(oo)
            uniq.append(oo)
        q["options"] = uniq[:4]

        ca = (q.get("correct_answer") or "").strip()
        if ca in q["options"]:
            q["correct_option_id"] = q["options"].index(ca)
        else:
            idx = q.get("correct_option_id", 0)
            idx = idx if isinstance(idx, int) and 0 <= idx <= 3 else 0
            q["correct_answer"] = q["options"][idx]
            q["correct_option_id"] = idx

        ek = (q.get("event_key") or "").strip()
        ek_n = _norm_text(ek)
        if not ek_n or ek_n in seen_event_keys:
            ek = (ek[:45] + " " + str(len(seen_event_keys) + 1)).strip()
            q["event_key"] = ek[:60]
        seen_event_keys.add(_norm_text(q["event_key"]))

        q["explanation"] = (q.get("explanation") or "")[:220]
        q["source_hint"] = (q.get("source_hint") or "")[:80]

        fixed["mcqs"].append(q)

    fixed["mcqs"] = fixed["mcqs"][:10]
    return fixed


# -------------------- GENERATE MCQS (PIPELINE) --------------------
def generate_mcqs():
    now_ist = dt.datetime.now(IST)
    today_str = now_ist.date().isoformat()

    hist = load_history()
    avoid_qs = (hist.get("questions", []) or [])[-250:]
    avoid_event_keys = (hist.get("event_keys", []) or [])[-250:]
    avoid_title_norms = set(hist.get("title_norms", []) or [])
    avoid_url_hashes = set(hist.get("url_hashes", []) or [])

    raw = fetch_news_pool()
    fresh = filter_news_today_ist(raw, avoid_title_norms, avoid_url_hashes)

    # Exclude banking/finance, allow economy
    fresh = [x for x in fresh if not is_banking_finance_item(x)]

    # UPSC-worthiness filter (removes shallow/low-quality)
    fresh = [x for x in fresh if is_upsc_worthy(x)]

    if len(fresh) < 16:
        raise RuntimeError(
            f"Not enough UPSC-worthy fresh items after filters. Found {len(fresh)}. "
            f"Try lowering MIN_TODAY_ITEMS_REQUIRED to 10."
        )

    ranked = rank_news(fresh)[:RERANK_TOP_N]
    news_context = build_news_context(ranked, max_items=CONTEXT_ITEMS)

    best = None
    best_good = -1

    # Try a few times to enforce UPSC quality
    for attempt in range(4):
        mcq_set = generate_mcqs_upsc(today_str, news_context, avoid_qs, avoid_event_keys)
        mcq_set["date"] = today_str
        mcq_set = fix_mapping_and_keys(mcq_set)

        # Dedupe within set by normalized question
        deduped = []
        seen_q = set()
        for q in mcq_set["mcqs"]:
            k = _norm_q(q["question"])
            if k not in seen_q:
                seen_q.add(k)
                deduped.append(q)
        mcq_set["mcqs"] = deduped[:10]

        good = [q for q in mcq_set["mcqs"] if looks_upsc_style(q)]
        if len(good) >= 10:
            mcq_set["mcqs"] = good[:10]
            best = mcq_set
            best_good = 10
            break

        if len(good) > best_good:
            best = mcq_set
            best_good = len(good)

        # strengthen avoidance with this attempt's questions
        avoid_qs = avoid_qs + [q["question"] for q in mcq_set["mcqs"]]

    if best is None or len(best["mcqs"]) < 10:
        raise RuntimeError("Could not generate 10 UPSC-quality MCQs. Retry once.")

    best["mcqs"] = best["mcqs"][:10]

    used_title_norms = {_norm_text(it["title"]) for it in ranked[:CONTEXT_ITEMS]}
    used_url_hashes = {_hash_url(it["link"]) for it in ranked[:CONTEXT_ITEMS]}
    return best, used_title_norms, used_url_hashes


# -------------------- QUIZ POSTING --------------------
def shuffle_options_and_fix_answer(q: dict) -> dict:
    """
    Shuffles options so correct answer isn't always A.
    Updates correct_option_id accordingly.
    """
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
    # Plain text (no Markdown) to avoid parsing failures
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
            "text": f"🧠 Daily UPSC-Style Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇",
        },
    )

    for i, q in enumerate(mcq_set["mcqs"], start=1):
        q = shuffle_options_and_fix_answer(q)

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": f"Q{i}. {q['question']}",
            "options": q["options"],  # Telegram expects list[str]
            "type": "quiz",
            "correct_option_id": q["correct_option_id"],
            "explanation": q["explanation"],
            "is_anonymous": True,
        }
        tg("sendPoll", payload)
        time.sleep(SLEEP_BETWEEN_POLLS)

    # After all 10 polls: message first, then score poll
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
    mcq_set, used_title_norms, used_url_hashes = generate_mcqs()
    post_to_channel(mcq_set)

    # Save history AFTER successful post
    hist = load_history()
    hist = update_history_with_set(
        hist,
        mcq_set,
        keep_last_days=KEEP_LAST_DAYS,
        used_title_norms=used_title_norms,
        used_url_hashes=used_url_hashes,
    )
    save_history(hist)

    print("✅ Posted 10 UPSC-style quiz polls + message + score poll successfully.")


if __name__ == "__main__":
    main()
