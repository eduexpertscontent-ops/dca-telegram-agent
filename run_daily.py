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

KEEP_LAST_DAYS = int(os.getenv("KEEP_LAST_DAYS", "15"))
SLEEP_BETWEEN_POLLS = float(os.getenv("SLEEP_BETWEEN_POLLS", "2"))
SLEEP_AFTER_QUIZ = float(os.getenv("SLEEP_AFTER_QUIZ", "2"))

NEWS_TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "25"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "250"))
CONTEXT_ITEMS = int(os.getenv("CONTEXT_ITEMS", "70"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "120"))
MIN_TODAY_ITEMS_REQUIRED = int(os.getenv("MIN_TODAY_ITEMS_REQUIRED", "18"))

HISTORY_FILE = "mcq_history.json"

client = OpenAI()
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


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


# -------------------- SCHEMA (adds mcq_type + difficulty + source_hint, not posted) --------------------
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
                    "difficulty": {"type": "string", "enum": ["Easy", "Moderate"]},
                    "question": {"type": "string", "minLength": 1, "maxLength": 300},
                    "options": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 100},
                    },
                    "correct_option_id": {"type": "integer", "minimum": 0, "maximum": 3},
                    "correct_answer": {"type": "string", "minLength": 1, "maxLength": 100},
                    "explanation": {"type": "string", "maxLength": 200},
                    "source_hint": {"type": "string", "maxLength": 120},
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


# -------------------- HISTORY --------------------
def load_history() -> dict:
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

    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > keep_last_days:
        for d in all_dates[:-keep_last_days]:
            hist["dates"].pop(d, None)

    qs, eks, tns, uhs = [], [], [], []
    for d in sorted(hist["dates"].keys()):
        qs.extend(hist["dates"][d].get("questions", []))
        eks.extend(hist["dates"][d].get("event_keys", []))
        tns.extend(hist["dates"][d].get("title_norms", []))
        uhs.extend(hist["dates"][d].get("url_hashes", []))

    hist["questions"] = qs[-600:]
    hist["event_keys"] = eks[-600:]
    hist["title_norms"] = tns[-1000:]
    hist["url_hashes"] = uhs[-1000:]
    return hist


# -------------------- NORMALIZATION --------------------
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
                "description": re.sub(r"<.*?>", "", desc)[:240],
            }
        )
    return items


# -------------------- AUTHENTIC SOURCES ONLY --------------------
def fetch_news_pool():
    """
    ONLY authentic current-affairs / official sources.
    NOTE: Some sites may not expose RSS reliably; we fail gracefully.
    """
    feeds = [
        # Official/primary
        ("PIB", "https://pib.gov.in/RssMain.aspx?mod=0&ln=1"),

        # Well-known current affairs sites (RSS/feeds)
        ("AffairsCloud", "https://affairscloud.com/feed/"),
        ("GKToday", "https://www.gktoday.in/feed/"),

        # Many candidates use these; if a feed fails, it’s okay (no crash)
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


def filter_news_today_ist(items, avoid_title_norms, avoid_url_hashes):
    now_ist = dt.datetime.now(IST)
    today = now_ist.date()

    def is_today(it):
        return it.get("published_ist") and it["published_ist"].date() == today

    today_items = [it for it in items if is_today(it)]

    # If too few today items across CA sources, allow last 24h (still fresh)
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


# -------------------- EXAM RELEVANCY SCORING (POSITIVE-ONLY) --------------------
# No LOW_VALUE list (as per your instruction). Only positive scoring.
POSITIVE_KEYWORDS = [
    # Governance/Polity/Schemes
    "cabinet", "parliament", "bill", "ordinance", "notification", "guidelines", "policy",
    "scheme", "yojana", "mission", "programme", "initiative",
    "commission", "authority", "tribunal", "supreme court", "high court",

    # Economy/Banking/Regulation
    "rbi", "sebi", "irdai", "pfrda", "nabard", "sidbi",
    "bank", "inflation", "gdp", "budget", "gst", "repo", "crr", "slr", "npa", "upi", "forex",

    # Reports/Indexes/Surveys
    "report", "index", "ranking", "survey", "census",

    # Defence/S&T/Environment
    "drdo", "missile", "exercise", "navy", "air force", "army",
    "isro", "satellite", "space", "ai", "quantum", "semiconductor",
    "wildlife", "tiger reserve", "national park", "pollution", "climate",

    # International orgs/summits (only when exam-relevant)
    "summit", "cop", "g20", "brics", "sco", "who", "un", "wto", "imf", "world bank",

    # Appointments/awards (exam-asked)
    "appointed", "elected", "chairman", "governor", "ceo", "chief",
    "award",
]

def exam_relevancy_score(item: dict) -> int:
    title = _norm_text(item.get("title", ""))
    desc = _norm_text(item.get("description", ""))
    text = f"{title} {desc}"

    score = 0

    # Source weighting (official > CA site > general)
    src = (item.get("source") or "").lower()
    if "pib" in src:
        score += 10
    if "affairscloud" in src:
        score += 6
    if "gktoday" in src:
        score += 6
    if "jagran" in src:
        score += 4
    if "adda" in src:
        score += 4

    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            score += 2

    # Slight boost for India context words (without using any blacklist)
    if "india" in text or "indian" in text:
        score += 2

    # Prefer informative titles
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


# -------------------- OPENAI: EXAM-STANDARD FRAMING --------------------
EXAM_STYLE_RULES = """
You must frame questions exactly like SSC/Bank/PCS exams.

Allowed MCQ TYPES (any mix; no fixed quotas):
- Direct Fact (who/what/where/when)
- Statement-based: "Consider the following statements..." (2 statements) ask which are correct
- Pairing: "Which of the following pairs is/are correctly matched?"
- Simplified Match (List-I / List-II) as 4 options

Exam framing rules (hard):
1) Start with: "Recently," OR "In the context of current affairs," OR "Which of the following..." OR "Consider the following statements:"
2) No casual quiz tone, no emojis.
3) Options must be same category (all persons / all organisations / all places / all schemes / all dates).
4) Avoid vague stems like "According to a report" unless you name the report (e.g., NFHS, ASER, WEO, etc.).
5) No opinion questions.
6) Difficulty only Easy/Moderate.
7) Explanation must have two parts within 200 chars: (a) current fact, (b) static GK link (definition/role/body).
8) Use ONLY the given news items. Do not invent.
"""

def generate_mcqs_exam_style(today_str: str, news_context: str, avoid_questions: list, avoid_event_keys: list):
    system = (
        "You are an exam question setter for SSC, Banking exams, and State PCS.\n"
        "Use ONLY the provided news list.\n"
        + EXAM_STYLE_RULES +
        "\nReturn STRICT JSON as per schema."
    )

    avoid_block = "\n".join([f"- {a}" for a in (avoid_questions or [])[:180]])
    avoid_keys_block = "\n".join([f"- {a}" for a in (avoid_event_keys or [])[:180]])

    user = (
        f"DATE (IST): {today_str}\n\n"
        "NEWS ITEMS (use ONLY these):\n"
        f"{news_context}\n\n"
        "Create EXACTLY 10 MCQs from the MOST exam-relevant items.\n"
        "- Each MCQ must be from a DIFFERENT news story.\n"
        "- Do NOT follow any fixed quotas by category.\n\n"
        "AVOID repeating ANY of these prior questions/topics:\n"
        f"{avoid_block}\n\n"
        "AVOID these prior event_keys/topics:\n"
        f"{avoid_keys_block}\n\n"
        "For each MCQ, include:\n"
        "- mcq_type (one allowed type)\n"
        "- difficulty (Easy/Moderate)\n"
        "- source_hint: just the source label like 'PIB' or 'GKToday' (no URL)\n"
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text={"format": {"type": "json_schema", "name": "daily_ca_mcqs_examstyle", "strict": True, "schema": SCHEMA}},
    )
    return json.loads(resp.output_text)


# -------------------- QUALITY FILTERS --------------------
def looks_exam_style(q: dict) -> bool:
    text = (q.get("question") or "").strip().lower()
    starters = ("recently", "in the context", "which of the following", "consider the following statements")
    if not text.startswith(starters):
        return False

    opts = q.get("options") or []
    if len(opts) != 4:
        return False
    if len(set(o.strip() for o in opts)) != 4:
        return False

    if q.get("difficulty") not in ("Easy", "Moderate"):
        return False

    exp = (q.get("explanation") or "").strip()
    if len(exp) < 45:
        return False

    return True


def fix_mapping_and_keys(mcq_set: dict) -> dict:
    fixed = {"date": mcq_set["date"], "mcqs": []}
    seen_event_keys = set()

    for q in mcq_set["mcqs"]:
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

        q["explanation"] = (q.get("explanation") or "")[:200]
        q["source_hint"] = (q.get("source_hint") or "")[:120]

        fixed["mcqs"].append(q)

    fixed["mcqs"] = fixed["mcqs"][:10]
    return fixed


def generate_mcqs():
    now_ist = dt.datetime.now(IST)
    today_str = now_ist.date().isoformat()

    hist = load_history()
    avoid_qs = (hist.get("questions", []) or [])[-220:]
    avoid_event_keys = (hist.get("event_keys", []) or [])[-220:]
    avoid_title_norms = set(hist.get("title_norms", []) or [])
    avoid_url_hashes = set(hist.get("url_hashes", []) or [])

    raw = fetch_news_pool()
    fresh = filter_news_today_ist(raw, avoid_title_norms, avoid_url_hashes)

    if len(fresh) < 15:
        raise RuntimeError(f"Not enough fresh items from authentic CA sources. Found {len(fresh)}.")

    ranked = rank_news(fresh)[:RERANK_TOP_N]
    news_context = build_news_context(ranked, max_items=CONTEXT_ITEMS)

    best = None
    best_good = 0

    for attempt in range(4):
        mcq_set = generate_mcqs_exam_style(today_str, news_context, avoid_qs, avoid_event_keys)
        mcq_set["date"] = today_str
        mcq_set = fix_mapping_and_keys(mcq_set)

        good = [q for q in mcq_set["mcqs"] if looks_exam_style(q)]
        if len(good) >= 10:
            mcq_set["mcqs"] = good[:10]
            best = mcq_set
            best_good = 10
            break

        if len(good) > best_good:
            best = mcq_set
            best_good = len(good)

        avoid_qs = avoid_qs + [q["question"] for q in mcq_set["mcqs"]]

    if best is None or len(best["mcqs"]) < 10:
        raise RuntimeError("Could not generate exam-standard MCQs. Retry once.")

    best["mcqs"] = best["mcqs"][:10]

    used_title_norms = {_norm_text(it["title"]) for it in ranked[:CONTEXT_ITEMS]}
    used_url_hashes = {_hash_url(it["link"]) for it in ranked[:CONTEXT_ITEMS]}
    return best, used_title_norms, used_url_hashes


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
    mcq_set, used_title_norms, used_url_hashes = generate_mcqs()
    post_to_channel(mcq_set)

    hist = load_history()
    hist = update_history_with_set(
        hist,
        mcq_set,
        keep_last_days=KEEP_LAST_DAYS,
        used_title_norms=used_title_norms,
        used_url_hashes=used_url_hashes,
    )
    save_history(hist)

    print("✅ Posted 10 exam-style quiz polls + message + score poll successfully.")


if __name__ == "__main__":
    main()
