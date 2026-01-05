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

NEWS_TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "20"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "180"))          # raw pool size
CONTEXT_ITEMS = int(os.getenv("CONTEXT_ITEMS", "70"))             # items sent to model
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "80"))               # top after exam scoring
MIN_TODAY_ITEMS_REQUIRED = int(os.getenv("MIN_TODAY_ITEMS_REQUIRED", "25"))  # else allow last 24h

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
                    "event_key": {"type": "string", "minLength": 3, "maxLength": 60},
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
                },
                "required": [
                    "event_key",
                    "question",
                    "options",
                    "correct_option_id",
                    "correct_answer",
                    "explanation",
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

    hist["questions"] = qs[-450:]
    hist["event_keys"] = eks[-450:]
    hist["title_norms"] = tns[-700:]
    hist["url_hashes"] = uhs[-700:]
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


# -------------------- RSS FETCH --------------------
def _get(url: str) -> str:
    r = requests.get(url, timeout=NEWS_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


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

        pub_dt = None
        if pub:
            try:
                pub_dt = parsedate_to_datetime(pub)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=dt.timezone.utc)
                pub_dt = pub_dt.astimezone(IST)
            except Exception:
                pub_dt = None

        items.append(
            {
                "title": title,
                "link": link,
                "published_ist": pub_dt,
                "description": re.sub(r"<.*?>", "", desc)[:220],
            }
        )
    return items


def fetch_news_pool():
    """
    India + exam-relevant pool:
    - Google News India + topical sections
    - Google News search RSS for govt/schemes/RBI/SEBI etc.
    - PIB RSS
    - AffairsCloud
    """
    feeds = []

    feeds += [
        ("GoogleNews-India", "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"),
        ("GoogleNews-Business", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en"),
        ("GoogleNews-Science", "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-IN&gl=IN&ceid=IN:en"),
        ("GoogleNews-Tech", "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en"),
        ("GoogleNews-Sports", "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en"),
        ("GoogleNews-World", "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en"),
    ]

    search_queries = [
        ("Govt-Policy", "cabinet OR parliament OR bill OR ordinance OR notification India"),
        ("Schemes", "scheme OR yojana OR mission OR initiative India"),
        ("Appointments", "appointed OR appointment chairman CEO governor India"),
        ("Reports-Index", "report OR survey OR index OR ranking India"),
        ("RBI", "RBI circular OR RBI notification OR monetary policy OR bank regulation"),
        ("SEBI", "SEBI circular OR SEBI guidelines OR capital markets India"),
        ("Defence", "DRDO OR Indian Navy OR Indian Air Force OR Indian Army OR defence exercise"),
        ("Environment", "tiger reserve OR national park OR pollution OR climate OR environment India"),
        ("Summits", "summit OR G20 OR BRICS OR SCO OR ASEAN OR COP"),
    ]

    for label, q in search_queries:
        q_enc = requests.utils.quote(q)
        feeds.append(
            (f"GoogleSearch-{label}",
             f"https://news.google.com/rss/search?q={q_enc}&hl=en-IN&gl=IN&ceid=IN:en")
        )

    feeds.append(("PIB", "https://pib.gov.in/RssMain.aspx?mod=0&ln=1"))
    feeds.append(("AffairsCloud", "https://affairscloud.com/feed/"))

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

    # If too few today, allow last 24h (still fresh)
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


# -------------------- EXAM RELEVANCY SCORING (NO FIXED SECTIONS) --------------------
HIGH_VALUE = [
    # Govt/Polity/Administration
    "cabinet", "parliament", "lok sabha", "rajya sabha", "bill", "ordinance", "notification", "gazette",
    "ministry", "department", "commission", "authority", "supreme court", "high court",
    # Schemes/Initiatives
    "scheme", "yojana", "mission", "initiative", "programme", "policy", "guidelines",
    # Economy/Banking
    "rbi", "sebi", "irdai", "pfrda", "nabard", "sidbi", "bank", "inflation", "gdp", "fiscal", "budget",
    "gst", "repo", "crr", "slr", "mclr", "npa", "upi", "digital payments", "forex",
    # Defence
    "drdo", "missile", "warship", "frigate", "submarine", "exercise", "navy", "air force", "army",
    # Environment/Science
    "national park", "tiger reserve", "biosphere", "wildlife", "climate", "pollution",
    "isro", "satellite", "space", "quantum", "semiconductor", "ai", "vaccine",
    # Awards/Reports/Indexes
    "award", "rank", "ranking", "index", "report", "survey",
    # Summits/International orgs (exam relevant)
    "summit", "cop", "g20", "brics", "sco", "who", "un", "wto", "imf", "world bank",
    # Appointments/Obituary (only if big)
    "appointed", "elected", "chairman", "governor", "chief", "ceo", "president", "prime minister",
]

LOW_VALUE = [
    "celebrity", "movie", "film", "box office", "dating", "rumour", "gossip", "instagram",
    "fashion", "bollywood", "hollywood", "cricket wife", "trolled", "viral", "meme",
]


def exam_relevancy_score(item: dict) -> int:
    """
    Higher score = more likely asked in SSC/Bank/PCS.
    No fixed section allocation; this just ranks.
    """
    title = _norm_text(item.get("title", ""))
    desc = _norm_text(item.get("description", ""))
    text = f"{title} {desc}"

    score = 0

    # source boosts
    src = (item.get("source") or "").lower()
    if "pib" in src:
        score += 8
    if "affairscloud" in src:
        score += 5
    if "googlesearch" in src:
        score += 2

    # keyword boosts
    for kw in HIGH_VALUE:
        if kw in text:
            score += 2

    # penalties for low value
    for kw in LOW_VALUE:
        if kw in text:
            score -= 6

    # India relevance boost (soft)
    if "india" in text or "indian" in text:
        score += 2

    # Very short/low info penalty
    if len(title) < 30:
        score -= 1

    return score


def rank_news_by_exam_relevance(items):
    scored = []
    for it in items:
        it = dict(it)
        it["exam_score"] = exam_relevancy_score(it)
        scored.append(it)

    scored.sort(key=lambda x: (x.get("exam_score", 0), x.get("published_ist") or dt.datetime.min.replace(tzinfo=IST)), reverse=True)
    return scored


def _build_news_context(news_items, max_items):
    picked = news_items[:max_items]
    lines = []
    for i, it in enumerate(picked, start=1):
        pub = it.get("published_ist")
        pub_s = pub.strftime("%Y-%m-%d %H:%M IST") if pub else "unknown time"
        lines.append(
            f"{i}. [score={it.get('exam_score',0)}] [{it['source']}] {pub_s} | {it['title']} | {it['link']} | {it.get('description','')}"
        )
    return "\n".join(lines)


# -------------------- OPENAI MCQ GENERATION (NO FIXED COUNTS, EXAM RELEVANCY FIRST) --------------------
def generate_mcqs_from_ranked_news(today_str: str, news_context: str, avoid_questions: list, avoid_event_keys: list):
    system = (
        "You create DAILY current affairs MCQs for SSC, Banking, and State PCS exams in ENGLISH.\n"
        "CRITICAL CONSTRAINTS:\n"
        "1) Use ONLY the news items provided by the user. Do NOT invent events.\n"
        "2) Freshness: Use today's IST items (or last-24h fallback already included). Do not use old years.\n"
        "3) Exam relevancy first: prefer govt/policy/schemes/reports/indexes/RBI/SEBI/banking/defence/environment/science/major awards/major sports results.\n"
        "4) Avoid low-value entertainment/celebrity gossip.\n"
        "5) Diversity (soft): Do not pick multiple MCQs from the same story/event. Each MCQ should come from a different item.\n\n"
        "OUTPUT RULES:\n"
        "- EXACTLY 10 MCQs.\n"
        "- Each MCQ must have a UNIQUE event_key (3–8 words).\n"
        "- correct_answer must EXACTLY match one of the 4 options.\n"
        "- correct_option_id must be the index of correct_answer in options.\n"
        "- All 4 options must be distinct.\n"
        "- explanation <= 200 characters.\n"
        "- Keep questions direct and factual (SSC/Bank/PCS style).\n"
    )

    avoid_block = "\n".join([f"- {a}" for a in (avoid_questions or [])[:160]])
    avoid_keys_block = "\n".join([f"- {a}" for a in (avoid_event_keys or [])[:160]])

    user = (
        f"DATE (IST): {today_str}\n\n"
        "NEWS ITEMS (use ONLY these):\n"
        f"{news_context}\n\n"
        "TASK:\n"
        "- Create EXACTLY 10 MCQs based on the MOST EXAM-RELEVANT items.\n"
        "- Do NOT follow any fixed quotas by category. Choose what is most likely for SSC/Bank/PCS.\n"
        "- Ensure all 10 are from different news stories.\n\n"
        "AVOID repeating ANY of these prior questions/topics:\n"
        f"{avoid_block}\n\n"
        "AVOID these prior event_keys/topics:\n"
        f"{avoid_keys_block}\n\n"
        "Return JSON exactly as per schema."
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "daily_ca_mcqs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    )
    return json.loads(resp.output_text)


def generate_mcqs():
    now_ist = dt.datetime.now(IST)
    today_str = now_ist.date().isoformat()

    hist = load_history()
    avoid_qs = (hist.get("questions", []) or [])[-180:]
    avoid_event_keys = (hist.get("event_keys", []) or [])[-180:]
    avoid_title_norms = set(hist.get("title_norms", []) or [])
    avoid_url_hashes = set(hist.get("url_hashes", []) or [])

    raw_news = fetch_news_pool()
    filtered_news = filter_news_today_ist(raw_news, avoid_title_norms, avoid_url_hashes)

    if len(filtered_news) < 18:
        raise RuntimeError(f"Not enough fresh news items for {today_str} (IST). Found {len(filtered_news)}.")

    ranked = rank_news_by_exam_relevance(filtered_news)

    # Take the top N by exam score
    ranked_top = ranked[:RERANK_TOP_N]

    # Build context for model (ranked list)
    news_context = _build_news_context(ranked_top, max_items=CONTEXT_ITEMS)

    mcq_set = generate_mcqs_from_ranked_news(
        today_str=today_str,
        news_context=news_context,
        avoid_questions=avoid_qs,
        avoid_event_keys=avoid_event_keys,
    )

    # Internal uniqueness guard
    def qn(q): return _norm_text(q.get("question", ""))
    def ekn(q): return _norm_text(q.get("event_key", ""))

    seen_q, seen_ek = set(), set()
    unique = []
    for q in mcq_set["mcqs"]:
        if qn(q) and qn(q) not in seen_q and ekn(q) and ekn(q) not in seen_ek:
            seen_q.add(qn(q))
            seen_ek.add(ekn(q))
            unique.append(q)

    if len(unique) < 10:
        raise RuntimeError("Duplicate MCQs detected. Increase CONTEXT_ITEMS or MAX_NEWS_ITEMS and retry.")

    mcq_set["date"] = today_str
    mcq_set["mcqs"] = unique[:10]

    used_title_norms = {_norm_text(it["title"]) for it in ranked_top[:CONTEXT_ITEMS]}
    used_url_hashes = {_hash_url(it["link"]) for it in ranked_top[:CONTEXT_ITEMS]}
    return mcq_set, used_title_norms, used_url_hashes


def validate_and_fix_mcqs(mcq_set: dict) -> dict:
    """
    Keep it safe: only fix option mapping/format. Do NOT browse extra sources here.
    """
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

        fixed["mcqs"].append(q)

    fixed["mcqs"] = fixed["mcqs"][:10]
    return fixed


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
    mcq_set = validate_and_fix_mcqs(mcq_set)

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

    print("✅ Posted 10 quiz polls + message + score poll successfully.")


if __name__ == "__main__":
    main()
