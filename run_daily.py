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

# News fetch settings
NEWS_TIMEOUT = int(os.getenv("NEWS_TIMEOUT", "20"))
MAX_NEWS_ITEMS = int(os.getenv("MAX_NEWS_ITEMS", "80"))  # total pool size before filtering
MIN_TODAY_ITEMS_REQUIRED = int(os.getenv("MIN_TODAY_ITEMS_REQUIRED", "18"))  # if < this, allow last 24h fallback

HISTORY_FILE = "mcq_history.json"

client = OpenAI()  # reads OPENAI_API_KEY from environment

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


# -------------------- TELEGRAM HELPERS --------------------
def tg(method: str, payload: dict) -> dict:
    """Telegram API wrapper with basic retry on rate limits."""
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


# -------------------- JSON SCHEMA (UNCHANGED) --------------------
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


# -------------------- HISTORY (NO-REPEAT ACROSS DAYS) --------------------
def load_history() -> dict:
    """
    {
      "dates": {
        "YYYY-MM-DD": {
          "questions":[...],
          "event_keys":[...],
          "title_norms":[...],
          "url_hashes":[...]
        }
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

    hist["questions"] = qs[-300:]
    hist["event_keys"] = eks[-300:]
    hist["title_norms"] = tns[-400:]
    hist["url_hashes"] = uhs[-400:]
    return hist


# -------------------- NORMALIZATION / DEDUPE --------------------
def _norm_text(s: str) -> str:
    s = s or ""
    s = s.lower().strip()
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


# -------------------- NEWS FETCH (Google News RSS + AffairsCloud RSS) --------------------
def _get(url: str) -> str:
    r = requests.get(
        url,
        timeout=NEWS_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    return r.text


def _parse_rss_items(xml_text: str):
    # minimal RSS parser (works for most feeds)
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        # Atom fallback (rare)
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
    Returns list of dicts with:
    {title, link, published_ist, description, source}
    """
    feeds = []

    # Google News RSS (India edition) - multiple topical queries for variety
    # NOTE: This is still "Google News", not a paid API.
    google_queries = [
        ("India", "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"),
        ("World", "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en"),
        ("Business", "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en"),
        ("Science", "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-IN&gl=IN&ceid=IN:en"),
        ("Technology", "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en"),
        ("Sports", "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en"),
    ]
    for label, url in google_queries:
        feeds.append(("GoogleNews-" + label, url))

    # AffairsCloud feed (current affairs site)
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

    # Basic cleanup
    all_items = [x for x in all_items if x.get("title") and x.get("link")]
    # Limit raw pool
    all_items = all_items[:MAX_NEWS_ITEMS]

    # Deduplicate by normalized title + url hash
    all_items = _dedupe_keep_order_by_key(all_items, lambda x: (_norm_text(x["title"]), _hash_url(x["link"])))
    return all_items


def filter_news_today_ist(items, avoid_title_norms, avoid_url_hashes):
    now_ist = dt.datetime.now(IST)
    today = now_ist.date()

    def is_today(it):
        return it.get("published_ist") and it["published_ist"].date() == today

    today_items = [it for it in items if is_today(it)]
    # If too few today, allow last 24h (but still recent)
    if len(today_items) < MIN_TODAY_ITEMS_REQUIRED:
        cutoff = now_ist - dt.timedelta(hours=24)
        recent_items = [it for it in items if it.get("published_ist") and it["published_ist"] >= cutoff]
        pool = recent_items
    else:
        pool = today_items

    # Apply history avoid
    out = []
    for it in pool:
        tn = _norm_text(it["title"])
        uh = _hash_url(it["link"])
        if tn in avoid_title_norms:
            continue
        if uh in avoid_url_hashes:
            continue
        out.append(it)

    # final dedupe by title norm
    out = _dedupe_keep_order_by_key(out, lambda x: _norm_text(x["title"]))
    return out


# -------------------- OPENAI GENERATION (GROUNDED ON PROVIDED NEWS ONLY) --------------------
def _build_news_context(news_items, max_items=35):
    """
    Build compact bullet list for LLM.
    Only top N items used to keep prompt small.
    """
    picked = news_items[:max_items]
    lines = []
    for i, it in enumerate(picked, start=1):
        pub = it.get("published_ist")
        pub_s = pub.strftime("%Y-%m-%d %H:%M IST") if pub else "unknown time"
        lines.append(
            f"{i}. [{it['source']}] {pub_s} | {it['title']} | {it['link']} | {it.get('description','')}"
        )
    return "\n".join(lines)


def _generate_n_mcqs_from_news(n: int, today_str: str, news_context: str, avoid_questions: list, avoid_event_keys: list):
    system = (
        "You create DAILY current affairs MCQs for competitive exams in ENGLISH.\n"
        "CRITICAL CONSTRAINTS:\n"
        "1) Use ONLY the NEWS ITEMS provided in the user message. Do NOT use memory. Do NOT invent events.\n"
        "2) Freshness: Questions MUST be based on events published today (IST) in the provided list.\n"
        "   If the list includes last-24h fallback items, still use ONLY those items.\n"
        "3) Diversity: Each MCQ must use a DIFFERENT news item (no 2 MCQs from same event/story).\n"
        "ABSOLUTE RULES:\n"
        "- Each MCQ must have a UNIQUE event_key (3–8 words).\n"
        "- correct_answer must EXACTLY match one of the 4 options.\n"
        "- correct_option_id must be the index of correct_answer in options.\n"
        "- All 4 options must be distinct.\n"
        "- explanation must be <= 200 characters.\n"
        "- Avoid repeating questions/topics/event_keys from the avoid lists.\n"
    )

    avoid_block = "\n".join([f"- {a}" for a in (avoid_questions or [])[:120]])
    avoid_keys_block = "\n".join([f"- {a}" for a in (avoid_event_keys or [])[:120]])

    user = (
        f"DATE (IST): {today_str}\n\n"
        "NEWS ITEMS (use ONLY these):\n"
        f"{news_context}\n\n"
        f"Create {n} UNIQUE MCQs from DIFFERENT news items.\n"
        "Mix: India, World, Economy/Banking, Science/Tech, Defence, Environment, Sports, Awards.\n\n"
        "AVOID repeating ANY of these prior questions/topics:\n"
        f"{avoid_block}\n\n"
        "AVOID these prior event_keys/topics:\n"
        f"{avoid_keys_block}\n\n"
        "Return JSON exactly as per schema."
    )

    schema = _schema_for_n(n)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "daily_ca_mcqs_n",
                "strict": True,
                "schema": schema,
            }
        },
    )
    return json.loads(resp.output_text)


def generate_mcqs():
    now_ist = dt.datetime.now(IST)
    today_str = now_ist.date().isoformat()

    hist = load_history()
    avoid_qs = (hist.get("questions", []) or [])[-140:]
    avoid_event_keys = (hist.get("event_keys", []) or [])[-140:]
    avoid_title_norms = set(hist.get("title_norms", []) or [])
    avoid_url_hashes = set(hist.get("url_hashes", []) or [])

    raw_news = fetch_news_pool()
    filtered_news = filter_news_today_ist(raw_news, avoid_title_norms, avoid_url_hashes)

    if len(filtered_news) < 12:
        # If not enough fresh news, stop instead of producing stale 2025 questions.
        # (Better to fail fast than post wrong-date MCQs.)
        raise RuntimeError(
            f"Not enough fresh news items for {today_str} (IST). Found only {len(filtered_news)} after filtering."
        )

    # Keep track of what we used (for history)
    # We'll choose a larger pool for the model and then later "map" usage by title/url norms
    news_context = _build_news_context(filtered_news, max_items=35)

    base = _generate_n_mcqs_from_news(
        10,
        today_str=today_str,
        news_context=news_context,
        avoid_questions=avoid_qs,
        avoid_event_keys=avoid_event_keys,
    )

    # Lightweight internal dedupe (question text + event_key)
    def q_norm(q): return _norm_text(q.get("question", ""))
    def ek_norm(q): return _norm_text(q.get("event_key", ""))

    unique = []
    seen_q = set()
    seen_ek = set()
    for q in base["mcqs"]:
        nq = q_norm(q)
        nek = ek_norm(q)
        if nq and nq not in seen_q and nek and nek not in seen_ek:
            seen_q.add(nq)
            seen_ek.add(nek)
            unique.append(q)

    tries = 0
    while len(unique) < 10 and tries < 3:
        missing = 10 - len(unique)
        extra = _generate_n_mcqs_from_news(
            missing,
            today_str=today_str,
            news_context=news_context,
            avoid_questions=avoid_qs + [q["question"] for q in unique],
            avoid_event_keys=avoid_event_keys + [q["event_key"] for q in unique],
        )
        for q in extra["mcqs"]:
            nq = q_norm(q)
            nek = ek_norm(q)
            if nq and nq not in seen_q and nek and nek not in seen_ek:
                seen_q.add(nq)
                seen_ek.add(nek)
                unique.append(q)
        tries += 1

    if len(unique) < 10:
        raise RuntimeError("Could not generate 10 unique MCQs without repeats. Try increasing MAX_NEWS_ITEMS.")

    base["date"] = today_str
    base["mcqs"] = unique[:10]

    # Determine used title/url norms heuristically (best-effort):
    # We mark top of filtered_news as used because model is constrained to that list.
    # To be safer, store the entire context pool as "used" for today so tomorrow it avoids same stories.
    used_title_norms = {_norm_text(it["title"]) for it in filtered_news[:35]}
    used_url_hashes = {_hash_url(it["link"]) for it in filtered_news[:35]}

    return base, used_title_norms, used_url_hashes


def validate_and_fix_mcqs(mcq_set: dict) -> dict:
    """
    IMPORTANT: We do NOT re-browse older web.
    We only fix internal consistency: options/correct answer mapping, duplicates, etc.
    (This prevents the model from pulling old 2025 background and contaminating freshness.)
    """
    fixed = {"date": mcq_set["date"], "mcqs": []}
    seen_event_keys = set()

    for q in mcq_set["mcqs"]:
        # Ensure distinct options
        opts = q["options"]
        # if duplicates, make them unique by slight edit (rare)
        seen_opt = set()
        new_opts = []
        for o in opts:
            oo = o.strip()
            if oo in seen_opt:
                oo = oo + " "
            seen_opt.add(oo)
            new_opts.append(oo)
        q["options"] = new_opts

        # Ensure correct_answer matches one of options
        ca = q["correct_answer"].strip()
        if ca not in q["options"]:
            # fallback: use correct_option_id to set correct_answer
            idx = q.get("correct_option_id", 0)
            idx = idx if isinstance(idx, int) and 0 <= idx <= 3 else 0
            q["correct_answer"] = q["options"][idx]
        else:
            q["correct_option_id"] = q["options"].index(ca)

        # Ensure unique event_key
        ek = q["event_key"].strip()
        ek_n = _norm_text(ek)
        if ek_n in seen_event_keys or not ek_n:
            ek = (ek[:45] + " " + str(len(seen_event_keys) + 1)).strip()
            q["event_key"] = ek[:60]
        seen_event_keys.add(_norm_text(q["event_key"]))

        # Cap explanation length
        q["explanation"] = (q.get("explanation") or "")[:200]

        fixed["mcqs"].append(q)

    # Ensure exactly 10
    fixed["mcqs"] = fixed["mcqs"][:10]
    return fixed


# -------------------- QUIZ POSTING --------------------
def shuffle_options_and_fix_answer(q: dict) -> dict:
    """Shuffles options so correct answer isn't always A; updates correct_option_id."""
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
        {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
    )


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
    tg(
        "sendMessage",
        {"chat_id": TELEGRAM_CHAT_ID, "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇"},
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
