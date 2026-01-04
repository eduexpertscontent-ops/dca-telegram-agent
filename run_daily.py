import os
import json
import time
import datetime as dt
import requests
import random
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

HISTORY_FILE = "mcq_history.json"

client = OpenAI()  # reads OPENAI_API_KEY from environment


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


# -------------------- HISTORY (NO-REPEAT ACROSS DAYS) --------------------
def load_history() -> dict:
    """
    { "dates": { "YYYY-MM-DD": {"questions":[...], "event_keys":[...]} }, "questions":[...], "event_keys":[...] }
    """
    if not os.path.exists(HISTORY_FILE):
        return {"dates": {}, "questions": [], "event_keys": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            hist = json.load(f)
        if "dates" not in hist:
            hist["dates"] = {}
        if "questions" not in hist:
            hist["questions"] = []
        if "event_keys" not in hist:
            hist["event_keys"] = []
        return hist
    except Exception:
        return {"dates": {}, "questions": [], "event_keys": []}


def save_history(hist: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def update_history_with_set(hist: dict, mcq_set: dict, keep_last_days: int) -> dict:
    today = mcq_set["date"]
    hist.setdefault("dates", {})
    hist["dates"][today] = {
        "questions": [q["question"] for q in mcq_set["mcqs"]],
        "event_keys": [q["event_key"] for q in mcq_set["mcqs"]],
    }

    # prune old dates
    all_dates = sorted(hist["dates"].keys())
    if len(all_dates) > keep_last_days:
        for d in all_dates[:-keep_last_days]:
            hist["dates"].pop(d, None)

    # rebuild flattened lists (bounded)
    qs, eks = [], []
    for d in sorted(hist["dates"].keys()):
        qs.extend(hist["dates"][d].get("questions", []))
        eks.extend(hist["dates"][d].get("event_keys", []))

    hist["questions"] = qs[-250:]
    hist["event_keys"] = eks[-250:]
    return hist


# -------------------- DEDUPE --------------------
def _norm_q(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def _dedupe_keep_order(mcqs):
    seen = set()
    out = []
    for q in mcqs:
        key = _norm_q(q["question"])
        if key not in seen:
            seen.add(key)
            out.append(q)
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


# -------------------- OPENAI GENERATION --------------------
def _generate_n_mcqs(n: int, avoid_questions):
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    system = (
        "You create DAILY current affairs MCQs for competitive exams in ENGLISH.\n"
        "Use web search.\n"
        "STRICT FRESHNESS:\n"
        "- Prefer events announced/updated in the last 72 hours.\n"
        "- If a topic is older, use it ONLY if it has a NEW update in last 72 hours.\n"
        "STRICT DIVERSITY:\n"
        "- Each MCQ must be based on a DIFFERENT news event/topic.\n"
        "- Do NOT repeat topics from the avoid list.\n"
        "ABSOLUTE RULES:\n"
        "- Each MCQ must have a UNIQUE event_key (3–8 words).\n"
        "- correct_answer must EXACTLY match one of the 4 options.\n"
        "- correct_option_id must be the index of correct_answer in options.\n"
        "- All 4 options must be distinct.\n"
        "- explanation must be <= 200 characters.\n"
    )

    avoid_block = "\n".join([f"- {a}" for a in (avoid_questions or [])[:120]])

    user = (
        f"Create {n} UNIQUE current affairs MCQs for date {today}.\n"
        "Mix: India, World, Economy/Banking, Science/Tech, Defence, Environment, Sports, Awards.\n"
        "AVOID repeating ANY of these questions/topics:\n"
        f"{avoid_block}\n"
        "Return JSON exactly as per schema."
    )

    schema = _schema_for_n(n)

    resp = client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type": "web_search"}],
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


def generate_mcqs() -> dict:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    hist = load_history()
    avoid_qs = (hist.get("questions", []) or [])[-120:]

    base = _generate_n_mcqs(10, avoid_questions=avoid_qs)
    unique = _dedupe_keep_order(base["mcqs"])

    tries = 0
    while len(unique) < 10 and tries < 3:
        missing = 10 - len(unique)
        avoid_list = avoid_qs + [q["question"] for q in unique]
        extra = _generate_n_mcqs(missing, avoid_questions=avoid_list)
        unique = _dedupe_keep_order(unique + extra["mcqs"])
        tries += 1

    base["date"] = today
    base["mcqs"] = unique[:10]
    return base


def validate_and_fix_mcqs(mcq_set: dict) -> dict:
    system = (
        "You are a strict fact-checker for current affairs MCQs.\n"
        "Verify each MCQ using web search. Fix any incorrect answers/options.\n"
        "Rules:\n"
        "- Keep EXACTLY 10 MCQs.\n"
        "- Keep options as 4.\n"
        "- correct_answer must match one option exactly.\n"
        "- correct_option_id must match correct_answer.\n"
        "- event_key must be unique across all 10.\n"
        "- Explanations <= 200 chars.\n"
        "Return JSON only in the same schema."
    )

    resp = client.responses.create(
        model=OPENAI_MODEL,
        tools=[{"type": "web_search"}],
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Fix and validate this set:\n{json.dumps(mcq_set)}"},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "validated_mcqs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    )
    return json.loads(resp.output_text)


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
            "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇",
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
    mcq_set = generate_mcqs()
    mcq_set = validate_and_fix_mcqs(mcq_set)

    post_to_channel(mcq_set)

    # Save history AFTER successful generation/post attempt
    hist = load_history()
    hist = update_history_with_set(hist, mcq_set, keep_last_days=KEEP_LAST_DAYS)
    save_history(hist)

    print("✅ Posted 10 quiz polls + message + score poll successfully.")


if __name__ == "__main__":
    main()
