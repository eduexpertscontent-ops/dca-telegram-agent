import os
import json
import time
import datetime as dt
import requests
import random
from openai import OpenAI

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]  # @UPPCSSUCCESS
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI()  # reads OPENAI_API_KEY from environment

def tg(method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=30)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram error: {data}")
    return data

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

def _generate_n_mcqs(n: int, avoid_questions):
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    system = (
        "You create DAILY current affairs MCQs for competitive exams in ENGLISH.\n"
        "Use web search and prefer facts from the last 24–48 hours.\n"
        "ABSOLUTE RULES:\n"
       "- Each MCQ must have a UNIQUE event_key (3–8 words). No two MCQs can share the same event_key.\n"
"- correct_answer must EXACTLY match one of the 4 options.\n"
"- correct_option_id must be the index of correct_answer in options.\n"
"- All 4 options must be distinct.\n"

    )

    avoid_block = "\n".join([f"- {a}" for a in (avoid_questions or [])[:50]])

    user = (
        f"Create {n} UNIQUE current affairs MCQs for date {today}. "
        "Mix: India, World, Economy/Banking, Science/Tech, Defence, Environment, Sports, Awards.\n"
        "Avoid duplicating ANY of these existing questions:\n"
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
        text={"format": {"type": "json_schema", "name": "daily_ca_mcqs_n", "strict": True, "schema": schema}},
    )
    return json.loads(resp.output_text)

def generate_mcqs() -> dict:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()

    base = _generate_n_mcqs(10, avoid_questions=[])
    unique = _dedupe_keep_order(base["mcqs"])

    tries = 0
    while len(unique) < 10 and tries < 3:
        missing = 10 - len(unique)
        avoid_list = [q["question"] for q in unique]
        extra = _generate_n_mcqs(missing, avoid_questions=avoid_list)
        unique = _dedupe_keep_order(unique + extra["mcqs"])
        tries += 1

    base["date"] = today
    base["mcqs"] = unique[:10]
    return base

def shuffle_options_and_fix_answer(q: dict) -> dict:
    """
    Shuffles options so correct answer isn't always A.
    Updates correct_option_id accordingly.
    """
    options = q["options"]
    correct_text = options[q["correct_option_id"]]

    # Shuffle with a stable seed so results are consistent per question/day
    seed = q["question"] + correct_text
    rnd = random.Random(seed)
    shuffled = options[:]
    rnd.shuffle(shuffled)

    q["options"] = shuffled
    q["correct_option_id"] = shuffled.index(correct_text)
    return q

def post_to_channel(mcq_set: dict):
    tg(
        "sendMessage",
        {"chat_id": TELEGRAM_CHAT_ID,
         "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇"}
    )

    for i, q in enumerate(mcq_set["mcqs"], start=1):
        q = shuffle_options_and_fix_answer(q)

        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": f"Q{i}. {q['question']}",
            "options": [{"text": opt} for opt in q["options"]],
            "type": "quiz",
            "correct_option_id": q["correct_option_id"],
            "explanation": q["explanation"],
            "is_anonymous": True,
        }
        tg("sendPoll", payload)
        time.sleep(1)
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
        text={"format": {"type": "json_schema", "name": "validated_mcqs", "strict": True, "schema": SCHEMA}},
    )
    return json.loads(resp.output_text)

def main():
    mcq_set = generate_mcqs()
    mcq_set = validate_and_fix_mcqs(mcq_set)

    post_to_channel(mcq_set)
    print("✅ Posted 10 quiz polls successfully.")

if __name__ == "__main__":
    main()
