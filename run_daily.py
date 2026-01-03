import os
import json
import time
import datetime as dt
import requests
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
                    "question": {"type": "string", "minLength": 1, "maxLength": 300},
                    "options": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1, "maxLength": 100},
                    },
                    "correct_option_id": {"type": "integer", "minimum": 0, "maximum": 3},
                    "explanation": {"type": "string", "maxLength": 200},
                },
                "required": ["question", "options", "correct_option_id", "explanation"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["date", "mcqs"],
    "additionalProperties": False,
}

def generate_mcqs() -> dict:
    today = dt.datetime.utcnow().date().isoformat()

    system = (
        "You create DAILY current affairs MCQs for competitive exams in ENGLISH.\n"
        "Use web search and prefer facts from the last 24–48 hours.\n"
        "Return EXACTLY 10 MCQs.\n"
        "Each MCQ must have exactly 4 options.\n"
        "Avoid ambiguous or opinion-based questions.\n"
        "Keep explanations <= 200 characters.\n"
        "Make questions short, clear, exam-relevant.\n"
    )

    user = (
        f"Create 10 current affairs MCQs for date {today}.\n"
        "Mix topics: India, World, Economy/Banking, Science/Tech, Defence, Environment, Sports, Awards.\n"
        "Ensure only one correct option and correct_option_id matches it."
    )

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
                "name": "daily_ca_mcqs",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    )
    return json.loads(resp.output_text)

def post_to_channel(mcq_set: dict):
    tg(
        "sendMessage",
        {"chat_id": TELEGRAM_CHAT_ID,
         "text": f"🧠 Daily Current Affairs Quiz ({mcq_set['date']})\n\n10 MCQ polls below 👇"}
    )

    for i, q in enumerate(mcq_set["mcqs"], start=1):
        payload = {
            "chat_id": TELELEGRAM_CHAT_ID if False else TELEGRAM_CHAT_ID,
            "question": f"Q{i}. {q['question']}",
            "options": [{"text": opt} for opt in q["options"]],
            "type": "quiz",
            "correct_option_id": q["correct_option_id"],
            "explanation": q["explanation"],
            "is_anonymous": True,
        }
        tg("sendPoll", payload)
        time.sleep(1)

def main():
    mcq_set = generate_mcqs()
    post_to_channel(mcq_set)
    print("✅ Posted 10 quiz polls successfully.")

if __name__ == "__main__":
    main()
