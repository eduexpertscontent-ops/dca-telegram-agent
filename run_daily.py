import os
import datetime
import requests
import json
import asyncio
from openai import OpenAI
from telegram import Bot

# Configuration from Render Environment Variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def get_today_news():
    """Fetch headlines from today's date only."""
    today = datetime.date.today().isoformat()
    url = f"https://newsapi.org/v2/top-headlines?country=in&from={today}&to={today}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    articles = response.get('articles', [])[:10]
    return [a['title'] for a in articles] if articles else None

async def post_mcqs():
    bot = Bot(token=TELEGRAM_TOKEN)
    headlines = get_today_news()
    today_date = datetime.date.today().strftime("%Y-%m-%d")
    
    if not headlines:
        print("No fresh news found today.")
        return

    # OpenAI MCQ Generation
    prompt = (
        f"Create 10 UPSC-level Current Affairs MCQs for date {today_date} based on: {headlines}. "
        "Return ONLY a JSON object with a key 'mcqs' containing a list of: "
        "{'question': str, 'options': [4 strings], 'correct_index': int, 'explanation': str}."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a professional UPSC exam paper setter."},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    data = json.loads(response.choices[0].message.content)
    mcqs = data.get('mcqs', [])

    # 1. Post the 10 MCQs
    for item in mcqs:
        try:
            await bot.send_poll(
                chat_id=CHAT_ID,
                question=item['question'][:300],
                options=item['options'],
                type='quiz',
                correct_option_id=item['correct_index'],
                explanation=item['explanation'][:200],
                is_anonymous=False
            )
            await asyncio.sleep(4) 
        except Exception as e:
            print(f"Error: {e}")

    # 2. Post the Footer Message
    footer_text = (
        "UPSC/UPPCS SUCCESS MIND✍️\n"
        "🏁 Today’s Practice Ends Here!\n\n"
        "Comment your score below 👇\n"
        "⏰ Back tomorrow at the same time."
    )
    await bot.send_message(chat_id=CHAT_ID, text=footer_text)

    # 3. Post the Final Score Poll
    score_options = ["25 ✅", "24–23", "22–20", "19–17", "16–13", "12–7", "6–0", "Below that 😅"]
    await bot.send_poll(
        chat_id=CHAT_ID,
        question=f"📊 Vote your score ({today_date}) ✅\nHow many were correct?",
        options=score_options,
        is_anonymous=False
    )

if __name__ == "__main__":
    asyncio.run(post_mcqs())
