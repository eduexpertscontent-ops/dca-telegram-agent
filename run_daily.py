# ... (Keep your existing imports and scraper functions)

# -------------------- UPDATED AI GENERATOR --------------------
def generate_mcqs(news_list):
    # Ensure we have enough news to work with
    log(f"Passing {len(news_list)} articles to AI...")
    
    prompt = f"""
    You are a UPSC Exam setter. I have provided a list of {len(news_list)} news articles.
    YOUR TASK: Generate EXACTLY 10 different MCQs.
    
    RULES:
    1. One MCQ per article (do not combine articles).
    2. Format must be factual and UPSC statement-style.
    3. Output must be a valid JSON with an array of 10 objects.
    
    NEWS DATA:
    {json.dumps(news_list)}
    
    OUTPUT FORMAT:
    {{
      "mcqs": [
        {{
          "question": "...",
          "options": ["A", "B", "C", "D"],
          "correct_index": 0,
          "source": "..."
        }}
      ]
    }}
    """
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "You are a professional examiner. You MUST return exactly 10 MCQs in JSON format."}, 
                      {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3 # Slightly higher to ensure variety
        )
        data = json.loads(resp.choices[0].message.content)
        mcqs = data.get("mcqs", [])
        log(f"AI successfully generated {len(mcqs)} MCQs.")
        return mcqs
    except Exception as e:
        log(f"AI Generation failed: {e}")
        return []

# -------------------- UPDATED POSTER --------------------
def post_to_telegram(mcqs):
    if not mcqs:
        log("No MCQs to post.")
        return

    # Post Header
    date_str = dt.datetime.now().strftime("%d %B %Y")
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                  json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚀 *Daily MCQ Bulletin: {date_str}*", "parse_mode": "Markdown"})

    # Post exactly what was generated
    count = 0
    for m in mcqs:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "question": f"{m['question'][:250]}\n\nSource: {m['source']}",
                "options": json.dumps(m['options']),
                "is_anonymous": True,
                "type": "quiz",
                "correct_option_id": m['correct_index']
            }
            r = requests.post(url, data=payload)
            if r.status_code == 200:
                count += 1
                log(f"Posted Poll {count}/10")
            else:
                log(f"Failed to post poll: {r.text}")
            time.sleep(3) # Increased delay to prevent Telegram flooding
        except Exception as e:
            log(f"Error posting poll: {e}")

    # Post Score Poll only if questions were posted
    if count > 0:
        score_payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": "📊 Final Score: How many did you get correct?",
            "options": json.dumps(["0-3 (Need Revision)", "4-6 (Good)", "7-8 (Very Good)", "9-10 (Excellent!)"]),
            "is_anonymous": True,
            "type": "regular"
        }
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPoll", data=score_payload)
