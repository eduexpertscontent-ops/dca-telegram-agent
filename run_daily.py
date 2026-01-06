import os
import re
import json
import time
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- CONFIG --------------------
BASE_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POLL_DELAY_SECONDS = 2.0 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------- CORE FUNCTIONS --------------------
def tg_call(method, payload):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    r = requests.post(url, json=payload, timeout=20)
    return r.json()

def fetch_detailed_data():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        r = requests.get(BASE_URL, headers=headers, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"Error: {e}")
        return []

    news_data = []
    # 1. Target the Table specifically to get Headlines + Source Links
    rows = soup.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline = cells[0].get_text(strip=True)
            # Find the link in the "Source" or "Headline" column
            link_tag = row.find('a', href=True)
            
            if link_tag and len(headline) > 20:
                href = link_tag['href']
                full_url = "https://www.nextias.com" + href if href.startswith("/") else href
                
                # Filter out ads
                if any(word in headline.lower() for word in ["course", "batch", "admission", "ias center"]):
                    continue
                
                news_data.append({"headline": headline, "url": full_url})

    # 2. Deep Extraction: Visit each link to get the actual news content
    detailed_news = []
    print(f"Found {len(news_data)} news links. Starting deep extraction...")
    
    for item in news_data[:12]: # Process top 12 to ensure we get 10 high-quality ones
        try:
            print(f"Extracting: {item['headline'][:40]}...")
            res = requests.get(item['url'], headers=headers, timeout=10)
            asoup = BeautifulSoup(res.text, "html.parser")
            
            # Target the actual article body
            content_div = asoup.find("div", class_=["daily-ca-content", "article-details", "entry-content"])
            if content_div:
                body_text = " ".join([p.get_text() for p in content_div.find_all(["p", "li"]) if len(p.get_text()) > 30])
            else:
                body_text = " ".join([p.get_text() for p in asoup.find_all("p") if len(p.get_text()) > 50])
            
            detailed_news.append({
                "headline": item['headline'],
                "content": body_text[:4000], # Limit per article for AI tokens
                "url": item['url']
            })
            time.sleep(1) # Polite delay
        except:
            continue
            
    return detailed_news

def generate_mcqs(news_list):
    if not news_list:
        return []

    # Building a prompt that forces UPSC framing and distinct topics
    prompt = f"""
    You are a Senior UPSC Faculty. Generate EXACTLY 10 MCQs based on the detailed news provided.
    
    NEWS CONTENT:
    {json.dumps(news_list, indent=2)}
    
    STRICT UPSC EXAM RULES:
    1. FRAMING: Use UPSC style (e.g., 'Consider the following statements', 'Which of the following are correctly matched?').
    2. DATA: Use specific numbers, years, organizations, and names mentioned in the text.
    3. UNIQUENESS: Generate exactly ONE question per news topic. Do not repeat topics.
    4. NO ADS: Never mention 'Next IAS', 'courses', or 'foundation'.
    
    OUTPUT FORMAT: JSON only.
    {{
      "mcqs": [
        {{
          "question": "...",
          "options": ["...", "...", "...", "..."],
          "correct_index": 0,
          "source": "..."
        }}
      ]
    }}
    """
    
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": "You are a professional examiner. Output strictly in JSON format."}, 
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content).get("mcqs", [])[:10]

# -------------------- MAIN --------------------
def main():
    print("Step 1: Fetching headlines and source links...")
    news_list = fetch_detailed_data()
    
    print(f"Step 2: Generating 10 professional MCQs from {len(news_list)} detailed sources...")
    mcqs = generate_mcqs(news_list)
    
    if not mcqs:
        print("Error: No MCQs generated.")
        return

    date_label = dt.datetime.now().strftime("%d %B %Y")
    tg_call("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": f"🔥 *UPSC Daily High-Yield MCQ: {date_label}*\n(Based on Detailed Reports)", "parse_mode": "Markdown"})
    
    for m in mcqs:
        tg_call("sendPoll", {
            "chat_id": TELEGRAM_CHAT_ID,
            "question": m["question"][:300],
            "options": m["options"][:4],
            "is_anonymous": True,
            "type": "quiz",
            "correct_option_id": m["correct_index"]
        })
        time.sleep(POLL_DELAY_SECONDS)

if __name__ == "__main__":
    main()
