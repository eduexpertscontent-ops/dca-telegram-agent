import os
import requests
from bs4 import BeautifulSoup
import datetime as dt
import time
import json
from openai import OpenAI

# --- CONFIG ---
HUB_URL = "https://www.nextias.com/daily-current-affairs"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_today_headlines_link():
    """Finds the specific link for today's headlines from the main hub."""
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        # Look for headlines of the day links
        for a in soup.find_all('a', href=True):
            if "/ca/headlines-of-the-day/" in a['href']:
                return "https://www.nextias.com" + a['href'] if a['href'].startswith("/") else a['href']
    except Exception as e:
        print(f"Error finding today's link: {e}")
    return None

def fetch_deep_content():
    target_url = get_today_headlines_link()
    if not target_url:
        print("Could not find today's specific headlines page.")
        return []

    print(f"Targeting specific news page: {target_url}")
    try:
        r = requests.get(target_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
    except:
        return []

    news_list = []
    # Target rows in the headlines table
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline = cells[0].get_text(strip=True)
            link_tag = row.find('a', href=True)
            
            # Filter ads and empty headers
            if link_tag and len(headline) > 20 and "admission" not in headline.lower():
                href = link_tag['href']
                full_url = "https://www.nextias.com" + href if href.startswith("/") else href
                
                # Visit detailed article for content
                try:
                    sub_r = requests.get(full_url, headers=HEADERS, timeout=10)
                    sub_soup = BeautifulSoup(sub_r.text, "html.parser")
                    # Target content divs
                    content = " ".join([p.text for p in sub_soup.find_all(['p', 'li']) if len(p.text) > 40])
                    news_list.append({"headline": headline, "content": content[:4000], "url": full_url})
                    time.sleep(1) 
                except: continue
                
                if len(news_list) >= 12: break
                
    return news_list

# Use your existing generate_mcqs and main logic from here...
