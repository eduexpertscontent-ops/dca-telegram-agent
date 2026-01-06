import os
import re
import json
import time
import datetime as dt
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# -------------------- CONFIG --------------------
HUB_URL = "https://www.nextias.com/daily-current-affairs"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POLL_DELAY_SECONDS = 2.0 

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# -------------------- IMPROVED SCRAPER --------------------
def get_latest_news_page():
    """Finds the most recent daily news link from the hub."""
    try:
        r = requests.get(HUB_URL, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        # Look for links containing current dates or daily-current-affairs patterns
        for a in soup.find_all('a', href=True):
            if "/ca/headlines-of-the-day/" in a['href'] or "/ca/current-affairs/" in a['href']:
                return "https://www.nextias.com" + a['href'] if a['href'].startswith("/") else a['href']
    except Exception as e:
        print(f"Error finding hub links: {e}")
    return HUB_URL

def fetch_content():
    target_url = get_latest_news_page()
    print(f"Targeting page: {target_url}")
    
    try:
        r = requests.get(target_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
    except:
        return []

    news_list = []
    # TARGET: Tables with 'Headline | Source | Syllabus'
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 2:
            headline = cells[0].get_text(strip=True)
            link_tag = row.find('a', href=True)
            
            if link_tag and len(headline) > 20:
                href = link_tag['href']
                full_url = "https://www.nextias.com" + href if href.startswith("/") else href
                if "course" in headline.lower() or "admission" in headline.lower(): continue
                
                # Fetch sub-article content for detail
                try:
                    sub_r = requests.get(full_url, headers=HEADERS, timeout=10)
                    sub_soup = BeautifulSoup(sub_r.text, "html.parser")
                    content = " ".join([p.text for p in sub_soup.find_all(['p', 'li']) if len(p.text) > 40])
                    news_list.append({"headline": headline, "content": content[:3500], "url": full_url})
                except: continue
                
                if len(news_list) >= 12: break # Limit to save tokens/time
                
    return news_list

# [Keep the generate_mcqs and main functions from the previous version]
