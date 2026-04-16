import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import pandas as pd
import time
import random

HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_urls(sitemap_url):
    response = requests.get(sitemap_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "xml")
    return [loc.text for loc in soup.find_all("loc")]

main_sitemap = "https://kalliope.org/sitemap.xml"
author_sitemaps = get_urls(main_sitemap)

all_urls = []

for sitemap in author_sitemaps:
    try:
        urls = get_urls(sitemap)
        all_urls.extend(urls)
    except:
        pass

poem_urls = [u for u in all_urls if "/text/" in u]

random.shuffle(poem_urls)

print(f"Found {len(poem_urls)} poem URLs")


poems = []

def extract_poem(page, url):
    page.goto(url, timeout=60000)
    page.wait_for_timeout(1500)

    title = page.title()

    content = page.query_selector("main") or page.query_selector("article")
    text = content.inner_text() if content else ""

    return {
        "text": text,
        "title": title,
        "url": url
    }

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    for url in poem_urls[:200]:
        try:
            poem = extract_poem(page, url)

            # filter out garbage pages
            if poem["text"] and len(poem["text"]) > 50:
                poems.append(poem)

            print("Collected:", len(poems))

            time.sleep(1)

        except Exception as e:
            print("Error:", e)

    browser.close()

df = pd.DataFrame(poems)
df.to_csv("poems.csv", index=False, encoding="utf-8")

print("Done! Saved poems.csv")