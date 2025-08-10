import os, time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient, ASCENDING, TEXT
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "news")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "toi_articles")
RSS_URL = os.getenv("RSS_URL", "https://www.timesofisrael.com/feed/")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
col = db[COLLECTION_NAME]

# Indexes
col.create_index([("url", ASCENDING)], unique=True)
col.create_index([("title", TEXT), ("content", TEXT)], name="text_search")

HEADERS = {"User-Agent": "Mozilla/5.0 (+news-loader; Contact: you@example.com)"}

def get_meta(soup, *names):
    for n in names:
        tag = soup.find("meta", attrs={"name": n}) or soup.find("meta", attrs={"property": n})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None

def clean_text_blocks(blocks):
    lines = []
    for b in blocks:
        t = b.get_text(" ", strip=True)
        if t:
            lines.append(t)
    out, prev = [], None
    for line in lines:
        if line != prev and len(line) > 2:
            out.append(line)
        prev = line
    return "\n\n".join(out)

def extract_toi_article(url):
    """Extract full article text from The Times of Israel."""
    r = requests.get(url, timeout=25, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    title = get_meta(soup, "og:title", "twitter:title") or (soup.title.string.strip() if soup.title else url)
    
    author = (
        get_meta(soup, "author", "article:author")
        or (soup.select_one('[rel="author"]') or soup.select_one(".author"))
        or None
    )
    if hasattr(author, "get_text"):
        author = author.get_text(" ", strip=True)
    published = get_meta(soup, "article:published_time", "og:article:published_time", "pubdate", "date")
    section = get_meta(soup, "article:section") or None

    # TOI main article text lives in div[itemprop="articleBody"] or .article-content
    selectors = [
        'div[itemprop="articleBody"] p',
        '.article-content p'
    ]
    paragraphs = []
    for sel in selectors:
        paragraphs = soup.select(sel)
        if paragraphs and sum(len(p.get_text(strip=True)) for p in paragraphs) > 300:
            break
    if not paragraphs:
        paragraphs = soup.find_all("p")

    content = clean_text_blocks(paragraphs)

    return {
        "url": url,
        "title": title,
        "author": author,
        "published_at": published,
        "content": content if content else None,
        "section": section,
        "source": urlparse(url).netloc,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

def fetch_from_rss(rss_url, limit=10):
    r = requests.get(rss_url, timeout=25, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "xml")

    out = []
    items = soup.find_all("item")
    if items:
        for it in items[:limit]:
            link = it.find("link")
            if link and link.text:
                out.append(link.text.strip())
        return out

    for e in soup.find_all("entry")[:limit]:
        link = e.find("link")
        if link and link.get("href"):
            out.append(link["href"].strip())
    return out

def upsert_article(doc):
    if not doc.get("content"):
        return False
    col.update_one({"url": doc["url"]}, {"$set": doc}, upsert=True)
    return True

if __name__ == "__main__":
    urls = fetch_from_rss(RSS_URL, limit=10)
    print(f"Found {len(urls)} RSS URLs")
    saved = 0
    for i, url in enumerate(urls, 1):
        try:
            art = extract_toi_article(url)
            if upsert_article(art):
                saved += 1
                print(f"[{i}] Saved: {art['title'][:90]}")
            else:
                print(f"[{i}] Skipped (no content): {url}")
            time.sleep(0.6)
        except Exception as e:
            print(f"[{i}] Error on {url}: {e}")
    print(f"Done. Saved {saved}/{len(urls)}. Total in collection: {col.count_documents({})}")
