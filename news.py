import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

TAG_URLS = ["https://www.suara.com/tag/mixue", "https://www.suara.com/tag/mixue-indonesia"]
TAG_PAGES = 4
MAX_ARTICLES = 60
MAX_COMMENT_PAGES = 8  # 50 per page, so at most 400 per article
SLEEP = 2

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "id-ID,id;q=0.9",
}
COMMENT_API = "https://comment.arkadia.me/go/api/comment"

BODY_SELECTORS = ["div.wrapper-full-article", "div.detail__body-text",
                  "div.article-body", "div.article-content-body", "article"]

ARTICLE_COLS = ["url", "title", "author", "date", "content"]
COMMENT_COLS = ["comment_id", "article_url", "username", "comment", "likes", "date"]


def get(url, **kw):
    try:
        return requests.get(url, headers=HEADERS, timeout=30, **kw)
    except requests.RequestException as e:
        print(f"  gagal mengambil {url[:70]}: {type(e).__name__}")
        return None


def find_articles():
    # the tag listing pages are plain html, no login, no api
    urls = []
    for base in TAG_URLS:
        for page in range(1, TAG_PAGES + 1):
            r = get(base if page == 1 else f"{base}?page={page}")
            if r is None:
                continue
            urls += re.findall(r"https://[a-z]+\.suara\.com/(?:news|read)/\d{4}/\d{2}/\d{2}/\d{6}/[a-z0-9-]+", r.text)
            time.sleep(1)
    return list(dict.fromkeys(urls))[:MAX_ARTICLES]


def pull_article(url):
    r = get(url)
    if r is None:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title:
        title = soup.title.get_text(strip=True)

    author, date = "", ""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            d = json.loads(tag.string or "")
        except Exception:
            continue
        if isinstance(d, dict) and d.get("@type") in ("NewsArticle", "Article", "ReportageNewsArticle"):
            date = d.get("datePublished") or ""
            a = d.get("author")
            if isinstance(a, dict):
                author = a.get("name") or ""
            elif isinstance(a, list) and a:
                author = a[0].get("name", "") if isinstance(a[0], dict) else str(a[0])
            break
    if not date:
        meta = soup.find("meta", property="article:published_time")
        date = meta["content"] if meta and meta.get("content") else ""
    if not author:
        meta = soup.find("meta", attrs={"name": "author"})
        author = meta["content"] if meta and meta.get("content") else ""

    body = None
    for sel in BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            body = node
            break
    paragraphs = []
    if body:
        paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) > 40]
    content = " ".join(paragraphs)

    # the tag page also shows recommended articles that have nothing to do with mixue
    if "mixue" not in (title + " " + content).lower():
        return None

    return {"url": url, "title": title, "author": author, "date": date, "content": content}


def pull_comments(url):
    rows, page = [], 1
    while page <= MAX_COMMENT_PAGES:
        r = get(COMMENT_API, params={"url": url, "page": page, "per_page": 50})
        if r is None:
            break
        try:
            d = r.json()
        except Exception:
            break
        data = d.get("data") or []
        for c in data:
            user = c.get("user")
            if isinstance(user, dict):
                name = user.get("name") or user.get("username") or ""
            else:
                name = str(user or "")
            rows.append({
                "comment_id": str(c.get("_id") or ""),
                "article_url": url,
                "username": name,
                "comment": str(c.get("text") or "").strip(),
                "likes": c.get("like") or 0,
                "date": str(c.get("created_at") or ""),
            })
        if len(data) < 50 or not d.get("next_page"):
            break
        page += 1
        time.sleep(0.5)
    return rows


def main():
    os.makedirs("data", exist_ok=True)
    out_articles = "data/news_articles.csv"
    out_comments = "data/news_comments.csv"

    done_urls, done_ids = set(), set()
    if os.path.exists(out_articles):
        done_urls = set(pd.read_csv(out_articles, dtype=str)["url"])
        print(f"lanjut: {len(done_urls)} artikel sudah tersimpan")
    if os.path.exists(out_comments):
        done_ids = set(pd.read_csv(out_comments, dtype=str)["comment_id"])

    urls = find_articles()
    print(f"ketemu {len(urls)} artikel mixue di suara.com")

    for url in urls:
        if url in done_urls:
            print(f"  sudah ada: {url[:70]}")
            continue
        art = pull_article(url)
        if art is None:
            print(f"  dilewati (bukan artikel mixue): {url[:70]}")
            continue
        pd.DataFrame([art], columns=ARTICLE_COLS).to_csv(
            out_articles, mode="a", header=not os.path.exists(out_articles), index=False)
        done_urls.add(url)

        comments = pull_comments(url)
        new = [c for c in comments if c["comment_id"] and c["comment_id"] not in done_ids]
        if new:
            done_ids.update(c["comment_id"] for c in new)
            pd.DataFrame(new, columns=COMMENT_COLS).to_csv(
                out_comments, mode="a", header=not os.path.exists(out_comments), index=False)

        print(f"  {art['title'][:60]!r} | {len(new)} komentar")
        time.sleep(SLEEP)

    print(f"\nselesai: {len(done_urls)} artikel, {len(done_ids)} komentar di data/")


if __name__ == "__main__":
    main()
