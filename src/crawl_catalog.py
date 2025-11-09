import re, time, json
from urllib.parse import urlparse
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
ROOT = "https://www.shl.com"
SITEMAP_INDEXES = [
    "https://www.shl.com/sitemap_index.xml",
    "https://www.shl.com/sitemap.xml",  # try both, one will usually exist
]
# Also try the two catalog landing pages (sometimes they have static anchors)
CATALOG_SEEDS = [
    "https://www.shl.com/solutions/products/product-catalog/",
    "https://www.shl.com/products/product-catalog/",
]

def get(url, **kw):
    r = requests.get(url, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r

def parse_xml_urls(xml_text):
    # naive extraction of <loc>...</loc> values
    return re.findall(r"<loc>(.*?)</loc>", xml_text, flags=re.I)

def discover_product_urls_from_sitemaps():
    found = set()
    # Fetch sitemap indexes
    for sm in SITEMAP_INDEXES:
        try:
            r = get(sm)
        except Exception:
            continue
        urls = parse_xml_urls(r.text)
        # Each URL may be a sitemap or a page; load sitemaps too
        for u in urls:
            if not u.lower().endswith(".xml"):
                # It might directly list pages (some sites do)
                if "shl.com" in u and "/product" in u.lower():
                    found.add(u)
                continue
            try:
                rsm = get(u)
            except Exception:
                continue
            page_urls = parse_xml_urls(rsm.text)
            for pu in page_urls:
                # Heuristic: keep likely product pages, ignore job-packages
                low = pu.lower()
                if "shl.com" not in low:
                    continue
                if "product" in low or "/solutions/products/" in low:
                    if "pre-packaged" in low or "job-solution" in low or "job-solution" in low:
                        continue
                    found.add(pu)
    return found

def discover_from_catalog_pages():
    found = set()
    for seed in CATALOG_SEEDS:
        try:
            r = get(seed)
        except Exception:
            continue
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/"):
                href = ROOT + href
            low = href.lower()
            if "shl.com" in low and ("/product" in low or "/solutions/products/" in low):
                if "pre-packaged" in low:
                    continue
                found.add(href)
    return found

def extract_product_info(url):
    """
    Try to pull: name (H1/title), url (given), desc (meta/first paragraph).
    Keep it resilient: use multiple fallbacks.
    """
    try:
        r = get(url)
    except Exception:
        return None
    soup = BeautifulSoup(r.text, "lxml")

    # name
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else None
    if not name:
        # fall back to <title>
        t = soup.find("title")
        name = t.get_text(strip=True) if t else None

    # description: prefer meta description, then first paragraph
    desc = None
    md = soup.find("meta", attrs={"name":"description"})
    if md and md.get("content"):
        desc = md["content"].strip()
    if not desc:
        p = soup.find("p")
        if p:
            desc = p.get_text(" ", strip=True)

    # filter out clearly non-product pages
    if not name:
        return None

    # Type hint heuristic
    low = f"{name} {desc}".lower() if desc else name.lower()
    type_hint = None
    if any(k in low for k in ["technical","coding","programming","knowledge","skills","aptitude","cognitive","numerical","verbal","inductive","deductive"]):
        type_hint = "K"
    if any(k in low for k in ["personality","behaviour","behavior","work style","situational judgment","sjt","motivation","values"]):
        # override to P if it's clearly behavioral
        type_hint = "P" if type_hint is None else type_hint

    return {
        "name": name.strip(),
        "url": url,
        "desc": (desc or "").strip(),
        "type_hint": type_hint or "",
    }

def main():
    # 1) find candidate URLs
    urls = set()
    urls |= discover_product_urls_from_sitemaps()
    urls |= discover_from_catalog_pages()

    # Keep only plausible product pages; drop obvious non-products
    cleaned = set()
    for u in urls:
        low = u.lower()
        if any(x in low for x in ["/blog/", "/news/", "/press", "/careers", "/events"]):
            continue
        if "pre-packaged" in low:
            continue
        cleaned.add(u)

    print(f"Found {len(cleaned)} candidate URLs. Extracting details...")
    rows = []
    seen = set()
    for i, u in enumerate(sorted(cleaned)):
        if u in seen: 
            continue
        seen.add(u)
        info = extract_product_info(u)
        if info and info["name"] and ROOT in u:
            rows.append(info)
        # be polite
        time.sleep(0.2)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"]).reset_index(drop=True)

    # If still empty, leave a helpful message
    if df.empty:
        print("No products extracted. The site may be heavily JS-driven. Try re-running later or use manual seeding.")
    else:
        df.to_csv("data/catalog_clean.csv", index=False)
        print(f"✅ Saved {len(df)} items to data/catalog_clean.csv")

if __name__ == "__main__":
    main()
