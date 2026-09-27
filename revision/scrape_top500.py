"""Collect TOP500 lists (Nov 2023, Jun 2024, Nov 2025): rank, system, site, country, Rmax, Rpeak (PFlop/s), segment.
Source: https://www.top500.org/lists/top500/list/YYYY/MM/ (public list pages) and /site/<id> pages for Segment."""
import re, time, csv, os, urllib.request, html
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0 (academic research; replication of CEEJ-2026-0071)"}
def get(url):
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "ignore")
        except Exception as e:
            print("retry", url, e); time.sleep(5)
    raise RuntimeError(url)
ROW = re.compile(r"<tr>\s*<td>(\d+)</td>\s*<td>\s*<a href=\"/system/(\d+)\">(.*?)</a>(.*?)<a href=\"/site/(\d+)\">(.*?)</a><br>(.*?)</td>\s*"
                 r"<td[^>]*>([\d,]*)</td>\s*<td[^>]*>([\d,.]*)</td>\s*<td[^>]*>([\d,.]*)</td>", re.S)
def clean(s): return html.unescape(re.sub(r"<[^>]+>", " ", s)).strip()
sites = {}
for ym in ["2023/11", "2024/06", "2025/11"]:
    rows = []
    for page in range(1, 6):
        h = get(f"https://www.top500.org/lists/top500/list/{ym}/?page={page}")
        for m in ROW.finditer(h):
            rows.append(dict(rank=int(m[1]), system_id=m[2], system=clean(m[3])[:120], site_id=m[5], site=clean(m[6]),
                             country=clean(m[7]), rmax_pflops=float(m[9].replace(",", "")), rpeak_pflops=float(m[10].replace(",", ""))))
        time.sleep(1)
    print(ym, len(rows)); assert len(rows) == 500, ym
    for r in rows: sites.setdefault(r["site_id"], None)
    with open(os.path.join(HERE, "data", f"top500_{ym.replace('/', '_')}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("unique sites", len(sites))
def seg(h):
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " | ", h)))
    m = re.search(r"Segment[ |]+([A-Za-z][A-Za-z ]*?) \|", t)
    return m[1].strip() if m else ""
with open(os.path.join(HERE, "data", "top500_site_segments.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["site_id", "segment"])
    for i, s in enumerate(sites):
        w.writerow([s, seg(get(f"https://www.top500.org/site/{s}/"))]); time.sleep(0.7)
        if i % 50 == 0: print("sites", i)
