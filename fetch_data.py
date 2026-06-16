#!/usr/bin/env python3
"""
fetch_data.py — assemble the country-level AI-inputs panel from public sources.

Inputs (real, public):
  data/ai_publications_cset_owid.csv  CSET/ETO Country Activity Tracker via Our World in Data
                                      -> AI knowledge OUTPUT (AI scholarly publications), latest year per country
  data/top500_2025_11.csv             TOP500 November 2025 per-country system counts (verified)
                                      -> COMPUTE input
  World Bank API (pulled live):
     SP.POP.SCIE.RD.P6  researchers in R&D per million          -> TALENT input (intensity)
     SP.POP.TOTL        population                              -> for absolute talent + per-capita compute
     NY.GDP.PCAP.PP.KD  GDP per capita, PPP (constant 2021 intl$) -> wealth control

Output: data/panel.csv  (one row per country with all variables + source years)

Every number traces to a primary source; no value is fabricated. See README and the
accompanying paper's facts file. Run:  python3 fetch_data.py
"""
import csv, os, json, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
AGG_EXCLUDE = {"OWID_WRL", "OWID_KOS"}

NAMES = {}  # iso3 -> country name, filled from top500 + pubs


def load_latest_output(fn="ai_publications_cset_owid.csv", valpart="AI scholarly publications",
                       min_year=2020):
    """Latest-year AI publications per ISO3 country (>= min_year)."""
    best = {}
    with open(os.path.join(DATA, fn)) as f:
        r = csv.DictReader(f)
        vc = [c for c in r.fieldnames if valpart in c][0]
        for row in r:
            code = (row.get("Code") or "").strip()
            if len(code) != 3 or code in AGG_EXCLUDE:
                continue
            try:
                y, v = int(row["Year"]), float(row[vc])
            except (ValueError, KeyError):
                continue
            if y < min_year:
                continue
            NAMES.setdefault(code, (row.get("Entity") or "").strip())
            if code not in best or y > best[code][0]:
                best[code] = (y, v)
    return best


def load_top500(fn="top500_2025_11.csv"):
    d = {}
    with open(os.path.join(DATA, fn)) as f:
        for row in csv.DictReader(f):
            d[row["iso3"]] = int(row["top500_systems"])
            NAMES.setdefault(row["iso3"], row["country"])
    return d


def wb(indicator, code):
    """Latest non-null World Bank value for an ISO3 country -> (year, value) or None."""
    url = (f"https://api.worldbank.org/v2/country/{code}/indicator/{indicator}"
           f"?format=json&per_page=90")
    try:
        d = json.load(urllib.request.urlopen(url, timeout=30))
        rows = [r for r in (d[1] or []) if r["value"] is not None] if len(d) > 1 else []
        if rows:
            return int(rows[0]["date"]), float(rows[0]["value"])
    except Exception as e:
        print("  WB error", indicator, code, e)
    return None


def main():
    output = load_latest_output()
    top500 = load_top500()
    print(f"AI publications: {len(output)} countries; TOP500: {len(top500)} countries")

    rows = []
    codes = sorted(output.keys())
    for i, c in enumerate(codes, 1):
        res = wb("SP.POP.SCIE.RD.P6", c)
        pop = wb("SP.POP.TOTL", c)
        gdp = wb("NY.GDP.PCAP.PP.KD", c)
        if res is None or pop is None:
            continue
        rpm = res[1]
        population = pop[1]
        total_res = rpm * population / 1e6
        systems = top500.get(c, 0)
        rows.append(dict(
            iso3=c, country=NAMES.get(c, c),
            ai_publications=output[c][1], pub_year=output[c][0],
            researchers_per_million=rpm, researchers_year=res[0],
            population=population,
            total_researchers=total_res,
            top500_systems=systems,
            gdp_pc_ppp=(gdp[1] if gdp else ""),
        ))
        if i % 25 == 0:
            print(f"  ...{i}/{len(codes)} countries processed")

    out = os.path.join(DATA, "panel.csv")
    cols = ["iso3", "country", "ai_publications", "pub_year",
            "researchers_per_million", "researchers_year", "population",
            "total_researchers", "top500_systems", "gdp_pc_ppp"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out} with n={len(rows)} countries.")
    kz = next((r for r in rows if r["iso3"] == "KAZ"), None)
    if kz:
        print("Kazakhstan:", {k: kz[k] for k in ("ai_publications", "researchers_per_million",
                                                 "top500_systems")})


if __name__ == "__main__":
    main()
