#!/usr/bin/env python3
"""
robustness.py — robustness battery for the talent-vs-compute result.

Re-estimates the joint model under alternative specifications to test how stable the
two coefficients are. All models use heteroskedasticity-robust (HC1) standard errors.
Dependent variable: log AI publications.

Specifications:
  (R1) main joint                  log talent + log(1+systems)
  (R2) extensive margin            log talent + 1[has any TOP500 system]
  (R3) excluding US and China      drops the two largest producers
  (R4) + R&D intensity (GERD/GDP)  World Bank GB.XPD.RSDV.GD.ZS (latest)
  (R5) + income-group fixed effects World Bank income groups
  (R6) + region fixed effects      World Bank regions

Reads data/panel.csv and data/worldbank_gerd.csv; pulls income/region from the World
Bank API. Writes results/robustness.json. Run: python3 robustness.py
"""
import os, csv, json, math, urllib.request
import numpy as np
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)


def load():
    rows = {}
    with open(os.path.join(HERE, "data", "panel.csv")) as f:
        for r in csv.DictReader(f):
            try:
                out = float(r["ai_publications"]); tot = float(r["total_researchers"])
                sysn = float(r["top500_systems"])
            except ValueError:
                continue
            if out <= 0 or tot <= 0:
                continue
            rows[r["iso3"]] = dict(out=out, tot=tot, sysn=sysn)
    gerd = {}
    with open(os.path.join(HERE, "data", "worldbank_gerd.csv")) as f:
        for r in csv.DictReader(f):
            try:
                y = int(r["year"]); v = float(r["gerd_pct_gdp"])
            except ValueError:
                continue
            c = r["iso3"]
            if c not in gerd or y > gerd[c][0]:
                gerd[c] = (y, v)
    inc, reg = {}, {}
    d = json.load(urllib.request.urlopen(
        "https://api.worldbank.org/v2/country?format=json&per_page=400", timeout=30))
    for c in d[1]:
        inc[c["id"]] = c.get("incomeLevel", {}).get("id")
        reg[c["id"]] = c.get("region", {}).get("id")
    return rows, gerd, inc, reg


def fit(y, X, names, label):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1")
    rec = {"label": label, "n": int(m.nobs), "r2": round(float(m.rsquared), 3), "coef": {}}
    for nm, b, se, p in zip(["const"] + names, m.params, m.bse, m.pvalues):
        if nm in ("const",) or nm.startswith(("inc_", "reg_")):
            continue
        rec["coef"][nm] = {"b": round(float(b), 3), "se": round(float(se), 3),
                           "p": float(f"{p:.3g}")}
    print(f"{label}: n={rec['n']} R2={rec['r2']} " +
          " ".join(f"{k}={v['b']}(p={v['p']})" for k, v in rec["coef"].items()))
    return rec


def main():
    rows, gerd, inc, reg = load()
    codes = [c for c in rows if c in inc]
    g = lambda f: np.array([f(rows[c]) for c in codes])
    y = np.log(g(lambda r: r["out"]))
    lt = np.log(g(lambda r: r["tot"]))
    lc = np.log1p(g(lambda r: r["sysn"]))
    binc = (g(lambda r: r["sysn"]) > 0).astype(float)

    out = {"note": "HC1 robust SE; DV = log AI publications", "models": []}
    out["models"].append(fit(y, np.column_stack([lt, lc]), ["log_talent", "log_compute"], "R1 main joint"))
    out["models"].append(fit(y, np.column_stack([lt, binc]), ["log_talent", "compute_present"], "R2 extensive margin"))
    idx = np.array([i for i, c in enumerate(codes) if c not in ("USA", "CHN")])
    out["models"].append(fit(y[idx], np.column_stack([lt[idx], lc[idx]]), ["log_talent", "log_compute"], "R3 excl US+China"))
    gi = [i for i, c in enumerate(codes) if c in gerd]
    Xg = np.column_stack([lt[gi], lc[gi], np.array([gerd[codes[i]][1] for i in gi])])
    out["models"].append(fit(y[gi], Xg, ["log_talent", "log_compute", "gerd_pct_gdp"], "R4 + R&D intensity"))
    levels = sorted(set(inc[c] for c in codes))
    dum = np.array([[1.0 if inc[c] == L else 0.0 for L in levels[1:]] for c in codes])
    out["models"].append(fit(y, np.column_stack([lt, lc, dum]), ["log_talent", "log_compute"] + ["inc_" + L for L in levels[1:]], "R5 + income FE"))
    regs = sorted(set(reg[c] for c in codes if reg[c]))
    rd = np.array([[1.0 if reg[c] == R else 0.0 for R in regs[1:]] for c in codes])
    out["models"].append(fit(y, np.column_stack([lt, lc, rd]), ["log_talent", "log_compute"] + ["reg_" + R for R in regs[1:]], "R6 + region FE"))

    # R7: lagged compute from the June 2024 TOP500 list (predates the output window;
    # Kazakhstan had zero systems before its November 2025 debut). Source: TOP500 June 2024.
    jun24 = {"USA": 171, "CHN": 80, "DEU": 40, "FRA": 24, "JPN": 29, "ITA": 11, "KOR": 13,
             "NLD": 9, "POL": 8, "BRA": 8, "RUS": 7, "GBR": 16, "CHE": 5, "CAN": 10,
             "SAU": 7, "IND": 4, "KAZ": 0}
    lc24 = np.array([math.log1p(jun24.get(c, 0)) for c in codes])
    out["models"].append(fit(y, np.column_stack([lt, lc24]), ["log_talent", "log_compute_2024"], "R7 lagged compute (June 2024)"))

    with open(os.path.join(RES, "robustness.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote results/robustness.json")


if __name__ == "__main__":
    main()
