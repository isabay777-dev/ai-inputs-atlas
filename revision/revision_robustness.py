"""Step 8a (CEEJ-2026-0071.R1): robustness table R1-R14 per specification_frozen.md sec. 4 (+R12-R14 from review).
Baseline M3: y = a + b*T + c*C, C = ln(1+Rmax PFlop/s), TOP500 June 2024. OLS HC1."""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
from revision_base import build
HERE = os.path.dirname(os.path.abspath(__file__))
GERD = os.path.join(HERE, "..", "data", "worldbank_gerd.csv")

def ols(d, xs, fe=None, label=""):
    d = d.dropna(subset=["y"] + xs).copy()
    X = d[xs]
    if fe:
        X = pd.concat([X, pd.get_dummies(d[fe], prefix=fe, drop_first=True, dtype=float)], axis=1)
    m = sm.OLS(d["y"], sm.add_constant(X)).fit(cov_type="HC1", use_t=True)
    ci = m.conf_int()
    out = {"label": label, "n": int(m.nobs), "r2": round(m.rsquared, 3)}
    for x in xs:
        out[x] = dict(b=round(m.params[x], 3), se=round(m.bse[x], 3), p=round(m.pvalues[x], 4),
                      ci=[round(ci.loc[x, 0], 3), round(ci.loc[x, 1], 3)])
    return out

base = build("2024_06")
meta = pd.read_csv(os.path.join(HERE, "data", "wb_country_meta.csv"))
base = base.merge(meta, on="iso3", how="left")
g = pd.read_csv(GERD).sort_values("year").groupby("iso3").tail(1)[["iso3", "gerd_pct_gdp"]]
base = base.merge(g, on="iso3", how="left")
base["D"] = (base.systems > 0).astype(float)

def swap(p, **kw):
    q = build(**kw)[["iso3", "C", "systems", "val"]]
    return p.drop(columns=["C", "systems", "val"]).merge(q, on="iso3")

R = [ols(base, ["T", "C"], label="Main (M3): ln(1+Rmax), Jun 2024")]
d = base.copy(); d["C"] = d.D; R.append(ols(d, ["T", "C"], label="R1 Any TOP500 system (Jun 2024)"))
d = base.copy(); d["C"] = np.log1p(d.systems); R.append(ols(d, ["T", "C"], label="R2 ln(1+systems), Jun 2024"))
R.append(ols(swap(base, list_id="2024_06", col="rpeak_pflops"), ["T", "C"], label="R3 ln(1+Rpeak), Jun 2024"))
R.append(ols(swap(base, list_id="2023_11"), ["T", "C"], label="R4 Rmax, Nov 2023 list"))
R.append(ols(swap(base, list_id="2025_11"), ["T", "C"], label="R5 Rmax, Nov 2025 list"))
R.append(ols(swap(base, list_id="2024_06", segments=["Academic", "Research"]), ["T", "C"], label="R6 Rmax, Academic+Research sites"))
R.append(ols(base[~base.iso3.isin(["USA", "CHN"])], ["T", "C"], label="R7 Excluding US and China"))
R.append(ols(base, ["T", "C", "gerd_pct_gdp"], label="R8 + R&D intensity (GERD % GDP)"))
R.append(ols(base, ["T", "C"], fe="income", label="R9 + income-group FE"))
R.append(ols(base, ["T", "C"], fe="region", label="R10 + region FE"))
R.append(ols(base[base.researchers_year >= 2015], ["T", "C"], label="R11 Researcher data year >= 2015"))
R.append(ols(base[base.researchers_year >= 2022], ["T", "C"], label="R11b Researcher data year 2022-2024"))
R.append(ols(base[base.researchers_year >= 2022], ["T", "C", "lgdp"], label="R11c Researcher data year 2022-2024, + GDP per capita"))
d = base.copy(); d["lnRmax_if"] = np.where(d.D > 0, np.log(d.val.where(d.val > 0, 1)), 0.0)
R.append(ols(d, ["T", "D", "lnRmax_if"], label="R12 Two-part: any system + ln(Rmax) if any"))
d = base[~base.iso3.isin(["USA", "CHN"])].copy()
d["Tc"] = d["T"] - d["T"].mean(); d["Cc"] = d.C - d.C.mean(); d["TxC"] = d.Tc * d.Cc
R.append(ols(d, ["Tc", "Cc", "TxC"], label="R13 Interaction, excl. US and China"))
d = base.copy(); d["Tc"] = d["T"] - d["T"].mean(); d["DxT"] = d.D * d.Tc
R.append(ols(d, ["Tc", "D", "DxT"], label="R14 Any system x talent"))
json.dump(R, open(os.path.join(HERE, "results", "results_robustness.json"), "w"), indent=1, default=float)
for r in R:
    ks = [k for k in r if k not in ("label", "n", "r2")]
    print(f"{r['label']:<48} n={r['n']:<4} R2={r['r2']:.3f}  " +
          "  ".join(f"{k}={r[k]['b']:.3f}({r[k]['se']:.3f})p={r[k]['p']:.3f}" for k in ks))
