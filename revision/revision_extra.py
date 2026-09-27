"""Additional statistics reported in the revised manuscript: standardised and index models, input correlation and VIF
(Table 2), joint model on the 131-economy GDP sample (Section 4.2), and sensitivity of the patent interaction (Table B4)."""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
from revision_base import build
HERE = os.path.dirname(os.path.abspath(__file__))
p = build(); z = lambda s: (s - s.mean()) / s.std()
d = p.assign(Tz=z(p["T"]), Cz=z(p.C), yz=z(p.y)); d["idx"] = d.Tz + d.Cz
ms = sm.OLS(d.yz, sm.add_constant(d[["Tz", "Cz"]])).fit(cov_type="HC1", use_t=True)
mi = sm.OLS(d.y, sm.add_constant(d[["idx"]])).fit(cov_type="HC1", use_t=True)
g = p.dropna(subset=["lgdp"]); mg = sm.OLS(g.y, sm.add_constant(g[["T", "C"]])).fit(cov_type="HC1", use_t=True)
r = np.corrcoef(p["T"], p.C)[0, 1]
extra = {"std_T": round(ms.params.Tz, 3), "std_C": round(ms.params.Cz, 3), "idx_b": round(mi.params.idx, 3), "idx_r2": round(mi.rsquared, 3),
         "corr_TC": round(r, 3), "VIF": round(1 / (1 - r**2), 2),
         "joint_131": {"C": round(mg.params.C, 3), "ci": [round(x, 3) for x in mg.conf_int().loc["C"]], "p": round(mg.pvalues.C, 3)}}
json.dump(extra, open(os.path.join(HERE, "results", "results_extra.json"), "w"), indent=1, default=float)
pat = pd.read_csv(os.path.join(HERE, "data", "eto_cat", "cat", "patents_yearly_applications.csv"))
pat = pat[(pat.field == "All") & pat.complete & (pat.year >= 2020)].sort_values("year").groupby("country").tail(1)[["country", "num_patent_applications"]]
out = {}
for lst, cases in [("2024_06", [("all", []), ("excl USA,CHN", ["USA", "CHN"]), ("excl LUX", ["LUX"]), ("excl USA,CHN,JPN,KOR", ["USA", "CHN", "JPN", "KOR"])]), ("2023_11", [("all", [])])]:
    q = build(lst).merge(pat, on="country")
    for lab, ex in cases:
        e = q[~q.iso3.isin(ex)].copy(); e["y"] = np.log1p(e.num_patent_applications)
        e["Tc"] = e["T"] - e["T"].mean(); e["Cc"] = e.C - e.C.mean(); e["TxC"] = e.Tc * e.Cc
        m = sm.OLS(e.y, sm.add_constant(e[["Tc", "Cc", "TxC"]])).fit(cov_type="HC1", use_t=True)
        out[f"{lst} {lab}"] = dict(n=len(e), b=round(m.params.TxC, 3), se=round(m.bse.TxC, 3), p=round(m.pvalues.TxC, 4))
json.dump(out, open(os.path.join(HERE, "results", "results_patents_robustness.json"), "w"), indent=1, default=float)
print(extra); print(out)
