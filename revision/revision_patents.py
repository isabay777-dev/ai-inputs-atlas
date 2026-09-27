"""Step 8b (CEEJ-2026-0071.R1): AI patents vs AI publications on the same countries (R1 comment 3a).
Per specification_frozen.md sec. 5: y = ln(1 + AI patent applications), CSET Emerging Technology Observatory,
Country Activity Tracker v1.12.0 (Melot et al., 2026, doi:10.5281/zenodo.22772306), field 'All', rows flagged complete,
latest complete year >= 2020 (2021 for most countries; 2022+ are flagged incomplete); missing = excluded. Compute = ln(1+Rmax), TOP500 June 2024 (main); Nov 2023 = disclosed sensitivity.
Models M3 (T + C) and M6 (centred T x C). Difference in compute coefficient across outcomes: stacked regression,
country-clustered SEs. OLS HC1, t-based inference."""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
from revision_base import build
HERE = os.path.dirname(os.path.abspath(__file__))
pat = pd.read_csv(os.path.join(HERE, "data", "eto_cat", "cat", "patents_yearly_applications.csv"))
pat = pat[(pat.field == "All") & pat.complete & (pat.year >= 2020)].sort_values("year").groupby("country").tail(1)
pat = pat.rename(columns={"year": "pat_year", "num_patent_applications": "patents"})[["country", "pat_year", "patents"]]

def est(y, X):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1", use_t=True); ci = m.conf_int()
    return {"n": int(m.nobs), "r2": round(m.rsquared, 3), **{k: dict(b=round(m.params[k], 3), se=round(m.bse[k], 3),
            p=round(m.pvalues[k], 4), ci=[round(ci.loc[k, 0], 3), round(ci.loc[k, 1], 3)]) for k in X.columns}}

out = {}
for lst in ["2024_06", "2023_11"]:
    p = build(lst).merge(pat, on="country", how="inner")
    p["y_pat"] = np.log1p(p.patents)
    p["Tc"] = p["T"] - p["T"].mean(); p["Cc"] = p.C - p.C.mean(); p["TxC"] = p.Tc * p.Cc
    r = {"n": len(p), "n_hpc": int((p.systems > 0).sum()), "pat_years": p.pat_year.value_counts().to_dict()}
    for nm, y in [("patents", "y_pat"), ("publications", "y")]:
        r[nm + "_M3"] = est(p[y], p[["T", "C"]]); r[nm + "_M6"] = est(p[y], p[["Tc", "Cc", "TxC"]])
    # stacked: compute coefficient difference (patents - publications), clustered by country
    s = pd.concat([p.assign(out=p.y_pat, pat=1.0), p.assign(out=p.y, pat=0.0)])
    X = pd.DataFrame({"pat": s.pat, "T": s["T"], "C": s.C, "T_pat": s["T"] * s.pat, "C_pat": s.C * s.pat})
    m = sm.OLS(s.out, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": s.iso3}, use_t=True)
    r["diff_C_patents_minus_pubs"] = dict(b=round(m.params.C_pat, 3), se=round(m.bse.C_pat, 3), p=round(m.pvalues.C_pat, 4))
    out["main_jun2024" if lst == "2024_06" else "sens_nov2023"] = r
json.dump(out, open(os.path.join(HERE, "results", "results_patents.json"), "w"), indent=1, default=float)
for k, r in out.items():
    print(k, "n", r["n"], "hpc", r["n_hpc"])
    for nm in ["patents_M3", "publications_M3", "patents_M6", "publications_M6"]:
        print("  ", nm, {kk: (v["b"], v["p"]) for kk, v in r[nm].items() if isinstance(v, dict)}, "R2", r[nm]["r2"])
    print("   diff C:", r["diff_C_patents_minus_pubs"])
