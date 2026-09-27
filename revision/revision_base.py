"""Revision (CEEJ-2026-0071.R1): baseline models M1-M5 per specification_frozen.md.
Compute = ln(1 + Rmax, PFlop/s), TOP500 June 2024 list (full list, data_revision/top500_2024_06.csv). OLS with HC1."""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "data", "panel.csv")

AGG = os.path.join(HERE, "data", "top500_country_aggregates.csv")

def compute(list_id, col="rmax_pflops", segments=None):
    raw = os.path.join(HERE, "data", f"top500_{list_id}.csv")
    if not os.path.exists(raw):  # system-level lists are not redistributed; fall back to derived country aggregates
        a = pd.read_csv(AGG, dtype={"list": str}); a = a[a.list == list_id].set_index("country")
        if segments:
            assert set(segments) == {"Academic", "Research"}
            return a.rename(columns={"systems_acad_research": "n", "rmax_acad_research": "v"})[["n", "v"]].rename(columns={"n": "systems", "v": "val"})
        return a.rename(columns={col: "val"})[["systems", "val"]]
    t = pd.read_csv(os.path.join(HERE, "data", f"top500_{list_id}.csv"), dtype={"site_id": str})
    if segments:
        seg = pd.read_csv(os.path.join(HERE, "data", "top500_site_segments.csv"), dtype=str)
        t = t.merge(seg, on="site_id")
        t = t[t.segment.isin(segments)]
    return t.groupby("country").agg(systems=("rank", "size"), val=(col, "sum"))

def build(list_id="2024_06", col="rmax_pflops", segments=None):
    p = pd.read_csv(PANEL)
    c = compute(list_id, col, segments)
    p = p.merge(c, left_on="country", right_index=True, how="left").fillna({"systems": 0, "val": 0.0})
    p["y"] = np.log(p.ai_publications)
    p["T"] = np.log(p.total_researchers)
    p["C"] = np.log1p(p.val)
    p["lgdp"] = np.log(p.gdp_pc_ppp)
    p["y_pc"] = np.log(p.ai_publications / (p.population / 1e6))
    p["T_pc"] = np.log(p.researchers_per_million)
    p["C_pc"] = np.log1p(p.val / (p.population / 1e6))
    return p

def fit(d, y, xs):
    d = d.dropna(subset=[y] + xs)
    m = sm.OLS(d[y], sm.add_constant(d[xs])).fit(cov_type="HC1", use_t=True)
    ci = m.conf_int()
    out = {"n": int(m.nobs), "r2": round(m.rsquared, 3)}
    for x in xs:
        out[x] = dict(b=round(m.params[x], 3), se=round(m.bse[x], 3), p=round(m.pvalues[x], 4),
                      ci=[round(ci.loc[x, 0], 3), round(ci.loc[x, 1], 3)])
    return m, out

if __name__ == "__main__":
    p = build()
    res = {}
    for name, y, xs in [("M1", "y", ["T"]), ("M2", "y", ["C"]), ("M3", "y", ["T", "C"]),
                        ("M4", "y", ["T", "C", "lgdp"]), ("M5", "y_pc", ["T_pc", "C_pc"])]:
        m, res[name] = fit(p, y, xs)
        if name == "M3":
            t = m.t_test("T = 1")
            res[name]["H0_T_eq_1"] = dict(t=round(float(t.tvalue.item()), 3), p=round(float(t.pvalue), 4))
    res["countries_with_hpc"] = int((p.systems > 0).sum())
    json.dump(res, open(os.path.join(HERE, "results", "results_base.json"), "w"), indent=1)
    for k, v in res.items(): print(k, v)
