#!/usr/bin/env python3
"""
econometrics.py — the transparent OLS core reported in the paper.

Knowledge-production-function style: log(AI output) on log(talent) and log(compute),
across the broad country panel. Tests:
  H1  AI output scales with inputs (talent; and a combined input index).
  H2  Talent is the binding input: once talent is controlled, compute adds little
      (compare standardized betas / significance).
  H3  Kazakhstan's input imbalance: compute rank >> talent rank.

Reads data/panel.csv (from fetch_data.py). Writes results/ols_results.json.
Run: python3 econometrics.py
"""
import os, csv, json, math
import numpy as np
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def load_panel():
    rows = []
    with open(os.path.join(HERE, "data", "panel.csv")) as f:
        for r in csv.DictReader(f):
            try:
                out = float(r["ai_publications"])
                tot = float(r["total_researchers"])
            except ValueError:
                continue
            if out <= 0 or tot <= 0:
                continue
            gdp = r.get("gdp_pc_ppp") or ""
            rows.append(dict(
                iso3=r["iso3"], country=r["country"],
                output=out, total_res=tot,
                rpm=float(r["researchers_per_million"]),
                systems=float(r["top500_systems"]),
                gdp=(float(gdp) if gdp else None),
            ))
    return rows


def ols(y, X, names):
    m = sm.OLS(y, sm.add_constant(X)).fit()
    return dict(n=int(m.nobs), r2=round(float(m.rsquared), 4),
               adj_r2=round(float(m.rsquared_adj), 4),
               coef={nm: round(float(b), 4) for nm, b in zip(["const"] + names, m.params)},
               p={nm: float(f"{p:.3g}") for nm, p in zip(["const"] + names, m.pvalues)}), m


def main():
    rows = load_panel()
    y = np.array([math.log(r["output"]) for r in rows])
    xt = np.array([math.log(r["total_res"]) for r in rows])
    xc = np.array([math.log1p(r["systems"]) for r in rows])

    out = {"n_countries": len(rows)}
    rA, _ = ols(y, xt, ["log_talent"]);                         out["H1_talent"] = rA
    rB, _ = ols(y, xc, ["log_compute"]);                        out["compute_only"] = rB
    rC, mC = ols(y, np.column_stack([xt, xc]), ["log_talent", "log_compute"])
    out["H2_joint"] = rC

    zt = (xt - xt.mean()) / xt.std()
    zc = (xc - xc.mean()) / xc.std()
    zy = (y - y.mean()) / y.std()
    ms = sm.OLS(zy, sm.add_constant(np.column_stack([zt, zc]))).fit()
    out["H2_standardized_betas"] = {"talent": round(float(ms.params[1]), 4),
                                    "compute": round(float(ms.params[2]), 4),
                                    "talent_p": float(f"{ms.pvalues[1]:.3g}"),
                                    "compute_p": float(f"{ms.pvalues[2]:.3g}")}

    # optional GDP-controlled robustness (talent should survive)
    g = [(i, r) for i, r in enumerate(rows) if r["gdp"]]
    if len(g) > 20:
        idx = [i for i, _ in g]
        yg = y[idx]
        Xg = np.column_stack([xt[idx], xc[idx],
                              np.log([rows[i]["gdp"] for i in idx])])
        rG, _ = ols(yg, Xg, ["log_talent", "log_compute", "log_gdp_pc"])
        out["H2_gdp_controlled"] = rG

    index = zt + zc
    rIdx, mIdx = ols(y, index, ["input_index"]); out["H1_input_index"] = rIdx

    # H3: Kazakhstan residual + ranks
    kz = next((i for i, r in enumerate(rows) if r["iso3"] == "KAZ"), None)
    if kz is not None:
        pred = float(mC.predict(sm.add_constant(np.column_stack([xt, xc])))[kz])
        comp_rank = 1 + sum(1 for r in rows if r["systems"] > rows[kz]["systems"])
        tal_rank = 1 + sum(1 for r in rows if r["rpm"] > rows[kz]["rpm"])
        out["H3_kazakhstan"] = dict(
            actual_log_output=round(float(y[kz]), 4),
            predicted_log_output=round(pred, 4),
            residual=round(float(y[kz]) - pred, 4),
            compute_rank=comp_rank, talent_rank=tal_rank, n=len(rows),
            compute_percentile=round(100 * (1 - (comp_rank - 1) / len(rows)), 1),
            talent_percentile=round(100 * (1 - (tal_rank - 1) / len(rows)), 1),
        )

    with open(os.path.join(RES, "ols_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nWrote results/ols_results.json")


if __name__ == "__main__":
    main()
