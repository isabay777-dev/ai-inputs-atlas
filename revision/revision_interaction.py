"""Step 7 (CEEJ-2026-0071.R1): Talent x Compute interaction, marginal effects, talent quartiles.
Spec: specification_frozen.md sec. 3. Compute = ln(1+Rmax PFlop/s), TOP500 June 2024. OLS HC1."""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from revision_base import build
HERE = os.path.dirname(os.path.abspath(__file__))

p = build()
p["Tc"] = p["T"] - p["T"].mean(); p["Cc"] = p["C"] - p["C"].mean(); p["TxC"] = p.Tc * p.Cc
m = sm.OLS(p.y, sm.add_constant(p[["Tc", "Cc", "TxC"]])).fit(cov_type="HC1", use_t=True)
V = m.cov_params(); df = m.df_resid; tcrit = stats.t.ppf(0.975, df)
res = {"M6": {"n": int(m.nobs), "r2": round(m.rsquared, 3),
              **{k: dict(b=round(m.params[k], 3), se=round(m.bse[k], 3), p=round(m.pvalues[k], 4),
                         ci=[round(x, 3) for x in m.conf_int().loc[k]]) for k in ["Tc", "Cc", "TxC"]}}}

# marginal effect of C at talent percentiles: dy/dC = c + d*t, Var = Vcc + 2t Vcd + t^2 Vdd
me = []
for q in [10, 25, 50, 75, 90]:
    t = np.percentile(p["T"], q) - p["T"].mean()
    b = m.params.Cc + m.params.TxC * t
    se = np.sqrt(V.loc["Cc", "Cc"] + 2 * t * V.loc["Cc", "TxC"] + t * t * V.loc["TxC", "TxC"])
    me.append(dict(pct=q, researchers=int(round(np.exp(np.percentile(p["T"], q)))),
                   me=round(b, 3), se=round(se, 3), ci=[round(b - tcrit * se, 3), round(b + tcrit * se, 3)]))
res["marginal_effects"] = me

# figure: marginal effect over observed talent range, with rug of HPC countries
grid = np.linspace(p["T"].min(), p["T"].max(), 200); tg = grid - p["T"].mean()
b = m.params.Cc + m.params.TxC * tg
se = np.sqrt(V.loc["Cc", "Cc"] + 2 * tg * V.loc["Cc", "TxC"] + tg ** 2 * V.loc["TxC", "TxC"])
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
ax.fill_between(grid, b - tcrit * se, b + tcrit * se, color="0.85", label="95% confidence interval")
ax.plot(grid, b, color="black", lw=1.5, label="Marginal association of compute")
ax.axhline(0, color="0.4", lw=0.8, ls="--")
ax.plot(p.loc[p.systems > 0, "T"], np.full((p.systems > 0).sum(), ax.get_ylim()[0]), "|", color="#2166ac", ms=10,
        label="Economies with TOP500 systems")
ax.plot(p.loc[p.systems == 0, "T"], np.full((p.systems == 0).sum(), ax.get_ylim()[0]), "|", color="0.6", ms=6,
        label="Economies without TOP500 systems")
ax.set_xlabel("ln(total researchers)"); ax.set_ylabel("∂ ln(AI publications) / ∂ ln(1 + Rmax)")
ax.legend(fontsize=7, frameon=False, loc="upper left"); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig_marginal_effect_draft.png")); plt.close(fig)

# talent quartiles (R1)
p["Q"] = pd.qcut(p["T"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
rows = []
for q, g in p.groupby("Q", observed=True):
    r = dict(quartile=str(q), n=len(g), n_hpc=int((g.systems > 0).sum()),
             researchers_range=[int(np.exp(g["T"].min())), int(np.exp(g["T"].max()))])
    if r["n_hpc"] >= 5:
        mq = sm.OLS(g.y, sm.add_constant(g[["T", "C"]])).fit(cov_type="HC1", use_t=True)
        r.update(b_C=round(mq.params.C, 3), se_C=round(mq.bse.C, 3), p_C=round(mq.pvalues.C, 4),
                 ci_C=[round(x, 3) for x in mq.conf_int().loc["C"]])
    else:
        r["b_C"] = "not estimable: insufficient support"
    rows.append(r)
res["quartiles"] = rows
json.dump(res, open(os.path.join(HERE, "results", "results_interaction.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
