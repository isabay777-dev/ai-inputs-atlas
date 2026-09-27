"""Figures 1-7 for CEEJ-2026-0071.R1, built from the revised data (full TOP500 lists). Style as in the first version."""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
from revision_base import build
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 10, "axes.facecolor": "white",
                     "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
FOCAL = ["USA", "CHN", "DEU", "NLD", "CHE", "POL", "KAZ"]
NAMES = {"USA": "United States", "CHN": "China", "DEU": "Germany", "NLD": "Netherlands", "CHE": "Switzerland", "POL": "Poland", "KAZ": "Kazakhstan"}
col = lambda c: "#c0392b" if c == "KAZ" else ("#2c3e50" if c in FOCAL else "#95a5a6")
save = lambda name: (plt.tight_layout(), plt.savefig(os.path.join(FIG, name)), plt.close())

n25, j24 = build("2025_11"), build("2024_06")
f25 = n25[n25.iso3.isin(FOCAL)].set_index("iso3").loc[FOCAL]

# Fig 1: TOP500 systems and aggregate Rmax, Nov 2025, focal economies
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
o = f25.sort_values("systems", ascending=False)
ax[0].bar([NAMES[c] for c in o.index], o.systems, color=[col(c) for c in o.index]); ax[0].set_ylabel("TOP500 systems")
ax[0].set_title("Number of systems", fontsize=10)
o = f25.sort_values("val", ascending=False)
ax[1].bar([NAMES[c] for c in o.index], o.val, color=[col(c) for c in o.index]); ax[1].set_yscale("log"); ax[1].set_ylabel("Aggregate Rmax, PFlop/s (log scale)")
ax[1].set_title("Aggregate performance", fontsize=10)
for a in ax: a.tick_params(axis="x", rotation=45); [t.set_ha("right") for t in a.get_xticklabels()]
save("fig1_compute_top500.png")

# Fig 2: researchers per million, focal economies (unchanged content)
o = f25.sort_values("researchers_per_million", ascending=False)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar([NAMES[c] for c in o.index], o.researchers_per_million, color=[col(c) for c in o.index])
ax.set_ylabel("Researchers per million inhabitants"); ax.tick_params(axis="x", rotation=45); [t.set_ha("right") for t in ax.get_xticklabels()]
save("fig2_talent_researchers.png")

# Fig 3: compute per inhabitant vs researchers per million, all 136 (Nov 2025)
d = n25.assign(cpc=np.log1p(n25.val / (n25.population / 1e6)), tpc=np.log(n25.researchers_per_million))
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.scatter(d.tpc, d.cpc, s=14, c=[col(c) for c in d.iso3], alpha=0.8, zorder=2)
OFF3 = {"USA": (-60, 6), "DEU": (6, 4), "NLD": (6, -10), "CHE": (6, 2), "POL": (6, -2), "CHN": (6, -8), "KAZ": (-52, 4)}
for c in FOCAL:
    r = d[d.iso3 == c].iloc[0]; ax.annotate(NAMES[c], (r.tpc, r.cpc), xytext=OFF3[c], textcoords="offset points", fontsize=8, color=col(c),
                                            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
ax.set_xlabel("ln(researchers per million)"); ax.set_ylabel("ln(1 + Rmax per million inhabitants, PFlop/s)")
save("fig3_quadrant.png")

# Fig 4: AI output vs combined input index (June 2024), fitted line
z = lambda s: (s - s.mean()) / s.std()
d = j24.assign(idx=z(j24["T"]) + z(j24.C)); m = sm.OLS(d.y, sm.add_constant(d.idx)).fit()
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.scatter(d.idx, d.y, s=14, c=[col(c) for c in d.iso3], alpha=0.8, zorder=2)
g = np.linspace(d.idx.min(), d.idx.max(), 50); ax.plot(g, m.params.const + m.params.idx * g, color="#2c3e50", lw=1.2)
OFF4 = {"USA": (-10, -16), "CHN": (-30, 8), "DEU": (6, 6), "NLD": (8, -4), "CHE": (8, -14), "POL": (8, -24), "KAZ": (8, -12)}
for c in FOCAL:
    r = d[d.iso3 == c].iloc[0]; ax.annotate(NAMES[c], (r.idx, r.y), xytext=OFF4[c], textcoords="offset points", fontsize=8, color=col(c),
                                            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
ax.set_xlabel("Combined input index, z(talent) + z(compute)"); ax.set_ylabel("ln(AI publications)")
ax.text(0.02, 0.97, f"n = {len(d)}; R² = {m.rsquared:.2f}", transform=ax.transAxes, va="top", fontsize=8)
save("fig4_input_output_frontier.png")

# Fig 5: like-for-like percentiles, focal economies (Nov 2025)
k = pd.read_csv(os.path.join(HERE, "results", "kazakhstan_positions.csv")).set_index("iso3").loc[FOCAL]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
x = np.arange(len(FOCAL)); w = 0.38
for a, (cc, tt, title) in zip(ax, [("pc_comp_int", "pc_tal_int", "Per inhabitant"), ("pc_comp_lvl", "pc_tal_lvl", "Totals")]):
    a.bar(x - w/2, k[cc], w, color="#2980b9", label="Compute (Rmax) percentile")
    a.bar(x + w/2, k[tt], w, color="#27ae60", label="Talent (researchers) percentile")
    a.set_xticks(x); a.set_xticklabels([NAMES[c] for c in FOCAL], rotation=45, ha="right"); a.set_title(title, fontsize=10)
ax[0].set_ylabel("Percentile among 136 economies"); ax[0].legend(fontsize=7, frameon=False, loc="lower left")
save("fig5_imbalance.png")

# Fig 6: ML permutation importance (held-out folds), mean and SD
ml = json.load(open(os.path.join(HERE, "results", "results_ml.json")))
lab = {"T": "Research capacity\n(ln total researchers)", "C": "Compute\n(ln(1 + Rmax))", "lgdp": "ln GDP per capita"}
pi = ml["perm_importance_mean_sd"]; keys = sorted(pi, key=lambda k: pi[k][0])
fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.barh([lab[k] for k in keys], [pi[k][0] for k in keys], xerr=[pi[k][1] for k in keys], color=["#95a5a6" if k != "T" else "#2c3e50" for k in keys], capsize=3)
ax.set_xlabel("Permutation importance on held-out folds (drop in R²)")
save("fig6_ml_importance.png")

# Fig 7: marginal association of compute by research capacity, unsupported region shaded
p = j24.copy(); p["Tc"] = p["T"] - p["T"].mean(); p["Cc"] = p.C - p.C.mean(); p["TxC"] = p.Tc * p.Cc
m = sm.OLS(p.y, sm.add_constant(p[["Tc", "Cc", "TxC"]])).fit(cov_type="HC1", use_t=True); V = m.cov_params(); tc = stats.t.ppf(0.975, m.df_resid)
grid = np.linspace(p["T"].min(), p["T"].max(), 200); t = grid - p["T"].mean()
b = m.params.Cc + m.params.TxC * t; se = np.sqrt(V.loc["Cc", "Cc"] + 2 * t * V.loc["Cc", "TxC"] + t**2 * V.loc["TxC", "TxC"])
lo = p.loc[p.systems > 0, "T"].min()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axvspan(p["T"].min(), lo, color="0.93", zorder=0, label="No economy with a listed system (extrapolation)")
ax.fill_between(grid, b - tc*se, b + tc*se, color="0.80", label="95% confidence interval")
ax.plot(grid[grid >= lo], b[grid >= lo], color="black", lw=1.5, label="Marginal association of compute")
ax.plot(grid[grid < lo], b[grid < lo], color="black", lw=1.2, ls="--")
ax.axhline(0, color="0.4", lw=0.8, ls=":")
y0 = ax.get_ylim()[0]
ax.plot(p.loc[p.systems > 0, "T"], np.full((p.systems > 0).sum(), y0), "|", color="#2166ac", ms=10, label="Economies with TOP500 systems")
ax.plot(p.loc[p.systems == 0, "T"], np.full((p.systems == 0).sum(), y0), "|", color="0.6", ms=6, label="Economies without TOP500 systems")
ax.set_xlabel("ln(total researchers)"); ax.set_ylabel("∂ ln(AI publications) / ∂ ln(1 + Rmax)")
ax.legend(fontsize=7, frameon=False, loc="upper right")
save("fig7_marginal_effect.png")
print("figures written; support boundary ln T =", round(lo, 3), "=", int(round(np.exp(lo))), "researchers")
