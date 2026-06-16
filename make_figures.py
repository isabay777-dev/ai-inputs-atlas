#!/usr/bin/env python3
"""
make_figures.py — figures for the AI-inputs analysis (300 dpi, white background).

Reads data/panel.csv and results/ (perm/shap importance). Writes figures/F1..F6.
Run after fetch_data.py, econometrics.py, ml_model.py.
"""
import os, csv, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(HERE, "results")

FOCAL = {"KAZ": "Kazakhstan", "USA": "United States", "CHN": "China",
         "DEU": "Germany", "NLD": "Netherlands", "CHE": "Switzerland", "POL": "Poland"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False,
})


def load():
    rows = []
    with open(os.path.join(HERE, "data", "panel.csv")) as f:
        for r in csv.DictReader(f):
            try:
                out = float(r["ai_publications"]); tot = float(r["total_researchers"])
            except ValueError:
                continue
            if out <= 0 or tot <= 0:
                continue
            rows.append(dict(iso3=r["iso3"], country=r["country"], output=out,
                             rpm=float(r["researchers_per_million"]),
                             pop=float(r["population"]),
                             systems=float(r["top500_systems"]), total_res=tot))
    return rows


def col(c): return "#c0392b" if c == "KAZ" else ("#2c3e50" if c in FOCAL else "#95a5a6")


def main():
    rows = load()
    by = {r["iso3"]: r for r in rows}

    # F1 compute (focal)
    order = sorted(FOCAL, key=lambda c: by.get(c, {}).get("systems", 0), reverse=True)
    vals = [by.get(c, {}).get("systems", 0) for c in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([FOCAL[c] for c in order], vals, color=[col(c) for c in order])
    ax.set_ylabel("TOP500 systems (Nov 2025)"); ax.set_title("AI compute input by country")
    for i, v in enumerate(vals): ax.text(i, v + 1, f"{int(v)}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "F1_compute.png")); plt.close()

    # F2 talent (focal)
    order = sorted(FOCAL, key=lambda c: by.get(c, {}).get("rpm", 0), reverse=True)
    vals = [by.get(c, {}).get("rpm", 0) for c in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([FOCAL[c] for c in order], vals, color=[col(c) for c in order])
    ax.set_ylabel("Researchers per million (World Bank)")
    ax.set_title("AI talent input by country")
    for i, v in enumerate(vals): ax.text(i, v + 80, f"{v:,.0f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "F2_talent.png")); plt.close()

    # F3 quadrant
    fig, ax = plt.subplots(figsize=(6.5, 6)); xs, ys = [], []
    for c in FOCAL:
        if c not in by: continue
        cpc = by[c]["systems"] / (by[c]["pop"] / 1e6); tal = by[c]["rpm"]
        xs.append(cpc); ys.append(tal)
        ax.scatter(cpc, tal, s=70, color=col(c), zorder=3)
        ax.annotate(FOCAL[c], (cpc, tal), xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.axhline(np.median(ys), color="gray", ls="--", lw=0.8)
    ax.axvline(np.median(xs), color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Compute per capita (TOP500 systems per million)")
    ax.set_ylabel("Talent (researchers per million)")
    ax.set_title("The compute x talent map")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F3_quadrant.png")); plt.close()

    # F4 input-output frontier (broad)
    xt = np.array([math.log(r["total_res"]) for r in rows])
    xc = np.array([math.log1p(r["systems"]) for r in rows])
    y = np.array([math.log(r["output"]) for r in rows])
    idx = (xt - xt.mean()) / xt.std() + (xc - xc.mean()) / xc.std()
    b1, b0 = np.polyfit(idx, y, 1)
    ss = 1 - np.sum((y - (b0 + b1 * idx))**2) / np.sum((y - y.mean())**2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, r in enumerate(rows):
        ax.scatter(idx[i], y[i], s=40, color=col(r["iso3"]), zorder=3,
                   alpha=0.85 if r["iso3"] in FOCAL else 0.5)
        if r["iso3"] in FOCAL:
            ax.annotate(r["iso3"], (idx[i], y[i]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
    xx = np.linspace(idx.min(), idx.max(), 50)
    ax.plot(xx, b0 + b1 * xx, color="#34495e", lw=1.5,
            label=f"fit: b={b1:.2f}, R2={ss:.2f}, n={len(rows)}")
    ax.set_xlabel("AI input index  z(talent)+z(compute)")
    ax.set_ylabel("log AI publications"); ax.legend()
    ax.set_title("AI inputs to AI output across countries")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F4_input_output.png")); plt.close()

    # F5 imbalance (focal percentile)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7, 4)); labels, cr, tr = [], [], []
    for c in FOCAL:
        if c not in by: continue
        labels.append(FOCAL[c])
        cr.append(100 * (1 - sum(1 for r in rows if r["systems"] > by[c]["systems"]) / n))
        tr.append(100 * (1 - sum(1 for r in rows if r["rpm"] > by[c]["rpm"]) / n))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, cr, w, label="Compute percentile (TOP500)", color="#2980b9")
    ax.bar(x + w/2, tr, w, label="Talent percentile (researchers/M)", color="#27ae60")
    ax.set_ylabel("Percentile within panel"); ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Compute-talent imbalance (gap = imbalance)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "F5_imbalance.png")); plt.close()

    # F6 ML feature importance (if available)
    pth = os.path.join(RES, "perm_importance.csv")
    if os.path.exists(pth):
        feats, imps = [], []
        with open(pth) as f:
            for r in csv.DictReader(f):
                feats.append(r["feature"]); imps.append(float(r["importance_mean"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        order = np.argsort(imps)
        ax.barh([feats[i] for i in order], [imps[i] for i in order], color="#8e44ad")
        ax.set_xlabel("Permutation importance (drop in R2)")
        ax.set_title("ML feature importance: which AI input drives output")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "F6_ml_importance.png")); plt.close()

    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
