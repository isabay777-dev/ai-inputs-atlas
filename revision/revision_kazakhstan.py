"""Step 8d (CEEJ-2026-0071.R1): Kazakhstan descriptive case per specification_frozen.md sec. 6.
Compute from TOP500 Nov 2025 (Kazakhstan's first listing) for ALL economies; described as the current input
configuration, not as an explanation of 2024 output. Percentile ranks among the 136 economies (ties = average rank).
Imbalance = compute percentile - talent percentile, on like-for-like measures:
  intensity: Rmax per million inhabitants vs researchers per million;  level: total Rmax vs total researchers.
Comparison groups fixed ex ante: (a) the six original benchmarks; (b) all upper-middle-income economies (WB) and all
former-Soviet economies in the sample."""
import os, json
import pandas as pd
from revision_base import build
HERE = os.path.dirname(os.path.abspath(__file__))
FSU = ["ARM", "AZE", "BLR", "EST", "GEO", "KAZ", "KGZ", "LVA", "LTU", "MDA", "RUS", "TJK", "TKM", "UKR", "UZB"]
BENCH = ["USA", "CHN", "DEU", "NLD", "CHE", "POL"]
p = build("2025_11").merge(pd.read_csv(os.path.join(HERE, "data", "wb_country_meta.csv")), on="iso3", how="left")
p["rmax_pc"] = p.val / (p.population / 1e6)
pct = lambda s: (s.rank(method="average", pct=True) * 100).round(1)
p["pc_comp_int"], p["pc_tal_int"] = pct(p.rmax_pc), pct(p.researchers_per_million)
p["pc_comp_lvl"], p["pc_tal_lvl"] = pct(p.val), pct(p.total_researchers)
p["gap_int"], p["gap_lvl"] = p.pc_comp_int - p.pc_tal_int, p.pc_comp_lvl - p.pc_tal_lvl
p["rank_comp_lvl"] = p.val.rank(ascending=False, method="min").astype(int)
p["rank_tal_lvl"] = p.total_researchers.rank(ascending=False, method="min").astype(int)
cols = ["iso3", "country", "systems", "val", "rmax_pc", "researchers_per_million", "pc_comp_int", "pc_tal_int", "gap_int",
        "pc_comp_lvl", "pc_tal_lvl", "gap_lvl", "rank_comp_lvl", "rank_tal_lvl"]
grp_b = p[(p.income == "UMC") | p.iso3.isin(FSU)].copy()
res = {"n_all": len(p), "n_hpc_nov2025": int((p.systems > 0).sum()),
       "kazakhstan": p[p.iso3 == "KAZ"][cols].round(2).to_dict("records")[0],
       "benchmarks": p[p.iso3.isin(BENCH + ["KAZ"])][cols].round(2).to_dict("records"),
       "group_b": {"n": len(grp_b), "n_with_hpc": int((grp_b.systems > 0).sum()),
                   "kaz_rank_gap_int": int(grp_b.gap_int.rank(ascending=False, method="min")[grp_b.iso3 == "KAZ"].iloc[0]),
                   "kaz_rank_gap_lvl": int(grp_b.gap_lvl.rank(ascending=False, method="min")[grp_b.iso3 == "KAZ"].iloc[0]),
                   "top5_gap_int": grp_b.nlargest(5, "gap_int")[["iso3", "systems", "gap_int"]].round(1).values.tolist(),
                   "members_with_hpc": grp_b[grp_b.systems > 0][["iso3", "systems", "gap_int", "gap_lvl"]].round(1).values.tolist()},
       "all_positive_gap_int": int((p.gap_int > 0).sum())}
h = p[p.systems > 0]
res["holders"] = {"n": len(h),
                  "kaz_rank_gap_int": int(h.gap_int.rank(ascending=False, method="min")[h.iso3 == "KAZ"].iloc[0]),
                  "kaz_rank_gap_lvl": int(h.gap_lvl.rank(ascending=False, method="min")[h.iso3 == "KAZ"].iloc[0])}
hb = grp_b[grp_b.systems > 0]
res["group_b_holders"] = {"n": len(hb),
                          "kaz_rank_gap_int": int(hb.gap_int.rank(ascending=False, method="min")[hb.iso3 == "KAZ"].iloc[0]),
                          "kaz_rank_gap_lvl": int(hb.gap_lvl.rank(ascending=False, method="min")[hb.iso3 == "KAZ"].iloc[0])}
p[cols + ["income"]].to_csv(os.path.join(HERE, "results", "kazakhstan_positions.csv"), index=False)
json.dump(res, open(os.path.join(HERE, "results", "results_kazakhstan.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float, ensure_ascii=False))
