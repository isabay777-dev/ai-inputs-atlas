#!/usr/bin/env python3
"""Automated smoke tests: the pipeline runs and the core findings hold.
Run: python3 test_smoke.py   (exits non-zero on any failure)

Covers: (1) panel.csv loads, has Kazakhstan, and contains the verified KZ inputs;
(2) OLS talent coefficient is positive and significant (H1); (3) in the joint model
talent dominates compute (H2); (4) the ML model corroborates that talent (not compute)
is the top feature; (5) Kazakhstan's compute rank is far above its talent rank (H3).
"""
import sys, os, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        check.failed += 1
check.failed = 0

# (1) panel
panel = os.path.join(HERE, "data", "panel.csv")
if os.path.exists(panel):
    rows = list(csv.DictReader(open(panel)))
    kz = next((r for r in rows if r["iso3"] == "KAZ"), None)
    check("panel.csv has >= 80 countries", len(rows) >= 80)
    check("Kazakhstan present in panel", kz is not None)
    if kz:
        check("KZ talent ~758 researchers/million (verified)",
              700 <= float(kz["researchers_per_million"]) <= 820)
        check("KZ TOP500 systems == 2 (verified)", int(float(kz["top500_systems"])) == 2)
else:
    print("SKIP panel checks (run fetch_data.py first)")

# (2,3) OLS results
ols = os.path.join(HERE, "results", "ols_results.json")
if os.path.exists(ols):
    o = json.load(open(ols))
    check("H1: talent coefficient positive", o["H1_talent"]["coef"]["log_talent"] > 0)
    check("H1: talent significant (p<0.001)", o["H1_talent"]["p"]["log_talent"] < 1e-3)
    sb = o["H2_standardized_betas"]
    check("H2: talent beta > compute beta (talent binds)", sb["talent"] > sb["compute"])
    h3 = o.get("H3_kazakhstan", {})
    if h3:
        check("H3: KZ compute rank stronger than talent rank",
              h3["compute_rank"] < h3["talent_rank"])
else:
    print("SKIP OLS checks (run econometrics.py first)")

# (4) ML corroboration
mt = os.path.join(HERE, "results", "metrics.json")
if os.path.exists(mt):
    m = json.load(open(mt))
    check("ML: cross-validated R2 > 0.6", m["cv_r2"] > 0.6)
    if "top_feature" in m:
        check("ML: top feature is talent-related",
              "talent" in m["top_feature"] or "researchers" in m["top_feature"])
else:
    print("SKIP ML checks (run ml_model.py first)")

print("\n%d failure(s)" % check.failed)
sys.exit(1 if check.failed else 0)
