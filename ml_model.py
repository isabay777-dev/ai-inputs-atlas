#!/usr/bin/env python3
"""
ml_model.py — machine-learning triangulation of the AI-inputs -> AI-output link.

A non-parametric check on the OLS result: can a gradient-boosting model predict a
country's AI knowledge output from its AI inputs, and which input does it rely on?
If talent dominates compute in a model that makes no linearity assumption, the OLS
finding (talent is the binding input, H2) is not an artifact of functional form.

Pipeline:
  features: log_talent (total researchers), log_compute (1+TOP500 systems),
            researchers_per_million, log_gdp_pc_ppp  [wealth control]
  target:   log(AI publications)
  model:    HistGradientBoostingRegressor
  validation: repeated K-fold CV (R2, MAE), out-of-fold predictions
  importance: permutation importance (CV) + SHAP (TreeExplainer)

Reads data/panel.csv. Writes results/metrics.json, results/perm_importance.csv,
results/shap_importance.csv. Fixed seed for reproducibility.
Run: python3 ml_model.py
"""
import os, csv, json, math
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_val_predict, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

FEATURES = ["log_talent", "log_compute", "researchers_per_million", "log_gdp_pc"]


def load_xy():
    X, y, codes = [], [], []
    with open(os.path.join(HERE, "data", "panel.csv")) as f:
        for r in csv.DictReader(f):
            try:
                out = float(r["ai_publications"]); tot = float(r["total_researchers"])
                rpm = float(r["researchers_per_million"]); sysn = float(r["top500_systems"])
                gdp = r.get("gdp_pc_ppp") or ""
            except ValueError:
                continue
            if out <= 0 or tot <= 0 or not gdp:
                continue
            X.append([math.log(tot), math.log1p(sysn), rpm, math.log(float(gdp))])
            y.append(math.log(out)); codes.append(r["iso3"])
    return np.array(X), np.array(y), codes


def main():
    X, y, codes = load_xy()
    print(f"ML panel: n={len(y)} countries, {X.shape[1]} features")

    model = HistGradientBoostingRegressor(random_state=SEED, max_depth=3,
                                          learning_rate=0.08, max_iter=400,
                                          l2_regularization=1.0)
    # out-of-fold predictions for a single partition (KFold), plus repeated-CV averages
    oof = cross_val_predict(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=SEED))
    rep = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    cv_r2 = cross_val_score(model, X, y, cv=rep, scoring="r2").mean()
    cv_mae = -cross_val_score(model, X, y, cv=rep, scoring="neg_mean_absolute_error").mean()
    metrics = dict(n=len(y), features=FEATURES,
                   cv_r2=round(float(cv_r2), 4),
                   cv_mae=round(float(cv_mae), 4),
                   oof_r2=round(float(r2_score(y, oof)), 4))

    # fit on full data for importance
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=50, random_state=SEED,
                                  scoring="r2")
    perm_rows = sorted(zip(FEATURES, perm.importances_mean, perm.importances_std),
                       key=lambda t: -t[1])
    with open(os.path.join(RES, "perm_importance.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "importance_mean", "importance_std"])
        for nm, m, s in perm_rows:
            w.writerow([nm, round(float(m), 5), round(float(s), 5)])
    metrics["permutation_importance"] = {nm: round(float(m), 5) for nm, m, _ in perm_rows}

    # SHAP
    try:
        import shap
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(X)
        shap_imp = np.abs(sv).mean(axis=0)
        order = sorted(zip(FEATURES, shap_imp), key=lambda t: -t[1])
        with open(os.path.join(RES, "shap_importance.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["feature", "mean_abs_shap"])
            for nm, v in order:
                w.writerow([nm, round(float(v), 5)])
        metrics["shap_importance"] = {nm: round(float(v), 5) for nm, v in order}
        metrics["top_feature"] = order[0][0]
    except Exception as e:
        print("SHAP skipped:", e)

    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("\nWrote results/metrics.json, perm_importance.csv, shap_importance.csv")


if __name__ == "__main__":
    main()
