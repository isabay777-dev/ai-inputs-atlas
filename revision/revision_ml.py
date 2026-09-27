"""Step 8c (CEEJ-2026-0071.R1): ML appendix per specification_frozen.md sec. 7.
HistGradientBoostingRegressor; features T = ln(total researchers), C = ln(1+Rmax, TOP500 June 2024), ln GDP pc (n = 131).
Nested CV: outer 5-fold x 20 repeats (seeds 0-19), inner 3-fold grid search (max_depth {2,3}, learning_rate {0.05,0.1},
max_iter {100,200}), scoring R2. Permutation importance (10 permutations, R2 drop) and mean |SHAP| computed only on
held-out outer folds. Stability = share of the 100 outer folds / 20 repeats in which each feature ranks first."""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
import shap
from revision_base import build
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ["T", "C", "lgdp"]
d = build("2024_06").dropna(subset=FEATS + ["y"]).reset_index(drop=True)
X, y = d[FEATS].values, d.y.values
GRID = {"max_depth": [2, 3], "learning_rate": [0.05, 0.1], "max_iter": [100, 200]}
r2s, maes, perm, shp, first_perm, params = [], [], [], [], [], []
rep_first = []
for seed in range(20):
    oof = np.zeros(len(y)); rep_perm = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        gs = GridSearchCV(HistGradientBoostingRegressor(random_state=seed), GRID, cv=KFold(3, shuffle=True, random_state=seed),
                          scoring="r2").fit(X[tr], y[tr])
        m = gs.best_estimator_; params.append(str(gs.best_params_))
        oof[te] = m.predict(X[te])
        pi = permutation_importance(m, X[te], y[te], n_repeats=10, random_state=seed, scoring="r2").importances_mean
        perm.append(pi); rep_perm.append(pi); first_perm.append(FEATS[int(np.argmax(pi))])
        shp.append(np.abs(shap.TreeExplainer(m).shap_values(X[te])).mean(0))
    r2s.append(r2_score(y, oof)); maes.append(mean_absolute_error(y, oof))
    rep_first.append(FEATS[int(np.argmax(np.mean(rep_perm, 0)))])
perm, shp = np.array(perm), np.array(shp)
names = {"T": "research capacity (ln total researchers)", "C": "compute (ln 1+Rmax)", "lgdp": "ln GDP per capita"}
res = {"n": len(y), "features": FEATS,
       "oof_r2_mean_sd": [round(np.mean(r2s), 3), round(np.std(r2s), 3)],
       "oof_mae_mean_sd": [round(np.mean(maes), 3), round(np.std(maes), 3)],
       "perm_importance_mean_sd": {f: [round(perm[:, i].mean(), 3), round(perm[:, i].std(), 3)] for i, f in enumerate(FEATS)},
       "shap_mean_abs_mean_sd": {f: [round(shp[:, i].mean(), 3), round(shp[:, i].std(), 3)] for i, f in enumerate(FEATS)},
       "share_rank1_by_fold": {f: round(first_perm.count(f) / len(first_perm), 3) for f in FEATS},
       "share_rank1_by_repeat": {f: round(rep_first.count(f) / len(rep_first), 3) for f in FEATS},
       "selected_params_top": pd.Series(params).value_counts().head(3).to_dict()}
json.dump(res, open(os.path.join(HERE, "results", "results_ml.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
