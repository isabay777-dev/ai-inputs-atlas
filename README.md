# ai-inputs-atlas

A reproducible toolkit and machine-learning analysis of national **AI inputs** — compute and talent — and their link to AI knowledge output across countries, with a focus on a catch-up economy (Kazakhstan) against advanced benchmarks.

This repository accompanies the economics manuscript of the same study and contains all code, the assembled country panel, the econometric core, an independent machine-learning triangulation, and the analysis outputs needed to reproduce every figure and number in the paper.

## What this is

Two of the three legs of the **AI Triad** (compute, data, talent) are observable at country level. This project measures them and asks which one actually binds national AI knowledge output.

- **A panel.** One row per country (`data/panel.csv`) combining AI knowledge output (AI scholarly publications), a talent input (researchers in R&D per million; total researchers), a compute input (TOP500 supercomputers), and GDP per capita (PPP) as a wealth control. All values trace to public primary sources.
- **An econometric core** (`econometrics.py`). Knowledge-production-function OLS: `log(AI output)` on `log(talent)` and `log(compute)`, with a combined input index, standardized-beta decomposition, a GDP-controlled robustness model, and Kazakhstan's input-imbalance diagnostics.
- **A machine-learning triangulation** (`ml_model.py`). A gradient-boosting model predicts AI output from the inputs under repeated cross-validation; permutation importance and SHAP identify the dominant input without assuming linearity.

## Key results (reproducible from this repository)

- **Talent scales with output (H1).** `log(AI publications) ~ log(total researchers)`: elasticity ≈ **0.92**, R² ≈ **0.84** (n ≈ 135 countries).
- **Talent is the binding input (H2).** In the joint model, standardized β for talent ≈ **0.88** (p ≪ 0.001) versus compute ≈ **0.06** (n.s.): once talent is controlled, compute adds essentially nothing. The machine-learning model agrees — talent is the top feature by both permutation importance and SHAP, and cross-validated R² > 0.7 with no linearity assumption.
- **Kazakhstan's compute–talent imbalance (H3).** On TOP500 system count Kazakhstan ranks ≈ **22nd of 135** countries (it entered the TOP500 for the first time in November 2025), yet on research-personnel intensity it ranks only ≈ **66th**. Among the focal set, Kazakhstan fields **2** TOP500 systems (comparable to Switzerland's 3) but has **758** researchers per million versus Switzerland's 6,108 — roughly an eight-fold talent gap. The country has bought the machines faster than it has built the minds.

## Installation

```bash
python -m venv venv && source venv/bin/activate    # Python 3.9+
pip install -r requirements.txt
```

No GPU required; the full pipeline runs on a standard CPU in minutes (network needed for the World Bank API pull).

## Reproducing the analysis

```bash
python fetch_data.py     # pull World Bank indicators, assemble data/panel.csv
python econometrics.py   # OLS core (H1/H2/H3) -> results/ols_results.json
python ml_model.py       # gradient boosting + permutation + SHAP -> results/
python make_figures.py   # figures/ (F1-F6)
python test_smoke.py     # quick end-to-end checks of the headline findings
```

## Repository layout

- `fetch_data.py`, `econometrics.py`, `ml_model.py`, `make_figures.py` — pipeline.
- `data/ai_publications_cset_owid.csv` — AI output (CSET/ETO via Our World in Data).
- `data/top500_2025_11.csv` — compute input (TOP500 Nov-2025 per-country counts, verified).
- `data/panel.csv` — assembled country panel (built by `fetch_data.py`).
- `results/` — `ols_results.json`, `metrics.json`, `perm_importance.csv`, `shap_importance.csv`.
- `figures/` — F1–F6 as in the manuscript.

## Data sources and their terms

Input datasets are governed by their own licences and terms, **not** the MIT licence of this repository. Cite the originals:

- **TOP500** (compute) — TOP500.org, November 2025 list. https://top500.org
- **Center for Security and Emerging Technology (CSET/ETO)** Country Activity Tracker (AI scholarly publications), distributed via **Our World in Data** (CC BY).
- **World Bank** World Development Indicators — `SP.POP.SCIE.RD.P6` (researchers per million, sourced from UNESCO Institute for Statistics), `SP.POP.TOTL`, `NY.GDP.PCAP.PP.KD`. https://data.worldbank.org
- **Stanford HAI AI Index** (per-capita AI talent benchmarks, cited in the paper). https://aiindex.stanford.edu

## Licence

Code is released under the **MIT Licence** (see `LICENSE`). Third-party data redistributed for convenience remain under their source terms above.

## Citation

If you use this code or panel, please cite the accompanying paper (citation to be added on publication) and the source datasets above. See `CITATION.cff`.

## AI-use note

AI assistants were used under the author's direction for language editing, code drafting and review, and a pre-submission self-review. All claims, numerical results, and interpretations were verified by the author against the source data and code.
