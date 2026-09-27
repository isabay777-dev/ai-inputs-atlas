# Changelog

## v2.0.0 — 2026-09-27 (revision of the accompanying manuscript)

All analyses for the revised manuscript are in `revision/` and can be reproduced with the scripts listed below. The v1 files in the repository root are kept unchanged so that the first version remains reproducible (the v1 script `robustness.py` downloads current World Bank income and region classifications, so re-running it today can shift some v1 fixed-effects estimates in the third decimal).

### Corrections to the data
- **TOP500 country file.** The v1 file `data/top500_2025_11.csv` had been compiled from list summaries and omitted systems in economies with few installations. Complete lists (November 2023, June 2024, November 2025) are now collected system by system from top500.org (`revision/scrape_top500.py`, `revision/data/top500_*.csv`). Economies in the 136-economy sample with at least one listed system: November 2025, 24 in v1 → 41; June 2024 (hard-coded in v1 `robustness.py`), 16 → 37.
- **Researcher-data years.** v1 described researcher data as covering 2022–2024. The World Bank series gives the latest available year, which ranges from 1997 to 2024 (92 economies in 2022–2024, 44 earlier). The year for every economy is in `revision/data/researcher_years.csv`.

### Changes to the specification
- Main compute measure: log(1 + aggregate Rmax, petaflop/s), June 2024 list (v1: log(1 + number of systems), November 2025).
- All regression models: HC1 standard errors with t-based inference and 95% confidence intervals; the test of the difference in the compute coefficient between patents and publications uses a stacked regression with country-clustered standard errors.
- Hypotheses: H1 is now a proportional-scaling benchmark (research-workforce elasticity = 1) and H2 an increasing conditional association of compute with output (positive interaction); the former H3 is replaced by a descriptive comparison of Kazakhstan.
- New: test of proportional scaling (research-workforce coefficient = 1); centred research-workforce × compute interaction with marginal effects and quartile estimates; AI-patent comparison (CSET/ETO Country Activity Tracker v1.12.0, doi:10.5281/zenodo.22772306); robustness battery R1–R12 and supplementary interaction checks (R13, R14); like-for-like Kazakhstan comparison; nested cross-validation for the machine-learning model with importance computed on held-out folds.
- Amendments made after the written specification (all documented in the manuscript's response to reviewers): compute list used in the machine-learning model aligned with the main specification (June 2024); patent source switched from Our World in Data (series withdrawn from download) to the primary CSET/ETO archive with the same definition; Kazakhstan imbalance compared only among economies holding a listed system.

### Data availability within this repository
- TOP500: system-level lists are **not redistributed** (© TOP500, https://top500.org). `revision/scrape_top500.py` rebuilds them from the public list pages; `revision/data/top500_country_aggregates.csv` holds the derived country-level totals, from which all results reproduce exactly if the raw lists are absent.
- CSET/ETO AI patents: `revision/data/eto_cat/cat/patents_yearly_applications.csv` is an unmodified file from the Country AI Activity Metrics archive v1.12.0 (Melot, Arnold, Abdulla & Chalal, 2026, https://doi.org/10.5281/zenodo.22772306), redistributed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).

### Reproduction
```
cd revision
python3 scrape_top500.py          # optional: rebuild system-level TOP500 lists (a few minutes)
python3 revision_base.py          # Table 2 baseline models
python3 revision_robustness.py    # Table 3
python3 revision_interaction.py   # Table 4 (publications), Figure 7 inputs, Tables B1-B3
python3 revision_patents.py       # Table 4 (patents)
python3 revision_extra.py         # Table 2 extras, joint model on GDP sample, Table B4
python3 revision_ml.py            # Online Appendix C (about 7 minutes)
python3 revision_kazakhstan.py    # Section 4.4, Figure 5, Table D1
python3 revision_figures.py       # Figures 1-7
```

## v1.0.1 — 2026-06-16
First public release accompanying the submitted manuscript.
