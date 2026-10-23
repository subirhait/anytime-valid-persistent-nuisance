# anytime-valid-persistent-nuisance

Code for **Hait, S. (2026). Anytime-valid sequential detection under persistent nuisance: predictable nulls and dispersion rescue.** This repository reproduces every simulation, figure and table in the paper and its supplement.

## Requirements
- Python 3.10 or later, with numpy, scipy, pandas and matplotlib (`pip install -r requirements.txt`)
- R with the EpiEstim package, used only to export the SARS data

## Reproduce
1. Export the SARS 2003 data once:

       Rscript export_sars_data.R

   This writes `epi_data/SARS2003_incidence.csv` and `epi_data/SARS2003_si_distr.csv`.

2. Run everything:

       python run_all.py --epi-data epi_data

   This takes about 4–6 minutes on a laptop. Use `python run_all.py --quick` for a fast check.

Outputs go to `results/`, and `results/versions.txt` records the software versions. Each scenario has its own random stream. A run on Windows (Python 3.13.9, NumPy 2.3.5) and a run on Linux (Python 3.13, NumPy 2.5.3) agree to within 2×10⁻⁷ in every reported quantity. The `results/` folder here contains the reference output.

## Files

| Script | Paper |
|---|---|
| `exp1_predictable_null.py` | Experiment 1, Figure 1, supplement Table 1 |
| `exp2_dispersion_rescue.py` | Experiment 2, Figure 2 (and supplement Figure 1) |
| `exp3_far_growth.py` | Experiment 3, supplement Figure 2 |
| `exp4_gi_robustness.py` | Experiment 4, supplement Table 2 |
| `exp5_competitors.py` | Experiment 5, supplement Table 3 |
| `exp6_false_alarms.py` | Experiment 6, supplement Figure 3 |
| `sars_application.py` | SARS application, Figure 3, supplement Table 4 |

`sim_common.py` holds the shared functions. `run_all.py` runs the whole study.

## Data
The SARS 2003 Hong Kong data are the `SARS2003` dataset distributed with the R package EpiEstim (Cori et al., 2013). They are not redistributed here; `export_sars_data.R` extracts them from an installed copy of EpiEstim.

## License
MIT (see `LICENSE`).
