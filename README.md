# TESS Exoplanet Discovery Pipeline 🪐

Autonomous exoplanet detection system for TESS light curves. Built for the ISRO Space Tech Hackathon.

## What this does

Processes TESS stellar light curves end-to-end to identify exoplanet transit signals:

1. **Downloads** FITS light curves from MAST (NASA archive)
2. **Preprocesses** — sigma-clips, detrends, normalises
3. **BLS Period Search** — Box Least Squares with TESS alias rejection (0.5, 1.0, 2.0, 13.5 day systematics)
4. **ML Classification** — Random Forest + 1D-CNN ensemble
5. **Batman Fitting** — Mandel & Agol physical transit model + radius plausibility check
6. **Reverse Pipeline** — independent trapezoidal fit + 6 physical consistency tests
7. **Three-Tier Decision** — KEEP / FLAG / DISCARD
8. **FLAG Deep-Analysis** — 8 deeper diagnostic tests to upgrade/downgrade ambiguous cases
9. **Auto-organised outputs** — `results/KEEP/`, `results/FLAG/`, `results/DISCARD/`

## Quick Start

```bash
# Single star
python pipeline.py --tic_id 261136679

# Run deep analysis on all flagged stars
python pipeline.py --analyze-flags

# Weekly Kaggle discovery run
# Open tess_weekly_runner.ipynb and click Run All Cells
```

## Batch Results (200-star cloud run)

| Decision | Count |
|---|---|
| DISCARD | 95 |
| FLAG → deep analysis | 16 |
| **KEEP** | **1** (TIC 188989177) |

After FLAG deep analysis:
- 🪐 KEEP: 1
- ⚠️ FLAG (human review): 9  
- 🗑️ DISCARD: 6

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Main CLI entry point |
| `flag_analyzer.py` | FLAG deep-analysis module (8 diagnostic tests) |
| `kaggle_discovery_runner.py` | Weekly batch runner (used by notebook) |
| `tess_weekly_runner.ipynb` | One-click Kaggle notebook — Run All Cells and walk away |
| `src/detect.py` | BLS period search + TESS alias rejection |
| `src/preprocess.py` | Light curve cleaning |
| `src/classify.py` | RF + CNN ensemble classifier |
| `src/fit_transit.py` | Batman transit model fitting |
| `src/reverse_pipeline.py` | Independent trapezoidal fit |
| `src/decision_engine.py` | Three-tier decision logic |
| `src/visualize.py` | Diagnostic plots |

## Kaggle Weekly Runner

`tess_weekly_runner.ipynb` is a 6-cell autonomous discovery notebook:

- **Cell 1** — Install libraries, clone this repo, create output folders
- **Cell 2** — Resume from last week's Kaggle dataset
- **Cell 3** — Build this week's target list from TIC catalog
- **Cell 4** — Run full pipeline on 800 stars (~8.5 hrs)
- **Cell 5** — Print summary table, flag new discoveries
- **Cell 6** — Push results back to Kaggle dataset for next week

## Physical Plausibility Gates

| Radius | Flag | Action |
|---|---|---|
| < 0.5 R⊕ | `too_small_flag` | Caution note, no forced discard |
| 0.5–25 R⊕ | — | Plausible planet range ✅ |
| 25–100 R⊕ | `giant_radius_eb_suspect` | Downgrade → Eclipsing Binary, DISCARD |
| > 100 R⊕ | `almost_certainly_eb` | Downgrade → Eclipsing Binary, DISCARD |

## Requirements

```
lightkurve wotan astropy astroquery batman-package
scipy scikit-learn imbalanced-learn tqdm joblib pandas numpy
```
