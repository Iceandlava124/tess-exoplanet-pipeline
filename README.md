# Autonomous TESS Exoplanet Discovery & Vetting Pipeline 🪐

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: CC0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Platform](https://img.shields.io/badge/platform-Kaggle%20%7C%20Local%20PC-orange)](https://www.kaggle.com/)
[![GitHub Repo](https://img.shields.io/badge/github-tess--exoplanet--pipeline-green)](https://github.com/Iceandlava124/tess-exoplanet-pipeline)

An end-to-end, machine-learning-accelerated exoplanet detection and physical vetting system. Designed to autonomously download, stitch, search, classify, and physically vet transiting exoplanet candidates from TESS Full Frame Images (FFIs).

---

## 🎯 The False Positive Challenge
In exoplanet transit surveys, **over 90% of periodic signals are false positives** (e.g., eclipsing binary stars, grazing binaries, starspots, or instrumental noise). 

This pipeline acts as a physical filter: it combines **1D Convolutional Neural Networks (CNNs)** and **Random Forests** for rapid screening, then applies a strict **multi-tiered physics-guided vetting battery** to eliminate false positives and validate genuine planet candidates.

---

## 🛠️ System Architecture

```text
                  +-----------------------------------+
                  |      NASA MAST Archive Queries     |
                  +-----------------+-----------------+
                                    |
                                    | [1] Multi-Source Stitching (SPOC, QLP, TGLC, Eleanor)
                                    v
                  +-----------------+-----------------+
                  |   Light Curve Preprocessing       |
                  |   (Sigma-clip, Adaptive SavGol)   |
                  +-----------------+-----------------+
                                    |
                                    | [2] Period Search (Box Least Squares + Systemic Rejections)
                                    v
                  +-----------------+-----------------+
                  |    ML Screening Classifiers       |
                  |    (Ensemble: 1D-CNN + RF)        |
                  +-----------------+------------+----+
                                    |            |
                         Candidate  |            | Noise / Variable
                                    v            v
  +---------------------------------+---+    +---+-----+
  | [3] Physics-Guided Vetting Suite    |    | Discard |
  | (TTVs, SWEET, Centroids, Odd-Even)  |    +---------+
  +-----------------+-------------------+
                    |
                    +-------------------+
                    |                   |
                    | Fail V-Shape /    | Pass & Borderline
                    | Odd-Even Check    |
                    v                   v
              +-----+------+      +-----+------+
              | Auto-Discard|      | KEEP / FLAG|
              |  Binary (EB)|      | Human Queue|
              +-------------+      +------------+
```

---

## 🔬 Core Vetting Capabilities

Our vetting suite implements advanced astrophysical diagnostics to classify candidates:

1. **U/V Shape Profiling**: Performs a dual-model optimization fit using a flat-bottomed **Trapezoid (planetary U-shape)** and a pointed **Triangle (binary V-shape)** to compare their Sum of Squared Residuals ($SSR_{\text{trapezoid}} / SSR_{\text{triangle}}$). A ratio $\ge 0.85$ paired with low flatness indicates a V-shaped grazing binary geometry.
2. **Odd-Even Transit Depth Check**: Compares alternating transits to catch eclipsing binaries with alternating primary and secondary eclipses.
3. **Stellar Variability (SWEET Test)**: Uses a Lomb-Scargle periodogram on out-of-transit data to ensure the detected period is not a phase-locked stellar rotation or starspot harmonic.
4. **Kinematic Duration Check**: Verifies that the observed transit duration ($T_{\text{obs}}$) matches physical limits set by Kepler's Third Law ($T_{\text{max}}$).
5. **False-Negative Shield**: Guards shallow transits (depth $< 2000\text{ ppm}$) from being auto-discarded if noise distorts the profile shape, routing them safely to human review.

---

## 🎮 Interactive Simulation Tool
We have built a gorgeous, interactive web-based **Exoplanet Injection & Recovery Simulator** to demonstrate the physical constraints of exoplanet pipelines.

You can simulate:
* **The Detrending Trap**: See how short filter windows erase planetary transits.
* **TTV Smearing**: See how gravitational interactions between planets disrupt phase-folding.
* **Signal-to-Noise Floor**: Interactively trace the detection boundaries of Earth-sized planets.

📂 **Run Locally**: Open [injection_recovery_simulator.html](file:///c:/Users/gudae/Desktop/Learn_ml/injection_recovery_simulator.html) directly in any web browser to play with it!

---

## 📂 Project Structure & Inventory

| File/Folder | Purpose |
| :--- | :--- |
| 📁 [src/](file:///c:/Users/gudae/Desktop/Learn_ml/src) | Main pipeline modules (detect, preprocess, classify, fit) |
| 📄 [pipeline.py](file:///c:/Users/gudae/Desktop/Learn_ml/pipeline.py) | Main CLI entry point for processing individual TIC targets |
| 📄 [flag_analyzer.py](file:///c:/Users/gudae/Desktop/Learn_ml/flag_analyzer.py) | Main automated diagnostic diagnostic suite |
| 📄 [cross_verify_sources.py](file:///c:/Users/gudae/Desktop/Learn_ml/cross_verify_sources.py) | Utility script to cross-compare a target across multiple data pipelines |
| 📄 [stitch_and_vet_local.py](file:///c:/Users/gudae/Desktop/Learn_ml/stitch_and_vet_local.py) | Local vetting loop for target batches |
| 📄 [injection_recovery_simulator.html](file:///c:/Users/gudae/Desktop/Learn_ml/injection_recovery_simulator.html) | Standalone interactive pipeline simulator page |
| 📁 [latex/](file:///c:/Users/gudae/Desktop/Learn_ml/latex) | LaTeX source files for the research publication manuscript |
| 📁 [notebooks/](file:///c:/Users/gudae/Desktop/Learn_ml/notebooks) | Jupyter documentation notebooks for each pipeline stage |

---

## ⚡ Quick Start

### 1. Installation
Install the necessary requirements (astronomy and ML libraries):
```bash
pip install -r requirements.txt
```

### 2. Process a Single Star (CLI)
Download, preprocess, search, and vet a target star:
```bash
python pipeline.py --tic_id 22529346
```

### 3. Run Vetting Batch
Process the custom batch list:
```bash
python stitch_and_vet_local.py
```

### 4. Cross-Pipeline Vetting
Compare a target across SPOC, QLP, TGLC, and Eleanor:
```bash
python cross_verify_sources.py
```

---

## 📜 Scientific Publication Manuscript
Our latest vetting findings have been compiled into a professional LaTeX document ready for AAS research notes:
* 📄 **LaTeX Document**: [latex/manuscript.tex](file:///c:/Users/gudae/Desktop/Learn_ml/latex/manuscript.tex)

This paper details why the pipeline successfully retired our short-period candidates as astrophysical eclipsing binaries using mathematical modeling.
