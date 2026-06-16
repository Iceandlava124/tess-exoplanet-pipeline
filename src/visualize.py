"""
src/visualize.py
================
Publication-quality visualisation of light curves and detection results.

Creates the 4-panel diagnostic figure used for each target:
  Panel 1: Raw + detrended light curve with transit markers
  Panel 2: BLS periodogram with best peak highlighted
  Panel 3: Phase-folded light curve with batman model
  Panel 4: Residuals (data - model) with confidence score
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette
PLANET_COLOR   = "#4FC3F7"   # sky blue
EB_COLOR       = "#FF7043"   # deep orange
BLEND_COLOR    = "#AB47BC"   # purple
NOSIG_COLOR    = "#78909C"   # blue grey
MODEL_COLOR    = "#FFD54F"   # amber
RESIDUAL_COLOR = "#80CBC4"   # teal

LABEL_COLORS = {0: NOSIG_COLOR, 1: PLANET_COLOR, 2: EB_COLOR, 3: BLEND_COLOR}
LABEL_NAMES  = {0: "No Signal", 1: "Planet Transit", 2: "Eclipsing Binary", 3: "False Positive"}


def plot_diagnostic(
    tic_id: int,
    time_raw: np.ndarray,
    flux_raw: np.ndarray,
    time_clean: np.ndarray,
    flux_clean: np.ndarray,
    bls_periods: np.ndarray,
    bls_power: np.ndarray,
    bls_params: Dict,
    transit_params: Dict,
    classification: Dict,
    phase_folded_flux: Optional[np.ndarray] = None,
    model_flux: Optional[np.ndarray] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Create a 4-panel diagnostic figure for one target.

    Args:
        tic_id:            TIC ID of the target star
        time_raw/flux_raw: Raw (pre-preprocessing) light curve
        time_clean/flux_clean: Detrended, normalised light curve
        bls_periods:       Period grid tested by BLS
        bls_power:         BLS power at each period
        bls_params:        Best BLS parameters dict
        transit_params:    Batman-fitted parameters dict
        classification:    Classification result dict from classify.py
        phase_folded_flux: Optional phase-folded flux array (sorted by phase)
        model_flux:        Optional batman model flux at same phases
        save:              If True, save PNG to results/figures/

    Returns:
        matplotlib Figure object
    """
    plt.style.use("dark_background")

    label = classification.get("label", 0)
    conf  = classification.get("confidence", 0.0)
    color = LABEL_COLORS[label]

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── Title with classification result ──────────────────────────────────────
    fig.suptitle(
        f"TIC {tic_id}  |  Classification: {LABEL_NAMES[label]}  "
        f"(Confidence: {100*conf:.1f}%)",
        fontsize=15, fontweight="bold", color=color, y=0.98
    )

    # ── Panel 1: Detrended light curve ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_clean, flux_clean, ".", ms=1.0, alpha=0.5, color="#90A4AE",
             label="Detrended flux")

    # Mark transit windows
    period = bls_params.get("period", 1.0)
    t0     = bls_params.get("t0", time_clean[0])
    dur    = bls_params.get("duration", 0.1)

    if period > 0 and label in [1, 2]:
        t_transit = t0
        while t_transit < time_clean[-1]:
            ax1.axvspan(t_transit - dur/2, t_transit + dur/2,
                        alpha=0.15, color=color, zorder=0)
            t_transit += period

    ax1.set_xlabel("Time (BTJD days)", fontsize=11)
    ax1.set_ylabel("Normalised Flux", fontsize=11)
    ax1.set_title(
        f"Detrended Light Curve  |  Best period: {period:.4f} d  "
        f"|  SNR: {bls_params.get('snr', 0):.1f}",
        fontsize=11, loc="left"
    )
    ax1.legend(loc="upper right", fontsize=9)

    # ── Panel 2: BLS Periodogram ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(bls_periods, bls_power, color="#B0BEC5", lw=0.7, alpha=0.8)
    ax2.axvline(period, color=color, lw=2, ls="--", label=f"Best period = {period:.4f} d")

    # Mark harmonics
    for h in [2, 3]:
        if period * h < bls_periods[-1]:
            ax2.axvline(period * h, color=color, lw=0.8, ls=":", alpha=0.5)

    ax2.set_xscale("log")
    ax2.set_xlabel("Period (days)", fontsize=11)
    ax2.set_ylabel("BLS Power (SNR²)", fontsize=11)
    ax2.set_title("BLS Periodogram", fontsize=11, loc="left")
    ax2.legend(fontsize=9)

    # ── Panel 3: Phase-folded light curve ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])

    if phase_folded_flux is not None and len(phase_folded_flux) > 0:
        n = len(phase_folded_flux)
        phases = np.linspace(-0.5, 0.5, n)

        ax3.plot(phases, phase_folded_flux, ".", ms=1.5, alpha=0.5,
                 color="#90A4AE", label="Data")

        # Bin the phase-folded data for clarity
        n_bins = 50
        bin_edges = np.linspace(-0.5, 0.5, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_flux = np.array([
            np.nanmedian(phase_folded_flux[
                (phases >= bin_edges[i]) & (phases < bin_edges[i+1])
            ]) if np.any((phases >= bin_edges[i]) & (phases < bin_edges[i+1]))
            else np.nan
            for i in range(n_bins)
        ])
        ax3.plot(bin_centres, bin_flux, "o", ms=3, color=color,
                 label="Binned (50 bins)", zorder=5)

        if model_flux is not None:
            ax3.plot(phases, model_flux, color=MODEL_COLOR, lw=2,
                     label="Batman model", zorder=6)

    depth = transit_params.get("transit_depth_pct", 0.0)
    ax3.set_xlabel("Phase", fontsize=11)
    ax3.set_ylabel("Normalised Flux", fontsize=11)
    ax3.set_title(
        f"Phase-folded  |  Depth: {depth:.3f}%  "
        f"|  Duration: {transit_params.get('transit_duration_hr', 0):.2f} hr",
        fontsize=11, loc="left"
    )
    ax3.legend(fontsize=9, loc="lower center")

    # ── Panel 4: Classification confidence bar + parameter summary ────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    combined_proba = classification.get("combined_proba", [0.25]*4)
    rf_proba       = classification.get("rf_proba", [0.25]*4)
    cnn_proba      = classification.get("cnn_proba", [0.25]*4)

    # Probability bars
    labels    = [LABEL_NAMES[i] for i in range(4)]
    bar_colors = [LABEL_COLORS[i] for i in range(4)]
    x = np.arange(4)
    width = 0.25

    ax4_inner = ax4.inset_axes([0.0, 0.4, 0.5, 0.55])
    ax4_inner.bar(x - width,   rf_proba,       width, color=bar_colors, alpha=0.6, label="Random Forest")
    ax4_inner.bar(x,           cnn_proba,      width, color=bar_colors, alpha=0.8, label="CNN")
    ax4_inner.bar(x + width,   combined_proba, width, color=bar_colors, alpha=1.0, label="Ensemble")
    ax4_inner.set_xticks(x)
    ax4_inner.set_xticklabels(labels, fontsize=9)
    ax4_inner.set_ylabel("Probability", fontsize=10)
    ax4_inner.set_title("Classification Probabilities", fontsize=11, loc="left")
    ax4_inner.legend(fontsize=8, loc="upper right")
    ax4_inner.set_ylim(0, 1.1)

    # Parameter table & Vetting Status
    vetting_status = classification.get("vetting_status", "Not Vetted")
    vetting_reason = classification.get("vetting_reason", "No vetting performed.")
    
    param_text = (
        f"{'FITTED PARAMETERS':^50}\n"
        f"{'─'*50}\n"
        f"  Orbital Period    : {transit_params.get('period', 0):.6f} ± "
        f"{transit_params.get('period_err', 0):.6f} days\n"
        f"  Transit Depth     : {transit_params.get('transit_depth_pct', 0):.4f} ± "
        f"{transit_params.get('transit_depth_err', 0)*100:.4f} %\n"
        f"  Transit Duration  : {transit_params.get('transit_duration_hr', 0):.3f} ± "
        f"{transit_params.get('transit_duration_hr_err', 0):.3f} hours\n"
        f"  Planet Radius     : {transit_params.get('rp_earth', 0):.2f} R⊕ (if Rstar=1 R☉)\n"
        f"  a/Rs              : {transit_params.get('a_over_rs', 0):.2f}\n"
        f"  Reduced χ²        : {transit_params.get('chi2_reduced', 0):.3f}\n"
        f"  BLS SNR           : {bls_params.get('snr', 0):.2f}\n"
        f"{'─'*50}\n"
        f"  Final ML Label    : {LABEL_NAMES[label]} ({100*conf:.1f}%)\n"
        f"  Vetting Status    : {vetting_status.upper()}\n"
        f"  Vetting Reason    : {vetting_reason}"
    )
    
    # Border color matching vetting status
    status_lower = vetting_status.lower()
    if "approved" in status_lower:
        border_color = "#4CAF50"  # Green
    elif "manual" in status_lower:
        border_color = "#FFC107"  # Yellow
    elif "rejected" in status_lower:
        border_color = "#F44336"  # Red
    else:
        border_color = color
        
    ax4.text(0.52, 0.95, param_text, transform=ax4.transAxes,
             fontsize=9, va="top", ha="left", fontfamily="monospace",
             color="white",
             bbox=dict(facecolor="#1a1a2e", edgecolor=border_color, boxstyle="round,pad=0.5"))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        out_path = FIGURES_DIR / f"TIC_{tic_id}_diagnostic.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
        logger.info(f"Saved diagnostic plot to {out_path}")

    return fig
