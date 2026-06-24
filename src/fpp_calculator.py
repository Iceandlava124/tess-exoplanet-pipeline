"""
src/fpp_calculator.py
=====================
TRICERATOPS calculates the statistical probability that a transit signal
is NOT a planet — called the False Positive Probability (FPP).
This is required for ExoFOP submission and taken seriously by astronomers.
FPP < 0.1 (10%) is generally considered a viable planet candidate.
FPP < 0.01 (1%) is considered statistically validated without spectroscopy.
"""

import logging
import numpy as np

# Monkeypatch numpy.int to avoid deprecation errors in pytransit / triceratops
np.int = int

# Disable global SSL verification to prevent stev.oapd.inaf.it connection errors
import ssl
import requests
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()

# Monkeypatch mechanicalsoup and requests to bypass SSL verification completely
from mechanicalsoup import StatefulBrowser
original_open = StatefulBrowser.open
def patched_open(self, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_open(self, url, *args, **kwargs)
StatefulBrowser.open = patched_open

original_submit = StatefulBrowser.submit
def patched_submit(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_submit(self, *args, **kwargs)
StatefulBrowser.open = patched_open
StatefulBrowser.submit = patched_submit

original_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request

logger = logging.getLogger(__name__)

def calculate_fpp(tic_id, period, epoch, depth, duration, 
                  sector, time, flux):
    """
    Runs TRICERATOPS false positive probability calculation.
    
    Returns:
        fpp: float 0.0-1.0 (probability signal is NOT a planet)
        nfp: float (probability it is a nearby false positive)
        fpp_components: dict of individual scenario probabilities
        is_viable_candidate: True if FPP < 0.1
        is_statistically_validated: True if FPP < 0.01
    """
    try:
        import triceratops.triceratops as tr
        
        # Initialise TRICERATOPS with the TIC ID and sector
        sec_list = [int(sector)] if sector is not None else [1]
        target = tr.target(ID=tic_id, sectors=sec_list)
        
        # Calculate depth in parts per thousand for TRICERATOPS
        # If depth is fractional (e.g. 0.01), convert to parts per thousand by * 1000
        # If depth is in ppm (e.g. 10000), convert to parts per thousand by / 1000
        if depth < 1.0:
            depth_ppt = depth * 1000.0
        else:
            depth_ppt = depth / 1000.0
        
        target.calc_depths(tdepth=depth_ppt)
        
        # Run the FPP calculation
        # This queries Gaia for nearby stars and calculates
        # the probability of each false positive scenario
        target.calc_probs(
            time=time,
            flux_0=flux,
            flux_err_0=np.ones_like(flux) * np.std(flux),
            P_orb=period
        )
        
        fpp = target.FPP
        nfp = target.NFPP
        
        return {
            "fpp": float(fpp),
            "nfp": float(nfp),
            "combined_fpp": float(fpp + nfp),
            "is_viable_candidate": (fpp + nfp) < 0.1,
            "is_statistically_validated": (fpp + nfp) < 0.01,
            "fpp_status": (
                "VALIDATED" if (fpp + nfp) < 0.01 else
                "VIABLE" if (fpp + nfp) < 0.1 else
                "LIKELY_FP"
            )
        }
        
    except Exception as e:
        # TRICERATOPS can fail if Gaia query times out
        # Return None values but don't crash the pipeline
        logger.warning(f"TRICERATOPS FPP calculation failed for TIC {tic_id}: {e}")
        return {
            "fpp": None,
            "nfp": None,
            "combined_fpp": None,
            "is_viable_candidate": None,
            "is_statistically_validated": None,
            "fpp_status": f"calculation_failed: {str(e)}"
        }
