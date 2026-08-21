"""Shared pricing math -- the revenue-maximising surge model from
notebooks/taxi_surge_pricing.py, Sec 7.2/7.5. Imported by both the case-study
notebook (which fits it) and streamlit_app.py (which lets executives probe it),
so the two never drift apart.
"""
import numpy as np

MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0]


def calibrate(eps_base, p1):
    """betaF, alpha from two interpretable beliefs: elasticity at 1.0x, and share won at base price."""
    betaF = eps_base / (1 - p1)
    alpha = np.log(p1 / (1 - p1)) + betaF
    return alpha, betaF


def P_of_m(m, alpha, betaF):
    return 1 / (1 + np.exp(-(alpha - betaF * m)))


def elasticity_of_m(m, alpha, betaF):
    Pm = P_of_m(m, alpha, betaF)
    return -betaF * m * (1 - Pm)


def expected_revenue(Q0, F, m, alpha, betaF):
    demand_reduction_pct = 1 - P_of_m(m, alpha, betaF) / P_of_m(1.0, alpha, betaF)
    return Q0 * (1 - demand_reduction_pct) * F * m, demand_reduction_pct


def optimise_row(row, eps_col, eps_base_used=None, p1_used=None, p1_default=0.5, multipliers=MULTIPLIERS):
    """Sweep the multiplier grid for one zone-hour and return the revenue-maximising choice."""
    eps = eps_base_used if eps_base_used is not None else row[eps_col]
    p1 = p1_used if p1_used is not None else p1_default
    a, b = calibrate(eps, p1)
    revs = {m: expected_revenue(row["forecast"], row["F"], m, a, b)[0] for m in multipliers}
    best_m = max(revs, key=revs.get)
    return revs[1.0], revs[best_m], best_m
