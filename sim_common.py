"""Shared helpers for the theorem-driven simulation study.

All simulations are vectorised over replicates: arrays have shape (reps, ...).
Renewal convention: Lambda_t(w) = sum_{s=1}^S w_s Y_{t-s}.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (journal requirement: no Type 3)
matplotlib.rcParams["ps.fonttype"] = 42
from scipy.stats import gamma as gamma_dist
from scipy.optimize import brentq
from scipy.special import logsumexp

S_DEFAULT = 15
CV_DEFAULT = 0.4  # fixed coefficient of variation for the Gamma GI family


def scenario_rng(seed: int, *keys: int) -> np.random.Generator:
    """Independent random stream for one scenario.

    Each scenario gets its own stream from (seed, keys), so results for one scenario do not
    depend on how many random numbers earlier scenarios consumed. This keeps Poisson
    scenarios bit-identical across NumPy versions and stops version differences in the
    negative-binomial (gamma) sampler from spreading to other scenarios.
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed)] + [int(k) for k in keys]))


def disc_gamma_weights(mean: float, sd: float | None = None, S: int = S_DEFAULT,
                       cv: float = CV_DEFAULT) -> np.ndarray:
    """Discretised Gamma GI: w_s proportional to P(s-1/2 < G <= s+1/2), s=1..S."""
    if sd is None:
        sd = cv * mean
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean
    lo = np.arange(0.5, S + 0.5)
    up = np.arange(1.5, S + 1.5)
    w = gamma_dist.cdf(up, a=shape, scale=scale) - gamma_dist.cdf(lo, a=shape, scale=scale)
    w = np.maximum(w, 0.0)
    return w / w.sum()


def renewal_R(w: np.ndarray, g: float) -> float:
    """Euler-Lotka reproduction number for growth rate g: R_w(g) = 1/sum_s w_s e^{-gs}."""
    s = np.arange(1, len(w) + 1)
    return 1.0 / float(np.sum(w * np.exp(-g * s)))


def mean_for_R(g: float, target_R: float, lo: float = 0.6, hi: float = 15.0,
               S: int = S_DEFAULT, cv: float = CV_DEFAULT) -> float:
    """GI mean m with R_m^disc(g) = target_R (R_m is increasing in m on the audited range)."""
    f = lambda m: renewal_R(disc_gamma_weights(m, S=S, cv=cv), g) - target_R
    return brentq(f, lo, hi, xtol=1e-12)


def poisson_kl(mu_p: np.ndarray, mu_q: np.ndarray) -> np.ndarray:
    """KL(Pois(mu_p) || Pois(mu_q)), elementwise, with 0 log 0 = 0."""
    mu_p = np.asarray(mu_p, float)
    mu_q = np.asarray(mu_q, float)
    out = mu_q - mu_p
    pos = mu_p > 0
    out[pos] += mu_p[pos] * np.log(mu_p[pos] / mu_q[pos])
    return out


def simulate_renewal(rng: np.random.Generator, w0: np.ndarray, R: float, T: int,
                     reps: int, init: float = 20.0, kappa: float | None = None,
                     W_extra: np.ndarray | None = None):
    """Simulate a renewal epidemic Y_t | F ~ Pois(R Lambda_t(w0)) or branching NB.

    Returns dict with
      Y      : (reps, T) counts for t = 1..T
      lam0   : (reps, T) Lambda_t(w0)
      lamW   : (reps, T, M) Lambda_t(w) for each column w of W_extra (S x M), if given
    The initial history Y_{1-S..0} is constant `init`.
    kappa=None gives Poisson; otherwise Y_t ~ NB(mean R*lam0, size kappa*lam0).
    """
    S = len(w0)
    hist = np.full((reps, S), float(init))  # hist[:, 0] = Y_{t-1}, hist[:, s-1] = Y_{t-s}
    Y = np.zeros((reps, T))
    lam0 = np.zeros((reps, T))
    lamW = None if W_extra is None else np.zeros((reps, T, W_extra.shape[1]))
    for t in range(T):
        l0 = hist @ w0
        lam0[:, t] = l0
        if W_extra is not None:
            lamW[:, t, :] = hist @ W_extra
        mu = R * l0
        if kappa is None:
            y = rng.poisson(mu)
        else:
            k = kappa * l0
            lam = rng.gamma(shape=np.maximum(k, 1e-300), scale=np.where(k > 0, mu / np.maximum(k, 1e-300), 0.0))
            y = rng.poisson(lam)
        Y[:, t] = y
        hist = np.concatenate([y[:, None].astype(float), hist[:, :-1]], axis=1)
    return {"Y": Y, "lam0": lam0, "lamW": lamW}


def safe_poisson_log_mixture(N: np.ndarray, B: np.ndarray, r0: float,
                             rho_grid: np.ndarray) -> np.ndarray:
    """log of a uniform mixture over rho of the robust Poisson e-process
    exp{log(rho/r0) N_t - (rho - r0) B_t}. N, B have shape (reps, T)."""
    a = np.log(rho_grid / r0)[None, None, :]
    b = (rho_grid - r0)[None, None, :]
    logE = a * N[..., None] - b * B[..., None]
    return logsumexp(logE, axis=-1) - np.log(len(rho_grid))


def first_crossing(logE: np.ndarray, level: float) -> np.ndarray:
    """First index (1-based day) with logE >= level; np.inf if never."""
    hit = logE >= level
    any_hit = hit.any(axis=1)
    idx = np.argmax(hit, axis=1).astype(float) + 1.0
    idx[~any_hit] = np.inf
    return idx
