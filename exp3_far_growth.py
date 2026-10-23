"""Experiment 3: far-growth calendar-time law (Theorems 8.1 and 8.2).

Theorem 8.1: in the positive first-order region R_min(g) > r0, the safe mixture
e-process crosses 1/alpha at
    tau_alpha = (1/g) [ log b_alpha - log{ I K(W_inf) } ] + O_P(1),   b_alpha = log(1/alpha),
where A_t(w0) = sum_{j<=t} Lambda_j(w0) = K(W_inf) e^{g t}(1 + o(1)) and
I = I_{-,epi} = (R/R_min) d_{kappa_-}(R_min || r0).
Theorem 8.2: no globally anytime-valid detector can improve the leading constant 1/g.

Because log log(1/alpha) moves slowly, we use alpha = 10^-k for k up to 100 (thresholds
are deterministic, so tiny alpha costs nothing) and check
  (a) the slope of median tau_alpha against log b_alpha equals 1/g;
  (b) the path-wise residual tau_alpha - tau_hat_alpha, with tau_hat computed from the
      path's own K_hat = A_T(w0) e^{-gT}, stays O(1) and stabilises as alpha -> 0.
An oracle benchmark (log-likelihood ratio of the true model against the least-favourable
null r0 with w = w(m_lo), not a valid test for the composite null) shows the same 1/g slope,
illustrating that robustness changes only the intercept.

Design: realistic Gamma GI (mean 5 d, CV 0.4, 15 lags); nuisance class means [3.5, 8];
r0 = R_min(g)/ratio with ratio in {1.05, 1.20}; g in {0.05, 0.10, 0.20};
Poisson offspring and branching NB with kappa0 = kappa_- = 0.5.

Usage: python exp3_far_growth.py [--reps 1000] [--quick]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammaln
from sim_common import scenario_rng, disc_gamma_weights, renewal_R, simulate_renewal

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261023
M_LO, M_HI, N_GRID = 3.5, 8.0, 41
SETTINGS = [(0.05, 400), (0.10, 220), (0.20, 120)]  # (g, horizon T)
RATIOS = [1.05, 1.20]
MODELS = [("Poisson", None), ("NB kappa=0.5", 0.5)]
KS = [1, 2, 3, 5, 10, 20, 50, 100]  # alpha = 10^-k
N_RHO = 60


def d_kappa(x, y, kappa):
    if kappa is None:
        return x * np.log(x / y) - x + y
    return x * np.log(x * (kappa + y) / (y * (kappa + x))) - kappa * np.log((kappa + x) / (kappa + y))


def ab(rho, r0, kappa):
    if kappa is None:
        return np.log(rho / r0), rho - r0
    return (np.log(rho * (kappa + r0) / (r0 * (kappa + rho))),
            kappa * np.log((kappa + rho) / (kappa + r0)))


def log_mixture(N, B, r0, kappa, rho_grid):
    """Running log of a uniform mixture over rho, accumulated to save memory."""
    acc = None
    for rho in rho_grid:
        a, b = ab(rho, r0, kappa)
        le = a * N - b * B
        acc = le if acc is None else np.logaddexp(acc, le)
    return acc - np.log(len(rho_grid))


def nb_or_pois_logpmf(y, mu, kappa_size):
    if kappa_size is None:
        return y * np.log(mu) - mu - gammaln(y + 1)
    k = kappa_size
    return (gammaln(y + k) - gammaln(k) - gammaln(y + 1)
            + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))


def crossing_times(logE, levels):
    out = np.full((logE.shape[0], len(levels)), np.inf)
    for j, lev in enumerate(levels):
        hit = logE >= lev
        anyh = hit.any(axis=1)
        out[anyh, j] = np.argmax(hit[anyh], axis=1) + 1.0
    return out


def run(g, T, ratio, model, kappa, reps, rng):
    w0 = disc_gamma_weights(5.0)
    R = renewal_R(w0, g)
    means = np.linspace(M_LO, M_HI, N_GRID)
    W = np.column_stack([disc_gamma_weights(m) for m in means])
    Rmin = renewal_R(W[:, 0], g)
    r0 = Rmin / ratio
    sim = simulate_renewal(rng, w0, R, T, reps, kappa=kappa, W_extra=W)
    Y, lam0, lamW = sim["Y"], sim["lam0"], sim["lamW"]
    surv = Y[:, -1] > 0
    Y, lam0, lamW = Y[surv], lam0[surv], lamW[surv]
    N = np.cumsum(Y, axis=1)
    A0 = np.cumsum(lam0, axis=1)
    B = np.cumsum(lamW, axis=1).max(axis=2)
    rho_grid = np.linspace(r0 * 1.001, 3 * r0, N_RHO)
    logE = log_mixture(N, B, r0, kappa, rho_grid)

    # oracle: true model vs least-favourable simple null (r0, w(m_lo), same kappa)
    lam_star = lamW[:, :, 0]
    ksz_p = None if kappa is None else kappa * lam0
    ksz_q = None if kappa is None else kappa * lam_star
    llr = (nb_or_pois_logpmf(Y, R * lam0, ksz_p) - nb_or_pois_logpmf(Y, r0 * lam_star, ksz_q))
    logO = np.cumsum(llr, axis=1)

    levels = np.array([k * np.log(10.0) for k in KS])  # log(1/alpha) = b_alpha
    tau = crossing_times(logE, levels)
    tau_or = crossing_times(logO, levels)
    I = (R / Rmin) * d_kappa(Rmin, r0, kappa)
    K_hat = A0[:, -1] * np.exp(-g * T)
    tau_hat = (np.log(levels)[None, :] - np.log(I * K_hat)[:, None]) / g

    rows = []
    for j, k in enumerate(KS):
        fin = np.isfinite(tau[:, j])
        res = tau[fin, j] - tau_hat[fin, j]
        rows.append(dict(g=g, model=model, ratio=ratio, R=R, R_min=Rmin, r0=r0, I_theory=I,
                         alpha_exp=k, log_b=np.log(levels[j]),
                         share_crossed=float(fin.mean()),
                         tau_median=float(np.median(tau[:, j])),
                         tau_oracle_median=float(np.median(tau_or[:, j])),
                         resid_median=float(np.median(res)) if fin.any() else np.nan,
                         resid_iqr=float(np.subtract(*np.quantile(res, [.75, .25]))) if fin.any() else np.nan,
                         n_paths=int(surv.sum()), T=T))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="200 reps for a fast check")
    args = ap.parse_args()
    reps = 200 if args.quick else args.reps
    OUT.mkdir(exist_ok=True)
    k = 0
    rows = []
    t0 = time.time()
    for g, T in SETTINGS:
        for ratio in RATIOS:
            for model, kappa in MODELS:
                rows += run(g, T, ratio, model, kappa, reps, scenario_rng(SEED, k)); k += 1
                print(f"g={g} ratio={ratio} {model} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp3_far_growth.csv", index=False)

    # slope of median tau on log b over the small-alpha range (k >= 10)
    srows = []
    for (g, model, ratio), sub in df.groupby(["g", "model", "ratio"]):
        s = sub[(sub.alpha_exp >= 10) & np.isfinite(sub.tau_median)]
        so = sub[(sub.alpha_exp >= 10) & np.isfinite(sub.tau_oracle_median)]
        slope = np.polyfit(s.log_b, s.tau_median, 1)[0]
        slope_or = np.polyfit(so.log_b, so.tau_oracle_median, 1)[0]
        srows.append(dict(g=g, model=model, ratio=ratio, one_over_g=1 / g,
                          slope_safe=slope, slope_oracle=slope_or,
                          slope_safe_times_g=slope * g, slope_oracle_times_g=slope_or * g,
                          resid_median_range=float(s.resid_median.max() - s.resid_median.min())))
    sl = pd.DataFrame(srows)
    sl.to_csv(OUT / "exp3_slopes.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for (g, model, ratio), sub in df.groupby(["g", "model", "ratio"]):
        if ratio != 1.05:
            continue
        ls = "-" if model == "Poisson" else "--"
        line, = axes[0].plot(sub.log_b, g * sub.tau_median, ls, marker="o", ms=3,
                             label=f"g={g}, {model}")
        axes[0].plot(sub.log_b, g * sub.tau_oracle_median, ls, color=line.get_color(), alpha=.35)
        axes[1].plot(sub.log_b, sub.resid_median * g, ls, marker="o", ms=3, color=line.get_color())
    xs = np.array([df.log_b.min(), df.log_b.max()])
    axes[0].plot(xs, xs - xs[0] + 1, ":", c="k", lw=1, label="slope 1 (theory)")
    axes[0].set_xlabel(r"$\log\log(1/\alpha)$"); axes[0].set_ylabel(r"$g\,\tau_\alpha$ (median)")
    axes[0].set_title(r"Safe detector (dark) and oracle (light), $R_{\min}/r_0=1.05$", fontsize=10)
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, c="grey", lw=.8)
    axes[1].set_xlabel(r"$\log\log(1/\alpha)$")
    axes[1].set_ylabel(r"$g\,(\tau_\alpha-\hat\tau_\alpha)$, median")
    axes[1].set_title("Residual from the Theorem 8.1 expansion (per-path $K$)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "exp3_far_growth.pdf"); fig.savefig(OUT / "exp3_far_growth.png", dpi=150)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(sl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
