"""Experiment 1: predictable-null impossibility (Theorem 5.1) and its boundary.

Poisson renewal alternative with a realistic discretised-Gamma GI (mean 5 d, CV 0.4,
15 lags). The nuisance class is the Gamma family with mean m in [m_lo, 8], CV fixed.
m_lo is chosen so that R_min(g) = ratio * r0 for
    ratio in {0.95, 0.98, 1.00, 1.02, 1.05, 1.10},
covering the strict interior (ratio < 1), the boundary (= 1), and the positive
first-order region (ratio > 1).

For each path we compute
  * the predictable compensator sum_j KL(P_j || Q_j) against the least-favourable
    predictable null Q: persistent w* = w(m_lo), r_t = min{r0, R Lambda_t(w0)/Lambda_t(w*)};
  * the last day on which r_t is clipped at r0 (mean matching not yet possible);
  * the robust safe Poisson e-process (uniform mixture over rho) with
    B_t = sup_{w in class} A_t(w), and its crossing times at several alpha.

Theory (Theorem 5.1, Proposition 3.2):
  ratio < 1 : compensator converges to a finite limit; detection probability -> 0 as alpha -> 0
  ratio > 1 : compensator / A_t(w0) -> c* d(R_min || r0), c* = R / R_min
  ratio = 1 : boundary; first-order rate is zero, lower-order growth only.

Usage: python exp1_predictable_null.py [--reps 1000] [--quick]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_common import (scenario_rng, disc_gamma_weights, renewal_R, mean_for_R, poisson_kl,
                        simulate_renewal, safe_poisson_log_mixture, first_crossing)

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261021
RATIOS = [0.95, 0.98, 1.00, 1.02, 1.05, 1.10]
SETTINGS = [  # (g, r0, T)
    (0.10, 1.40, 150),
    (0.05, 1.12, 250),
]
M_NOMINAL, M_HI, N_GRID = 5.0, 8.0, 41
ALPHAS = [0.05, 1e-3, 1e-6]
CHECKPOINTS = [50, 100, 150, 200, 250]


def run_setting(g, r0, T, ratio, reps, rng):
    w0 = disc_gamma_weights(M_NOMINAL)
    R = renewal_R(w0, g)
    Rmin = ratio * r0
    m_lo = mean_for_R(g, Rmin)
    means = np.linspace(m_lo, M_HI, N_GRID)
    W = np.column_stack([disc_gamma_weights(m) for m in means])  # S x M, column 0 = w*
    sim = simulate_renewal(rng, w0, R, T, reps, W_extra=W)
    Y, lam0, lamW = sim["Y"], sim["lam0"], sim["lamW"]
    lam_star = lamW[:, :, 0]
    mu_p = R * lam0

    # least-favourable predictable null
    raw = mu_p / lam_star
    r_t = np.minimum(r0, raw)
    kl = poisson_kl(mu_p, r_t * lam_star)
    comp = np.cumsum(kl, axis=1)
    clipped = raw > r0
    last_clip = np.where(clipped.any(axis=1), T - np.argmax(clipped[:, ::-1], axis=1), 0)

    # safe robust Poisson e-process
    N = np.cumsum(Y, axis=1)
    A0 = np.cumsum(lam0, axis=1)
    B = np.cumsum(lamW, axis=1).max(axis=2)
    rho = np.linspace(r0 + 0.01, r0 + 1.5, 60)
    logE = safe_poisson_log_mixture(N, B, r0, rho)

    cstar = R / Rmin
    d_inf = Rmin * np.log(Rmin / r0) - Rmin + r0
    rate_theory = cstar * d_inf if ratio > 1 else 0.0

    row = dict(g=g, r0=r0, ratio=ratio, R=R, R_min=Rmin, m_lo=m_lo, T=T, reps=reps,
               extinct_share=float((Y[:, -1] == 0).mean()),
               last_clip_median=float(np.median(last_clip)),
               last_clip_p95=float(np.quantile(last_clip, .95)),
               share_clipped_at_T=float(clipped[:, -1].mean()),
               rate_theory=rate_theory,
               comp_rate_T_median=float(np.median(comp[:, -1] / A0[:, -1])),
               logE_rate_T_median=float(np.median(logE[:, -1] / A0[:, -1])),
               comp_slope_last50_median=float(np.median((comp[:, -1] - comp[:, -51]) / 50.0)))
    for c in CHECKPOINTS:
        if c <= T:
            row[f"comp_d{c}_median"] = float(np.median(comp[:, c - 1]))
            row[f"comp_d{c}_p95"] = float(np.quantile(comp[:, c - 1], .95))
    for a in ALPHAS:
        tau = first_crossing(logE, np.log(1 / a))
        row[f"detect_prob_{a:g}"] = float(np.isfinite(tau).mean())
        row[f"detect_day_median_{a:g}"] = float(np.median(tau)) if np.isfinite(tau).mean() > .5 else np.inf
    curves = dict(comp_median=np.median(comp, axis=0), comp_p95=np.quantile(comp, .95, axis=0),
                  logE_median=np.median(logE, axis=0))
    return row, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="200 reps for a fast check")
    args = ap.parse_args()
    reps = 200 if args.quick else args.reps
    OUT.mkdir(exist_ok=True)
    rows, curves = [], {}
    t0 = time.time()
    for g, r0, T in SETTINGS:
        for ratio in RATIOS:
            rng = scenario_rng(SEED, len(rows))
            row, cv = run_setting(g, r0, T, ratio, reps, rng)
            rows.append(row); curves[(g, ratio)] = cv
            print(f"g={g} ratio={ratio:.2f} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp1_summary.csv", index=False)

    # figure: median predictable compensator and median log e-process
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex="col")
    cmap = plt.get_cmap("viridis")
    for j, (g, r0, T) in enumerate(SETTINGS):
        days = np.arange(1, T + 1)
        for i, ratio in enumerate(RATIOS):
            col = cmap(i / (len(RATIOS) - 1))
            cvs = curves[(g, ratio)]
            axes[0, j].plot(days, np.maximum(cvs["comp_median"], 1e-3), color=col,
                            label=fr"$R_{{\min}}/r_0={ratio:.2f}$")
            axes[1, j].plot(days, cvs["logE_median"], color=col)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(fr"$g={g}$, $r_0={r0}$")
        axes[0, j].set_ylabel("median cumulative KL vs LF predictable null")
        axes[1, j].axhline(np.log(1 / 0.05), ls=":", c="grey")
        axes[1, j].axhline(np.log(1e6), ls="--", c="grey")
        axes[1, j].set_ylabel("median log safe e-process")
        axes[1, j].set_xlabel("day")
        axes[1, j].set_ylim(-60, 60)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp1_predictable_null.pdf"); fig.savefig(OUT / "exp1_predictable_null.png", dpi=150)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.T.to_string())


if __name__ == "__main__":
    main()
