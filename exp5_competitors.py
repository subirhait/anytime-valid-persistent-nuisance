"""Experiment 5: competitor comparison.

Detectors (all alarm the first day their statistic crosses its threshold):
  oracle LR            log-likelihood ratio of the true model vs (r0, true GI); GI and R known.
                       Valid only against that single simple null; a benchmark, not a method.
                       Under the nulls its design R is fixed at 1.617.
  safe, GI known       our e-process with B_t = A_t(w0) (true GI known exactly).
  safe robust (ours)   our e-process with B_t = sup_{w in W} A_t(w), W = Gamma means [3.5, 8].
  pointwise robust     like ours, but B_t = sum_j sup_w Lambda_j(w): the GI may change every day.
  plug-in naive R-hat  predictable plug-in bet rho_t = clip(N_{t-1}/A_{t-1}(w_nom)), i.e. an
                       estimate of R under the nominal GI, with robust compensator
                       exp{sum a(rho_j) Y_j - sup_w sum b(rho_j) Lambda_j(w)}. Valid (dominated by
                       the fixed-w supermartingale at the true w), but bets on R, not theta_eff.
  plug-in robust theta-hat  same, with rho_t = clip(N_{t-1}/B_{t-1}), an estimate of the
                       growth-optimal bet theta_eff = R_min(g) (Theorem 3.1).
  EpiEstim-style Rt    Cori et al. (2013) Gamma posterior for R_t over a 7-day window
                       (prior shape 1, scale 5), true GI; alarm when P(R_t > r0 | data) > 0.95.
  growth-rate          OLS slope of log(Y + 0.5) over a 14-day window; alarm when the one-sided
                       95% lower bound exceeds g0, the growth rate implied by r0 under the true GI.
The e-processes use alpha = 0.05 (threshold 1/alpha). The last two are not anytime-valid.

Scenarios (true GI: Gamma mean 5 d, CV 0.4, 15 lags; r0 = 1.30; T = 150 days):
  null boundary R = r0;  strict null R = 0.95 r0;  alternatives R = 1.45 and R = 1.617 (g = 0.10).
Offspring: Poisson and branching NB with kappa0 = kappa_- = 0.5.

Usage: python exp5_competitors.py [--reps 2000] [--quick]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammaln, gammaincc
from scipy.optimize import brentq
from sim_common import scenario_rng, disc_gamma_weights, renewal_R, simulate_renewal

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261025
S, INIT, T = 15, 20.0, 150
R0, ALPHA = 1.30, 0.05
R_DESIGN = 1.617
SCENARIOS = [("null boundary R=r0", R0), ("strict null R=0.95 r0", 0.95 * R0),
             ("alt R=1.45", 1.45), ("alt R=1.617 (g=0.10)", 1.617)]
MODELS = [("Poisson", None), ("NB kappa=0.5", 0.5)]
N_RHO = 60
EPI_WIN, EPI_A, EPI_SCALE, EPI_PROB = 7, 1.0, 5.0, 0.95
GR_WIN, Z95 = 14, 1.6448536
DETECTORS = ["oracle LR", "safe, GI known", "safe robust (ours)", "pointwise robust",
             "plug-in naive R-hat", "plug-in robust theta-hat", "EpiEstim-style Rt", "growth-rate"]


def ab(rho, kappa):
    if kappa is None:
        return np.log(rho / R0), rho - R0
    return (np.log(rho * (kappa + R0) / (R0 * (kappa + rho))),
            kappa * np.log((kappa + rho) / (kappa + R0)))


def lagged(Y):
    """L[:, t, s-1] = Y_{t-s} for days t = 1..T (initial history = INIT)."""
    reps, Tn = Y.shape
    full = np.concatenate([np.full((reps, S), INIT), Y], axis=1)
    return np.stack([full[:, S - s:S - s + Tn] for s in range(1, S + 1)], axis=-1)


def mixture_logE(N, B, kappa):
    acc = None
    for rho in np.linspace(R0 * 1.001, 3 * R0, N_RHO):
        a, b = ab(rho, kappa)
        le = a * N - b * B
        acc = le if acc is None else np.logaddexp(acc, le)
    return acc - np.log(N_RHO)


def logpmf(y, mu, kappa_size):
    if kappa_size is None:
        return y * np.log(mu) - mu - gammaln(y + 1)
    k = kappa_size
    return gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu))


def window_sum(X, w):
    c = np.cumsum(X, axis=1)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def first_day(alarm, start=1):
    alarm = alarm.copy()
    alarm[:, :start - 1] = False
    anyh = alarm.any(axis=1)
    return np.where(anyh, np.argmax(alarm, axis=1) + 1.0, np.inf)


def detect_days(Y, w0, W, kappa, R_oracle, g0):
    L = lagged(Y)
    lam0 = L @ w0
    lamW = L @ W                      # (reps, T, M)
    N = np.cumsum(Y, axis=1)
    lev = np.log(1 / ALPHA)
    days = {}
    # oracle LR
    kp = None if kappa is None else kappa * lam0
    llr = np.cumsum(logpmf(Y, R_oracle * lam0, kp) - logpmf(Y, R0 * lam0, kp), axis=1)
    days["oracle LR"] = first_day(llr >= lev)
    # safe, GI known
    days["safe, GI known"] = first_day(mixture_logE(N, np.cumsum(lam0, axis=1), kappa) >= lev)
    # safe robust (ours): sup over persistent w of cumulative exposure
    B = np.cumsum(lamW, axis=1).max(axis=2)
    days["safe robust (ours)"] = first_day(mixture_logE(N, B, kappa) >= lev)
    # pointwise robust: sup inside the sum
    Bpw = np.cumsum(lamW.max(axis=2), axis=1)
    days["pointwise robust"] = first_day(mixture_logE(N, Bpw, kappa) >= lev)
    # predictable plug-ins with robust compensator
    A_nom = np.cumsum(lam0, axis=1)
    for name, denom in [("plug-in naive R-hat", A_nom), ("plug-in robust theta-hat", B)]:
        prev_ratio = np.concatenate([np.full((Y.shape[0], 1), R0 * 1.2),
                                     N[:, :-1] / np.maximum(denom[:, :-1], 1e-12)], axis=1)
        rho_t = np.clip(prev_ratio, R0 * 1.001, 3 * R0)
        a_t, b_t = ab(rho_t, kappa)
        comp = np.cumsum(b_t[..., None] * lamW, axis=1).max(axis=2)
        days[name] = first_day(np.cumsum(a_t * Y, axis=1) - comp >= lev)
    # EpiEstim-style
    sY = window_sum(Y, EPI_WIN)
    sL = window_sum(lam0, EPI_WIN)
    shape = EPI_A + sY
    rate = 1.0 / EPI_SCALE + sL
    p_exceed = gammaincc(shape, rate * R0)
    days["EpiEstim-style Rt"] = first_day(p_exceed > EPI_PROB, start=EPI_WIN + 1)
    # growth-rate monitoring (rolling OLS of log(Y+0.5) on time)
    ly = np.log(Y + 0.5)
    x = np.arange(1, Y.shape[1] + 1, dtype=float)[None, :]
    n = GR_WIN
    Sx, Sy = window_sum(np.broadcast_to(x, Y.shape), n), window_sum(ly, n)
    Sxx, Sxy = window_sum(np.broadcast_to(x ** 2, Y.shape), n), window_sum(x * ly, n)
    Syy = window_sum(ly ** 2, n)
    sxx = Sxx - Sx ** 2 / n
    slope = (Sxy - Sx * Sy / n) / sxx
    rss = np.maximum(Syy - Sy ** 2 / n - slope ** 2 * sxx, 0.0)
    se = np.sqrt(rss / (n - 2) / sxx)
    days["growth-rate"] = first_day(slope - Z95 * se > g0, start=n)
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="400 reps for a fast check")
    args = ap.parse_args()
    reps = 400 if args.quick else args.reps
    OUT.mkdir(exist_ok=True)
    w0 = disc_gamma_weights(5.0)
    W = np.column_stack([disc_gamma_weights(m) for m in np.linspace(3.5, 8.0, 31)])
    g0 = brentq(lambda g: renewal_R(w0, g) - R0, 1e-6, 1.0)
    rows = []
    t0 = time.time()
    for mi, (model, kappa) in enumerate(MODELS):
        for si, (scen, R) in enumerate(SCENARIOS):
            sim = simulate_renewal(scenario_rng(SEED, mi, si), w0, R, T, reps, init=INIT, kappa=kappa)
            is_null = R <= R0 + 1e-12
            R_or = R_DESIGN if is_null else R
            days = detect_days(sim["Y"], w0, W, kappa, R_or, g0)
            for det in DETECTORS:
                d = days[det]
                rows.append(dict(model=model, scenario=scen, R_true=R, is_null=is_null,
                                 detector=det, alarm_prob=float(np.isfinite(d).mean()),
                                 median_day=float(np.median(d)),
                                 q25_day=float(np.quantile(d, .25, method="inverted_cdf")),
                                 q75_day=float(np.quantile(d, .75, method="inverted_cdf"))))
            print(f"{model} {scen} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp5_competitors.csv", index=False)

    # wide summary
    fa = df[df.scenario == "null boundary R=r0"].pivot(index="detector", columns="model", values="alarm_prob")
    fa2 = df[df.scenario == "strict null R=0.95 r0"].pivot(index="detector", columns="model", values="alarm_prob")
    d1 = df[df.scenario == "alt R=1.45"].pivot(index="detector", columns="model", values="median_day")
    d2 = df[df.scenario == "alt R=1.617 (g=0.10)"].pivot(index="detector", columns="model", values="median_day")
    summ = pd.concat({"false alarm, R=r0": fa, "false alarm, R=0.95 r0": fa2,
                      "median day, R=1.45": d1, "median day, R=1.617": d2}, axis=1).loc[DETECTORS]
    summ.to_csv(OUT / "exp5_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = np.arange(len(DETECTORS))
    for k, (model, _) in enumerate(MODELS):
        off = (k - 0.5) * 0.35
        axes[0].bar(x + off, summ[("false alarm, R=r0", model)], 0.35, label=model)
        dd = summ[("median day, R=1.617", model)].replace(np.inf, np.nan)
        axes[1].bar(x + off, dd, 0.35, label=model)
    axes[0].axhline(ALPHA, ls="--", c="k", lw=1)
    axes[0].set_ylabel(r"P(ever alarm), null boundary $R=r_0$")
    axes[1].set_ylabel(r"median detection day, $R=1.617$")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(DETECTORS, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp5_competitors.pdf"); fig.savefig(OUT / "exp5_competitors.png", dpi=150)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(summ.round(4).to_string())


if __name__ == "__main__":
    main()
