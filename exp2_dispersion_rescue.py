"""Experiment 2: dispersion rescue against the composite predictable null (Proposition 5.2).

Alternative: branching-consistent negative-binomial renewal epidemic
    Y_t | F ~ NB(mean R Lambda_t(w0), size kappa0 Lambda_t(w0)),
with a realistic discretised-Gamma GI (mean 5 d, CV 0.4, 15 lags) and g = 0.10.
The nuisance class is the Gamma family with mean in [m_lo, 8]; m_lo is set so that
R_min(g) = r0/1.10, i.e. we are inside the predictable mean-matching interior,
where the Poisson model has only finite information (Experiment 1).

For each scenario we evaluate a grid of persistent nulls (w, kappa') with the
mean-matching predictable signal r_t = min{r0, R Lambda_t(w0)/Lambda_t(w)}:
    w      : R_w(g) = r0/(1+delta), delta in {0.01, 0.03, 0.08}  (exposure ratio c = R/R_w)
    kappa' : kappa_-, 1.5 kappa_-, 3 kappa_-, and the size-matching value kappa0/c when admissible.
The empirical information rate of each null is the mean slope of the log-likelihood
ratio log dP/dQ over the last WINDOW days (its expectation is the one-step KL).

Theory: the rate of the null (c, kappa') is J(c,kappa') = (nu - 1 - log nu)/2 with
    nu = (1 + R/kappa0) / (1 + R/(kappa' c)),
and the least-favourable rate over all predictable nulls is
    J_LF = (nu_LF - 1 - log nu_LF)/2,  nu_LF = (kappa0 + R)/(kappa0 + chi_kappa R)   if chi_kappa < 1,
    0                                                                       if chi_kappa >= 1,
where chi_kappa = kappa0 r0 / (kappa_- R). The boundary is chi_kappa = 1, i.e. kappa0 r0 = kappa_- R.
Since kappa0 >= kappa_-, chi_kappa >= r0/R, so the rescue region is r0/R <= chi_kappa < 1.

Usage: python exp2_dispersion_rescue.py [--reps 2000] [--quick]
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
from sim_common import scenario_rng, disc_gamma_weights, renewal_R, mean_for_R, simulate_renewal

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261022
G = 0.10
T, WINDOW = 200, 80
R0_LIST = [1.40, 1.25]
KAPPA0_LIST = [0.2, 0.5, 2.0]
CHI_LIST = [0.80, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.20, 1.50]
DELTAS = [0.01, 0.03, 0.08]
KMULTS = [1.0, 1.5, 3.0]


def nb_logpmf(y, mu, k):
    return (gammaln(y + k) - gammaln(k) - gammaln(y + 1)
            + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))


def J_of(nu):
    return 0.5 * (nu - 1.0 - np.log(nu))


def run_scenario(r0, kappa0, chi_kappa, reps, rng):
    w0 = disc_gamma_weights(5.0)
    R = renewal_R(w0, G)
    kappa_minus = kappa0 * r0 / (chi_kappa * R)
    m_lo = mean_for_R(G, r0 / 1.10)
    Ws = np.column_stack([disc_gamma_weights(mean_for_R(G, r0 / (1 + d))) for d in DELTAS])
    sim = simulate_renewal(rng, w0, R, T, reps, kappa=kappa0, W_extra=Ws)
    Y, lam0, lamW = sim["Y"], sim["lam0"], sim["lamW"]
    alive = (lam0[:, -WINDOW - 1:] > 0).all(axis=1) & (Y[:, -1] > 0)
    Y, lam0, lamW = Y[alive], lam0[alive], lamW[alive]
    mu_p = R * lam0
    ll_p = nb_logpmf(Y, mu_p, kappa0 * lam0)
    rows = []
    sl = slice(T - WINDOW, T)
    for j, d in enumerate(DELTAS):
        lw = lamW[:, :, j]
        Rw = r0 / (1 + d)
        c = R / Rw
        r_t = np.minimum(r0, mu_p / lw)
        clipped_share = float((mu_p[:, sl] / lw[:, sl] > r0).mean())
        kps = [(f"{m:g}x kappa_-", m * kappa_minus) for m in KMULTS]
        if kappa0 / c >= kappa_minus:
            kps.append(("size-matched kappa0/c", kappa0 / c))
        for label, kp in kps:
            ll_q = nb_logpmf(Y, r_t * lw, kp * lw)
            inc = (ll_p - ll_q)[:, sl]
            slope = inc.mean(axis=1)
            nu = (1 + R / kappa0) / (1 + R / (kp * c))
            rows.append(dict(r0=r0, kappa0=kappa0, chi_kappa=chi_kappa, kappa_minus=kappa_minus, R=R,
                             delta=d, c=c, null_kappa=label, kappa_prime=kp,
                             J_theory=float(J_of(nu)),
                             J_empirical=float(slope.mean()),
                             J_empirical_se=float(slope.std(ddof=1) / np.sqrt(len(slope))),
                             clipped_share_window=clipped_share,
                             n_paths=int(alive.sum()), m_lo=m_lo))
    nu_lf = (kappa0 + R) / (kappa0 + chi_kappa * R)
    J_LF = float(J_of(nu_lf)) if chi_kappa < 1 else 0.0
    for r in rows:
        r["J_LF"] = J_LF
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="300 reps for a fast check")
    args = ap.parse_args()
    reps = 300 if args.quick else args.reps
    OUT.mkdir(exist_ok=True)
    k = 0
    rows = []
    t0 = time.time()
    for r0 in R0_LIST:
        R = renewal_R(disc_gamma_weights(5.0), G)
        for kappa0 in KAPPA0_LIST:
            chis = [r0 / R] + [x for x in CHI_LIST if x > r0 / R + 1e-9]  # chi_kappa = r0/R is known kappa
            for chi_kappa in chis:
                rows += run_scenario(r0, kappa0, chi_kappa, reps, scenario_rng(SEED, k)); k += 1
                print(f"r0={r0} kappa0={kappa0} chi_kappa={chi_kappa:.2f} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "exp2_all_nulls.csv", index=False)

    # least favourable over the null grid, per scenario
    key = ["r0", "kappa0", "chi_kappa"]
    idx = df.groupby(key)["J_theory"].idxmin()  # LF null in the grid chosen by theory, not by noisy minimum
    lf = df.loc[idx, key + ["kappa_minus", "J_LF", "J_empirical", "J_empirical_se",
                            "J_theory", "delta", "null_kappa", "clipped_share_window", "n_paths"]]
    lf = lf.rename(columns={"J_empirical": "J_min_empirical", "J_theory": "J_theory_of_min_null",
                            "delta": "delta_of_min_null", "null_kappa": "kappa_of_min_null"})
    lf["days_to_3_nats_at_J_LF"] = np.where(lf.J_LF > 0, 3.0 / lf.J_LF.where(lf.J_LF > 0, 1), np.inf)
    lf["known_kappa"] = np.isclose(lf.kappa_minus, lf.kappa0)
    lf.to_csv(OUT / "exp2_least_favourable.csv", index=False)
    # agreement across all grid nulls
    z = (df.J_empirical - df.J_theory) / df.J_empirical_se
    print(f"All {len(df)} grid nulls: |z|<2 for {np.mean(np.abs(z)<2):.3f}; corr(theory, empirical)={np.corrcoef(df.J_theory, df.J_empirical)[0,1]:.4f}")

    # figure
    fig, axes = plt.subplots(1, len(R0_LIST), figsize=(10, 4.2), sharey=True)
    markers = {0.2: "o", 0.5: "s", 2.0: "^"}
    for ax, r0 in zip(np.atleast_1d(axes), R0_LIST):
        R = renewal_R(disc_gamma_weights(5.0), G)
        for kappa0 in KAPPA0_LIST:
            sub = lf[(lf.r0 == r0) & (lf.kappa0 == kappa0)].sort_values("chi_kappa")
            rr = np.linspace(r0 / R, max(CHI_LIST), 300)
            th = np.where(rr < 1, J_of((kappa0 + R) / (kappa0 + rr * R)), 0.0)
            line, = ax.plot(rr, th, lw=1)
            ax.errorbar(sub.chi_kappa, sub.J_min_empirical, yerr=2 * sub.J_empirical_se,
                        fmt=markers[kappa0], color=line.get_color(), ms=5,
                        label=fr"$\kappa_0={kappa0}$")
            ax.plot(sub.chi_kappa, sub.J_theory_of_min_null, markers[kappa0], mfc="none",
                    color=line.get_color(), ms=9, alpha=.6)
        ax.axvline(1.0, ls=":", c="grey")
        ax.set_title(fr"$r_0={r0}$, $R={R:.3f}$, $R_{{\min}}=r_0/1.10$")
        ax.set_xlabel(r"$\chi_\kappa=\kappa_0 r_0/(\kappa_- R)$")
    np.atleast_1d(axes)[0].set_ylabel("least-favourable information rate (nats/day)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Dispersion rescue: lines = $J_{LF}$ (theory); filled = empirical rate of least-favourable grid null ($\\pm$2 SE); hollow = its theoretical rate", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "exp2_dispersion_rescue.pdf"); fig.savefig(OUT / "exp2_dispersion_rescue.png", dpi=150)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(lf.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
