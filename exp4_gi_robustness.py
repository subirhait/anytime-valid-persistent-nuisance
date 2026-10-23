"""Experiment 4: generation-interval (GI) uncertainty classes — validity and delay.

The safe e-process exp{a N_t - b B_t} needs B_t = sup_{w in class} A_t(w) to dominate
the true cumulative exposure A_t(w0). We compare uncertainty classes:
  correct        : {w0} (GI known exactly)
  gamma narrow   : Gamma family, mean in [4.5, 5.5], CV 0.4
  gamma wide     : Gamma family, mean in [3.5, 8.0], CV 0.4
  KL eps         : KL ball {w : KL(w || w0) <= eps}, eps in {0.02, 0.10, 0.30}
  missp. slightly long : Gamma family, mean in [5.5, 8.0]  (truth just outside)
  missp. long    : Gamma family, mean in [6.5, 8.0]  (excludes the truth; GI assumed too long)
  missp. short   : Gamma family, mean in [2.5, 3.5]  (excludes the truth; GI assumed too short)
True GI: discretised Gamma, mean 5 d, CV 0.4, 15 lags.

(a) Validity: data generated at the null boundary R = r0 = 1.30 (constant). We record the
    probability that the e-process ever crosses 1/alpha within T_NULL days.
(b) Power/delay: data generated with growth g = 0.10 (R = 1.617); median crossing day.
Both for Poisson offspring and branching NB with kappa0 = kappa_- = 0.5.

KL-ball suprema use the dual  sup w'z = inf_lam {lam eps + lam log sum_s w0_s e^{z_s/lam}}
evaluated on a lambda grid; any lambda gives an upper bound, so the grid version is
conservative and validity is preserved.

Usage: python exp4_gi_robustness.py [--reps 2000] [--quick]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sim_common import scenario_rng, disc_gamma_weights, renewal_R, simulate_renewal

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261024
S, INIT = 15, 20.0
R0 = 1.30
G_ALT = 0.10
T_NULL, T_ALT = 150, 120
ALPHAS = [0.05, 0.01, 1e-3]
MODELS = [("Poisson", None), ("NB kappa=0.5", 0.5)]
N_RHO = 60
N_LAMBDA = 60


def family(lo, hi, n=31):
    return np.column_stack([disc_gamma_weights(m) for m in np.linspace(lo, hi, n)])


CLASSES = [
    ("correct", ("family", None)),
    ("gamma narrow [4.5,5.5]", ("family", (4.5, 5.5))),
    ("gamma wide [3.5,8]", ("family", (3.5, 8.0))),
    ("KL eps=0.02", ("kl", 0.02)),
    ("KL eps=0.10", ("kl", 0.10)),
    ("KL eps=0.30", ("kl", 0.30)),
    ("missp. slightly long [5.5,8]", ("family", (5.5, 8.0))),
    ("missp. long [6.5,8]", ("family", (6.5, 8.0))),
    ("missp. short [2.5,3.5]", ("family", (2.5, 3.5))),
]


def lag_cumsums(Y, init):
    """Z[:, t, s-1] = sum_{j<=t} Y_{j-s}; Y has shape (reps, T) for days 1..T."""
    reps, T = Y.shape
    full = np.concatenate([np.full((reps, S), init), Y], axis=1)  # index S+t-1 is day t
    Z = np.zeros((reps, T, S))
    for s in range(1, S + 1):
        lagged = full[:, S - s:S - s + T]  # Y_{t-s} for t = 1..T
        Z[:, :, s - 1] = np.cumsum(lagged, axis=1)
    return Z


def kl_sup(Z, w0, eps):
    """Upper bound on sup_{KL(w||w0)<=eps} w'Z, via the dual on a lambda grid (vectorised)."""
    scale = Z.max(axis=-1, keepdims=True) + 1e-12
    best = np.full(Z.shape[:-1], np.inf)
    logw = np.log(w0)
    for f in np.logspace(-4, 3, N_LAMBDA):
        lam = f * scale
        val = lam[..., 0] * (eps + logsumexp(Z / lam + logw, axis=-1))
        best = np.minimum(best, val)
    return best


def kl_sup_vec(z, w0, eps):
    from scipy.optimize import minimize_scalar
    f = lambda ll: np.exp(ll) * (eps + logsumexp(z / np.exp(ll) + np.log(w0)))
    return minimize_scalar(f, bounds=(-12, 8), method="bounded").fun


def B_for_class(spec, Z, w0):
    kind, par = spec
    if kind == "family":
        W = w0[:, None] if par is None else family(*par)
        return (Z @ W).max(axis=-1)
    return kl_sup(Z, w0, par)


def R_min_class(spec, w0, g):
    s = np.arange(1, S + 1)
    z = np.exp(-g * s)
    kind, par = spec
    if kind == "family":
        W = w0[:, None] if par is None else family(*par)
        return 1.0 / (z @ W).max()
    return 1.0 / kl_sup_vec(z, w0, par)


def ab(rho, kappa):
    if kappa is None:
        return np.log(rho / R0), rho - R0
    return (np.log(rho * (kappa + R0) / (R0 * (kappa + rho))),
            kappa * np.log((kappa + rho) / (kappa + R0)))


def log_mixture(N, B, kappa):
    acc = None
    for rho in np.linspace(R0 * 1.001, 3 * R0, N_RHO):
        a, b = ab(rho, kappa)
        le = a * N - b * B
        acc = le if acc is None else np.logaddexp(acc, le)
    return acc - np.log(N_RHO)


def evaluate(Y, w0, kappa):
    Z = lag_cumsums(Y, INIT)
    N = np.cumsum(Y, axis=1)
    out = {}
    for name, spec in CLASSES:
        B = B_for_class(spec, Z, w0)
        logE = log_mixture(N, B, kappa)
        out[name] = (logE, B, Z @ w0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="400 reps for a fast check")
    args = ap.parse_args()
    reps = 400 if args.quick else args.reps
    OUT.mkdir(exist_ok=True)
    w0 = disc_gamma_weights(5.0)
    R_alt = renewal_R(w0, G_ALT)
    rows = []
    t0 = time.time()
    for mi, (model, kappa) in enumerate(MODELS):
        # (a) null boundary R = r0
        sim = simulate_renewal(scenario_rng(SEED, mi, 0), w0, R0, T_NULL, reps, init=INIT, kappa=kappa)
        res_null = evaluate(sim["Y"], w0, kappa)
        # (b) growth alternative
        sim = simulate_renewal(scenario_rng(SEED, mi, 1), w0, R_alt, T_ALT, reps, init=INIT, kappa=kappa)
        res_alt = evaluate(sim["Y"], w0, kappa)
        for name, spec in CLASSES:
            logE0, B0, A0 = res_null[name]
            logE1, _, _ = res_alt[name]
            Rmin = R_min_class(spec, w0, G_ALT)
            row = dict(model=model, gi_class=name, R_min_at_g=Rmin, R_alt=R_alt, r0=R0,
                       share_steps_B_below_true_A=float((B0 < A0 * (1 - 1e-9)).mean()))
            for a in ALPHAS:
                lev = np.log(1 / a)
                row[f"false_alarm_{a:g}"] = float((logE0.max(axis=1) >= lev).mean())
                hit = logE1 >= lev
                day = np.where(hit.any(axis=1), np.argmax(hit, axis=1) + 1.0, np.inf)
                row[f"power_{a:g}"] = float(np.isfinite(day).mean())
                row[f"median_day_{a:g}"] = float(np.median(day))
            rows.append(row)
        print(f"{model} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    se = np.sqrt(0.05 * 0.95 / reps)
    df["false_alarm_0.05_exceeds_alpha_by_2SE"] = df["false_alarm_0.05"] > 0.05 + 2 * se
    df.to_csv(OUT / "exp4_gi_robustness.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    names = [c[0] for c in CLASSES]
    x = np.arange(len(names))
    for k, (model, _) in enumerate(MODELS):
        sub = df[df.model == model].set_index("gi_class").loc[names]
        off = (k - 0.5) * 0.35
        axes[0].bar(x + off, sub["false_alarm_0.05"], 0.35, label=model)
        days = sub["median_day_0.05"].replace(np.inf, np.nan)  # never-detected -> no bar
        axes[1].bar(x + off, days, 0.35, label=model)
        for xi, d in zip(x, days):
            if np.isnan(d):
                axes[1].text(xi + off, 0.5, "never", rotation=90, ha="center", va="bottom", fontsize=7)
    axes[0].axhline(0.05, ls="--", c="k", lw=1)
    axes[0].set_ylabel(r"P(ever alarm) at null boundary $R=r_0$, $\alpha=0.05$")
    axes[1].set_ylabel(r"median detection day, $g=0.10$, $\alpha=0.05$")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp4_gi_robustness.pdf"); fig.savefig(OUT / "exp4_gi_robustness.png", dpi=150)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["model", "gi_class", "R_min_at_g", "share_steps_B_below_true_A", "false_alarm_0.05",
                "false_alarm_0.01", "false_alarm_0.001", "power_0.05", "median_day_0.05", "median_day_0.001"]
        print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
