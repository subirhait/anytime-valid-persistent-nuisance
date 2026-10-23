"""Experiment 6: anytime-valid false-alarm control under adversarial predictable nulls.

Theorem 2.3 claims sup over the whole predictable null H0^pred of P(ever alarm) <= alpha.
No simulation can search all of H0^pred, so we stress the safe robust detector
(Gamma GI class, means [3.5, 8]; kappa_- = 0.5 for NB data) with a battery of null strategies.
Every strategy keeps r_t <= r0 = 1.30, uses a persistent GI inside the class, and a
persistent dispersion kappa >= kappa_-, so all are members of H0^pred, except the last one
(GI switching daily), which lies outside the persistent-nuisance model and is reported as a
robustness check.

Strategies:
  const r0, GI mean 5 / 3.5 / 8          the null boundary with each GI in the class
  const r0, GI 3.5, surge history        initial history from a recent fast-growing wave
  weekly oscillation                     r_t alternates r0 and 0.7 r0 every 7 days
  momentum                               r_t = r0 after an above-mean count, else 0.8 r0
  e-chasing                              r_t = r0 while log E_{t-1} > -1, else 0.6 r0
  mean-matching (Thm 5.1)                r_t = min{r0, 1.617 Lambda_t(w0)/Lambda_t(w*)}, w* = mean 3.5
  [outside model] GI switches daily      GI alternates means 3.5 and 8 each day, r_t = r0
  [positive control]                     same harness, but the detector assumes the GI is known
                                         (mean 5) while the data use mean 3.5; it must fail,
                                         showing the harness can detect invalidity
Data models: Poisson; NB with kappa = kappa_- = 0.5; NB with kappa = 2.5 (less dispersed than kappa_-).
Horizon T = 200 days, alpha in {0.05, 0.01, 0.001}.

Usage: python exp6_false_alarms.py [--reps 4000] [--quick]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sim_common import scenario_rng, disc_gamma_weights

OUT = Path(__file__).resolve().parent / "results"
SEED = 20261026
S, T, R0 = 15, 200, 1.30
KAPPA_MINUS = 0.5
ALPHAS = [0.05, 0.01, 1e-3]
N_RHO = 60
RHO = np.linspace(R0 * 1.001, 3 * R0, N_RHO)
MEANS_CLASS = np.linspace(3.5, 8.0, 31)
DATA_MODELS = [("Poisson", None), ("NB kappa=0.5 (=kappa_-)", 0.5), ("NB kappa=2.5 (>kappa_-)", 2.5)]

STRATEGIES = [
    ("const r0, GI mean 5", dict(m=5.0)),
    ("const r0, GI mean 3.5", dict(m=3.5)),
    ("const r0, GI mean 8", dict(m=8.0)),
    ("const r0, GI 3.5, surge history", dict(m=3.5, init="surge")),
    ("weekly oscillation", dict(m=3.5, rule="oscillate")),
    ("momentum", dict(m=3.5, rule="momentum")),
    ("e-chasing", dict(m=3.5, rule="echase")),
    ("mean-matching (Thm 5.1)", dict(m=3.5, rule="meanmatch")),
    ("[outside model] GI switches daily", dict(m=None, rule="switch")),
    ("[positive control] detector assumes GI 5, data GI 3.5", dict(m=3.5, det="nominal")),
]


def ab(rho, kappa):
    if kappa is None:
        return np.log(rho / R0), rho - R0
    return (np.log(rho * (kappa + R0) / (R0 * (kappa + rho))),
            kappa * np.log((kappa + rho) / (kappa + R0)))


def run(strategy, kappa_data, reps, rng):
    kappa_det = None if kappa_data is None else KAPPA_MINUS
    a_r, b_r = ab(RHO, kappa_det)                       # (N_RHO,)
    W = np.column_stack([disc_gamma_weights(m) for m in MEANS_CLASS])
    if strategy.get("det") == "nominal":  # misspecified detector: class = {GI mean 5}
        W = disc_gamma_weights(5.0)[:, None]
    w0 = disc_gamma_weights(5.0)
    w_short, w_long = disc_gamma_weights(3.5), disc_gamma_weights(8.0)
    m = strategy.get("m")
    w_true = None if m is None else disc_gamma_weights(m)
    rule = strategy.get("rule", "const")
    if strategy.get("init") == "surge":
        hist = np.tile(20.0 * np.exp(-0.2 * np.arange(S)), (reps, 1)) * 5  # recent wave
    else:
        hist = np.full((reps, S), 20.0)
    N = np.zeros(reps)
    Bcum = np.zeros((reps, W.shape[1]))
    logE = np.full(reps, 0.0)
    maxlogE = np.full(reps, -np.inf)
    prev_y = np.zeros(reps); prev_mu = np.ones(reps)
    for t in range(T):
        wt = (w_short if t % 2 == 0 else w_long) if rule == "switch" else w_true
        lam = hist @ wt
        if rule == "const" or rule == "switch":
            r = np.full(reps, R0)
        elif rule == "oscillate":
            r = np.full(reps, R0 if (t // 7) % 2 == 0 else 0.7 * R0)
        elif rule == "momentum":
            r = np.where(prev_y >= prev_mu, R0, 0.8 * R0) if t > 0 else np.full(reps, R0)
        elif rule == "echase":
            r = np.where(logE > -1.0, R0, 0.6 * R0)
        elif rule == "meanmatch":
            r = np.minimum(R0, 1.617 * (hist @ w0) / np.maximum(lam, 1e-12))
        mu = r * lam
        if kappa_data is None:
            y = rng.poisson(mu).astype(float)
        else:
            k = np.maximum(kappa_data * lam, 1e-300)
            y = rng.poisson(rng.gamma(shape=k, scale=np.where(lam > 0, mu / k, 0.0))).astype(float)
        # detector update (uses only data)
        N += y
        Bcum += hist @ W
        B = Bcum.max(axis=1)
        le = a_r[None, :] * N[:, None] - b_r[None, :] * B[:, None]
        m_ = le.max(axis=1, keepdims=True)
        logE = (m_[:, 0] + np.log(np.exp(le - m_).sum(axis=1))) - np.log(N_RHO)
        maxlogE = np.maximum(maxlogE, logE)
        prev_y, prev_mu = y, mu
        hist = np.concatenate([y[:, None], hist[:, :-1]], axis=1)
    return maxlogE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--quick", action="store_true", help="800 reps for a fast check")
    args = ap.parse_args()
    reps = 800 if args.quick else args.reps
    OUT.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    for dname, kd in DATA_MODELS:
        for sname, strat in STRATEGIES:
            mx = run(strat, kd, reps, scenario_rng(SEED, len(rows)))
            row = dict(data_model=dname, strategy=sname, in_null=not sname.startswith("[outside"),
                       valid_detector=not sname.startswith("[positive"), reps=reps)
            for a in ALPHAS:
                p = float((mx >= np.log(1 / a)).mean())
                row[f"false_alarm_{a:g}"] = p
                row[f"upper95_{a:g}"] = p + 1.96 * np.sqrt(max(p * (1 - p), a * (1 - a)) / reps)
            rows.append(row)
            print(f"{dname} | {sname} done ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    for a in ALPHAS:
        df[f"exceeds_{a:g}"] = df[f"false_alarm_{a:g}"] > a + 2 * np.sqrt(a * (1 - a) / reps)
    df.to_csv(OUT / "exp6_false_alarms.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    names = [s[0] for s in STRATEGIES]
    x = np.arange(len(names))
    for k, (dname, _) in enumerate(DATA_MODELS):
        sub = df[df.data_model == dname].set_index("strategy").loc[names]
        ax.bar(x + (k - 1) * 0.27, sub["false_alarm_0.05"], 0.27, label=dname)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.5)
    ax.axhline(0.05, ls="--", c="k", lw=1, label=r"$\alpha=0.05$")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P(ever alarm within 200 days)")
    ax.set_title("Safe robust detector under adversarial predictable nulls", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "exp6_false_alarms.pdf"); fig.savefig(OUT / "exp6_false_alarms.png", dpi=150)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[["data_model", "strategy", "false_alarm_0.05", "false_alarm_0.01", "false_alarm_0.001",
                  "exceeds_0.05", "exceeds_0.01", "exceeds_0.001"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
