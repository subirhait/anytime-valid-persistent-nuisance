"""Historical application: SARS 2003, Hong Kong (EpiEstim dataset SARS2003).

Question: when could growth above a threshold r0 have been declared with anytime-valid
evidence, and how much did generation-interval (GI) and offspring-dispersion uncertainty
delay that declaration?

Data: daily symptom-onset incidence (107 days, day 1 = first case) and the serial-interval
(SI) distribution shipped with EpiEstim (mean 8.4 d, SD 3.8 d). The SI is used as a proxy
for the GI, and all cases from the monitoring start onward are treated as locally acquired.
Monitoring starts at day t0, the first day with Lambda_t(SI) >= 0.5; earlier days only
build the infection history (the e-process is valid conditionally on F_{t0-1}).

GI classes (lags 1..24):
  known SI                  the EpiEstim SI distribution
  Gamma +/-1 d, +/-2 d      discretised Gamma, CV fixed at the SI's 0.45, means 8.4 +/- 1 or 2
  short side [5.4, 8.4]     widened toward shorter intervals (recommended direction)
  KL eps = 0.1, 0.3         KL balls around the SI distribution
  long side [8.4, 11.4]     widened only toward longer intervals (NOT valid if the truth is
                            shorter; shown to illustrate Remark 2.4)
Dispersion: Poisson, and branching NB calibrated at kappa_- in {1, 0.5, 0.16}
(0.16 is the Singapore SARS offspring-dispersion estimate reported by Lloyd-Smith et al. 2005; it is used here as a sensitivity value, not as a Hong Kong-specific estimate).
Comparators (not anytime-valid, daily looks): EpiEstim-style P(R_t > r0) > 0.95 on a 7-day
window (prior shape 1, scale 5), and a 14-day log-linear growth-rate lower bound > g0(r0).

Sensitivity: the Amoy Gardens outbreak (onset days 38-40 here) was largely a common-source
(environmental) event, not person-to-person branching. A third analysis starts on day 25 and
does not score days 38-40: those counts stay in the infection history but neither they nor
their exposure enter the e-process.

Usage: python sars_application.py [path/to/epi_data]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from scipy.special import logsumexp, gammaincc
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "epi_data"
OUT = HERE / "results"
R0_LIST = [1.0, 1.5, 2.0]
ALPHAS = [0.05, 0.01, 0.001]
KAPPAS = [("Poisson", None), ("NB kappa-=1", 1.0), ("NB kappa-=0.5", 0.5), ("NB kappa-=0.16", 0.16)]
N_RHO = 80


def load():
    y = pd.read_csv(DATA / "SARS2003_incidence.csv")["I"].to_numpy(float)
    si = pd.read_csv(DATA / "SARS2003_si_distr.csv")
    p = si["p"].to_numpy(float)
    w = p[1:] / p[1:].sum()  # lag 0 has zero mass; renewal uses lags 1..S
    return y, w


def disc_gamma(mean, cv, S):
    sd = cv * mean
    shape, scale = (mean / sd) ** 2, sd ** 2 / mean
    lo, up = np.arange(0.5, S + 0.5), np.arange(1.5, S + 1.5)
    w = gamma_dist.cdf(up, a=shape, scale=scale) - gamma_dist.cdf(lo, a=shape, scale=scale)
    w = np.maximum(w, 0)
    return w / w.sum()


def lag_matrix(y, S):
    full = np.r_[np.zeros(S), y]
    return np.stack([full[S - s:S - s + len(y)] for s in range(1, S + 1)], axis=1)  # (T, S)


def kl_sup_path(Z, wbar, eps, n_lam=200):
    """Upper bound on sup_{KL(w||wbar)<=eps} w'Z_t for each row of Z (dual on a grid)."""
    scale = Z.max(axis=1, keepdims=True) + 1e-12
    best = np.full(Z.shape[0], np.inf)
    lw = np.log(np.maximum(wbar, 1e-300))
    for f in np.logspace(-4, 3, n_lam):
        lam = f * scale
        best = np.minimum(best, lam[:, 0] * (eps + logsumexp(Z / lam + lw, axis=1)))
    return best


def ab(rho, r0, kappa):
    if kappa is None:
        return np.log(rho / r0), rho - r0
    return (np.log(rho * (kappa + r0) / (r0 * (kappa + rho))),
            kappa * np.log((kappa + rho) / (kappa + r0)))


def log_mixture(N, B, r0, kappa):
    rho = np.linspace(r0 * 1.001, 4 * r0, N_RHO)
    a, b = ab(rho, r0, kappa)
    return logsumexp(a[None, :] * N[:, None] - b[None, :] * B[:, None], axis=1) - np.log(N_RHO)


def first_day(mask, days):
    idx = np.flatnonzero(mask)
    return int(days[idx[0]]) if idx.size else None


def conventional_days(y, lam_si, w_si, lags, r0, t0, days):
    """First alarm day (daily looks from t0) of the two non-anytime-valid rules."""
    T = len(y)
    g0 = 0.0 if r0 == 1.0 else brentq(lambda g: 1.0 / np.sum(w_si * np.exp(-g * lags)) - r0, -0.5, 2.0)
    win = 7
    sY = pd.Series(y).rolling(win, min_periods=win).sum().to_numpy()
    sL = pd.Series(lam_si).rolling(win, min_periods=win).sum().to_numpy()
    pexc = gammaincc(1.0 + sY, (1 / 5.0 + sL) * r0)
    ok = (np.arange(T) >= t0) & np.isfinite(pexc)
    epi_day = first_day(ok & (pexc > 0.95), days)
    n, ly, gr_day = 14, np.log(y + 0.5), None
    for t in range(max(t0, n - 1), T):
        xs, ys = np.arange(n, dtype=float), ly[t - n + 1:t + 1]
        b, a0 = np.polyfit(xs, ys, 1)
        res = ys - (a0 + b * xs)
        se = np.sqrt(res @ res / (n - 2) / ((xs - xs.mean()) ** 2).sum())
        if b - 1.6448536 * se > g0:
            gr_day = int(days[t]); break
    return epi_day, gr_day


def main():
    OUT.mkdir(exist_ok=True)
    y, w_si = load()
    S, T = len(w_si), len(y)
    lags = np.arange(1, S + 1)
    m_si = float((lags * w_si).sum())
    cv_si = float(np.sqrt((lags ** 2 * w_si).sum() - m_si ** 2) / m_si)
    L = lag_matrix(y, S)
    lam_si = L @ w_si
    days = np.arange(1, T + 1)
    t0_auto = int(np.argmax(lam_si >= 0.5))

    def fam(lo, hi, n=41):
        return np.column_stack([disc_gamma(m, cv_si, S) for m in np.linspace(lo, hi, n)])

    # classes that cover the SI also include the SI itself; the long-side class deliberately does not
    classes = {
        "known SI": w_si[:, None],
        "Gamma +/-1 d": np.column_stack([fam(m_si - 1, m_si + 1), w_si]),
        "Gamma +/-2 d": np.column_stack([fam(m_si - 2, m_si + 2), w_si]),
        "short side [m-3, m]": np.column_stack([fam(m_si - 3, m_si), w_si]),
        "KL eps=0.1": 0.1,
        "KL eps=0.3": 0.3,
        "long side [m, m+3] (invalid if GI shorter)": fam(m_si, m_si + 3),
    }
    # (label, 0-based start index, days whose counts are not scored)
    skip_amoy = set(range(38, 41))  # days 38-40: Amoy Gardens common-source spike
    starts = [(f"start day {t0_auto + 1} (auto)", t0_auto, set()),
              ("start day 25 (after hospital cluster)", 24, set()),
              ("start day 25, days 38-40 not scored", 24, skip_amoy)]
    rows, traj = [], {}
    for start_label, t0, skip in starts:
        yy, LL, dd = y[t0:], L[t0:], days[t0:]
        # skipped days still enter the infection history (L) but are not bet on:
        # their counts and their exposure are both left out of the e-process
        keep = np.array([d not in skip for d in dd], dtype=float)
        N = np.cumsum(yy * keep)
        Zcum = np.cumsum(LL * keep[:, None], axis=0)
        Bs = {name: ((Zcum @ par).max(axis=1) if isinstance(par, np.ndarray) else kl_sup_path(Zcum, w_si, par))
              for name, par in classes.items()}
        for r0 in R0_LIST:
            for cname, B in Bs.items():
                for kname, kap in KAPPAS:
                    le = log_mixture(N, B, r0, kap)
                    traj[(start_label, r0, cname, kname)] = (dd, le)
                    row = dict(start=start_label, r0=r0, gi_class=cname, dispersion=kname,
                               t0_day=int(days[t0]), max_log_evidence=float(le.max()))
                    for a in ALPHAS:
                        row[f"alarm_day_{a:g}"] = first_day(le >= np.log(1 / a), dd)
                    rows.append(row)
            epi_day, gr_day = conventional_days(y, lam_si, w_si, lags, r0, t0, days)
            for nm, dday in [("EpiEstim-style P(R_t>r0)>0.95", epi_day), ("growth-rate 95% lower bound", gr_day)]:
                rows.append(dict(start=start_label, r0=r0, gi_class="known SI", dispersion=nm,
                                 t0_day=int(days[t0]), max_log_evidence=np.nan,
                                 **{f"alarm_day_{a:g}": (dday if a == 0.05 else None) for a in ALPHAS}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sars_alarm_days.csv", index=False)

    # figure: incidence/exposure, then log evidence for r0 = 1.5 from both start days
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4, 1.4]))
    axes[0].bar(days, y, color="grey", label="onsets")
    axes[0].plot(days, lam_si, "k-", lw=1, label=r"exposure $\Lambda_t$ (SI)")
    axes[0].axvline(t0_auto + 1, ls=":", c="k"); axes[0].axvline(25, ls=":", c="k")
    axes[0].set_ylabel("per day"); axes[0].legend(fontsize=8)
    axes[0].set_title("SARS 2003, Hong Kong", fontsize=10)
    r0 = 1.5
    for ax, (start_label, _, _) in zip(axes[1:], starts[:2]):
        for cname, ls in [("known SI", "-"), ("Gamma +/-2 d", "--"), ("short side [m-3, m]", ":")]:
            for kname, col in [("Poisson", "C0"), ("NB kappa-=0.5", "C1"), ("NB kappa-=0.16", "C3")]:
                dd, le = traj[(start_label, r0, cname, kname)]
                ax.plot(dd, le, ls, color=col, lw=1.2, label=f"{cname}, {kname}")
        ax.axhline(np.log(20), c="grey", lw=.8); ax.axhline(np.log(1000), c="grey", lw=.8, ls="--")
        ax.set_ylim(-20, 40)
        ax.set_ylabel(r"log e-process, $H_0$: $R_t\leq1.5$")
        ax.set_title(start_label, fontsize=9)
    axes[1].legend(fontsize=6.5, ncol=3)
    axes[2].set_xlabel("day since first case")
    fig.tight_layout()
    fig.savefig(OUT / "sars_application.pdf"); fig.savefig(OUT / "sars_application.png", dpi=150)

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    print(f"SI mean {m_si:.2f} d, CV {cv_si:.3f}; automatic monitoring start: day {t0_auto + 1}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
