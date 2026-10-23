"""Run the full theorem-driven simulation study (Experiments 1-6) and the SARS application.

Put this file in the same folder as sim_common.py, exp1...exp6 and sars_application.py, then:
    python run_all.py                    # full study, about 5-6 minutes on a laptop
    python run_all.py --quick            # fast check of every experiment, about 1-2 minutes
    python run_all.py --epi-data PATH    # folder with SARS2003_incidence.csv and SARS2003_si_distr.csv
                                         # (default: an 'epi_data' folder next to this file)
Results (CSV + PDF/PNG figures) go to the 'results' subfolder, together with versions.txt,
which records the Python and package versions used.
"""
import platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "exp1_predictable_null.py",
    "exp2_dispersion_rescue.py",
    "exp3_far_growth.py",
    "exp4_gi_robustness.py",
    "exp5_competitors.py",
    "exp6_false_alarms.py",
]

args = sys.argv[1:]
epi = HERE / "epi_data"
if "--epi-data" in args:
    i = args.index("--epi-data")
    epi = Path(args[i + 1]).expanduser().resolve()
    del args[i:i + 2]

missing = [s for s in ["sim_common.py"] + SCRIPTS if not (HERE / s).exists()]
if missing:
    sys.exit("Missing files in " + str(HERE) + ":\n  " + "\n  ".join(missing))

out = HERE / "results"
out.mkdir(exist_ok=True)
lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"python: {sys.version.split()[0]}",
         f"platform: {platform.platform()}",
         f"mode: {'quick' if '--quick' in args else 'full'}"]
for mod in ["numpy", "scipy", "pandas", "matplotlib"]:
    try:
        lines.append(f"{mod}: {__import__(mod).__version__}")
    except Exception as e:  # pragma: no cover
        lines.append(f"{mod}: not available ({e})")
(out / "versions.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

t0 = time.time()
for s in SCRIPTS:
    print(f"\n===== {s} =====", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, str(HERE / s)] + args, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{s} failed (exit code {r.returncode})")
    print(f"----- {s} finished in {time.time() - t:.0f}s", flush=True)

sars = HERE / "sars_application.py"
if sars.exists() and (epi / "SARS2003_incidence.csv").exists():
    print("\n===== sars_application.py =====", flush=True)
    r = subprocess.run([sys.executable, str(sars), str(epi)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"sars_application.py failed (exit code {r.returncode})")
else:
    print(f"\n(SARS application skipped: run 'Rscript export_sars_data.R' first, or provide --epi-data PATH containing the two SARS2003 CSV files; expected folder: {epi})")

print(f"\nAll done in {time.time() - t0:.0f}s. Results in {out}")
