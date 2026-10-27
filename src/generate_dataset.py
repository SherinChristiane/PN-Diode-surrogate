"""
Generate a training dataset of PN-diode forward I-V curves with DEVSIM.

Design parameters are sampled with a Latin hypercube (better coverage than
random sampling for the same number of simulations):

    NA        1e15 - 1e18 cm^-3   (log-uniform)
    ND        1e15 - 1e18 cm^-3   (log-uniform)
    tau       1e-8 - 1e-5 s       (log-uniform, SRH lifetime)
    length    1 - 20 um           (uniform, junction at the centre)

Output: data/diode_iv_dataset.csv with one row per device:
    log10_NA, log10_ND, log10_tau, length_um, logJ_0.05, ..., logJ_0.80

Usage:
    python src/generate_dataset.py --n 1200 --workers 2
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(__file__))

BOUNDS_LOW = np.array([15.0, 15.0, -8.0, 1.0])
BOUNDS_HIGH = np.array([18.0, 18.0, -5.0, 20.0])
PARAM_NAMES = ["log10_NA", "log10_ND", "log10_tau", "length_um"]


def _run_one(x):
    """Worker: simulate one device. DEVSIM's solver log is sent to /dev/null."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        from simulate import simulate_diode
        J = simulate_diode(NA=10 ** x[0], ND=10 ** x[1], tau=10 ** x[2], length_um=x[3])
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)
    if J is None or np.any(~np.isfinite(J)) or np.any(J <= 0):
        return None
    return np.log10(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/diode_iv_dataset.csv")
    args = ap.parse_args()

    from simulate import BIAS

    sampler = qmc.LatinHypercube(d=4, seed=args.seed)
    X = qmc.scale(sampler.random(args.n), BOUNDS_LOW, BOUNDS_HIGH)

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(_run_one, list(X), chunksize=10)
    elapsed = time.time() - t0

    rows = [np.concatenate([x, y]) for x, y in zip(X, results) if y is not None]
    n_fail = sum(r is None for r in results)
    cols = PARAM_NAMES + [f"logJ_{v:.2f}" for v in BIAS]
    df = pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6f")

    print(f"Simulated {args.n} devices in {elapsed/60:.1f} min "
          f"({elapsed/args.n:.2f} s/device with {args.workers} workers)")
    print(f"Converged: {len(df)}   Failed: {n_fail}")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
