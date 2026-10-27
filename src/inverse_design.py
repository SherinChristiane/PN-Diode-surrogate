"""
Inverse design: given a target J-V curve, find the device parameters
(NA, ND, tau, length) that produce it.

Because the surrogate is differentiable, we can run gradient descent on the
*inputs* instead of the weights. Each optimisation takes well under a second,
whereas a brute-force search with DEVSIM would need thousands of simulations.

Steps
  1. For each target curve (unseen test devices), start from 32 random guesses.
  2. Optimise the inputs with Adam to minimise the log-current mismatch.
     Inputs are kept inside the valid design range with a sigmoid mapping.
  3. Take the best guess and VERIFY it by running a real DEVSIM simulation.

Usage:
    python src/inverse_design.py --n_targets 20
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import PhysicsSurrogate, X_LOW, X_HIGH  # noqa: E402
import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

torch.manual_seed(1)


def load_model(Y_train_first_col):
    m = PhysicsSurrogate(n_bias=16, y0_mean=0.0, y0_std=1.0)
    m.load_state_dict(torch.load("models/physics_surrogate.pt"))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def invert(model, target, n_starts=32, steps=600):
    """Return the best physical parameters (NA, ND, tau, L) for one target curve."""
    u = torch.randn(n_starts, 4, requires_grad=True)          # unconstrained variables
    opt = torch.optim.Adam([u], lr=0.05)
    tgt = torch.tensor(target, dtype=torch.float32).expand(n_starts, -1)
    for _ in range(steps):
        x = X_LOW + (X_HIGH - X_LOW) * torch.sigmoid(u)        # always inside the design range
        loss = ((model(x) - tgt) ** 2).mean(dim=1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        x = X_LOW + (X_HIGH - X_LOW) * torch.sigmoid(u)
        loss = ((model(x) - tgt) ** 2).mean(dim=1)
        k = torch.argmin(loss)
    return x[k].numpy(), float(np.sqrt(loss[k].item()))


def devsim_check(x):
    devnull = os.open(os.devnull, os.O_WRONLY); saved = os.dup(1); os.dup2(devnull, 1)
    try:
        from simulate import simulate_diode
        J = simulate_diode(NA=10 ** x[0], ND=10 ** x[1], tau=10 ** x[2], length_um=float(x[3]))
    finally:
        os.dup2(saved, 1); os.close(devnull); os.close(saved)
    return None if J is None else np.log10(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_targets", type=int, default=20)
    args = ap.parse_args()

    df = pd.read_csv("data/diode_iv_dataset.csv")
    te = np.load("results/test_index.npy")[: args.n_targets]
    X = df.iloc[:, :4].values[te]
    Y = df.iloc[:, 4:].values[te]
    bias = np.array([float(c.split("_")[1]) for c in df.columns[4:]])
    model = load_model(None)

    rows = []
    for i, (x_true, y_true) in enumerate(zip(X, Y)):
        x_hat, rmse_sur = invert(model, y_true)
        y_dev = devsim_check(x_hat)
        rmse_dev = float(np.sqrt(np.mean((y_dev - y_true) ** 2))) if y_dev is not None else None
        rows.append(dict(
            target=i,
            true_log10_NA=x_true[0], found_log10_NA=x_hat[0],
            true_log10_ND=x_true[1], found_log10_ND=x_hat[1],
            true_log10_tau=x_true[2], found_log10_tau=x_hat[2],
            true_length_um=x_true[3], found_length_um=x_hat[3],
            surrogate_fit_rmse_dec=rmse_sur, devsim_verified_rmse_dec=rmse_dev,
            _y_true=y_true, _y_dev=y_dev))
        print(f"target {i:2d}: surrogate fit {rmse_sur:.3f} dec | DEVSIM-verified {rmse_dev:.3f} dec")

    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    res.to_csv("results/inverse_design_results.csv", index=False, float_format="%.4f")

    ok = res["devsim_verified_rmse_dec"].dropna()
    # The doping of the LIGHTLY doped side controls the current; the heavy side barely matters.
    light_err, heavy_err = [], []
    for _, r in res.iterrows():
        if r.true_log10_NA < r.true_log10_ND:
            light_err.append(abs(r.found_log10_NA - r.true_log10_NA))
            heavy_err.append(abs(r.found_log10_ND - r.true_log10_ND))
        else:
            light_err.append(abs(r.found_log10_ND - r.true_log10_ND))
            heavy_err.append(abs(r.found_log10_NA - r.true_log10_NA))
    summary = {
        "n_targets": int(len(res)),
        "devsim_verified_rmse_decades_median": float(ok.median()),
        "devsim_verified_mean_abs_pct_error_J_median": float((10 ** ok.median() - 1) * 100),
        "targets_matched_within_0.05_dec": int((ok < 0.05).sum()),
        "median_abs_error_log10_doping_lightly_doped_side": float(np.median(light_err)),
        "median_abs_error_log10_doping_heavily_doped_side": float(np.median(heavy_err)),
        "median_abs_error_log10_tau": float(np.median(abs(res.found_log10_tau - res.true_log10_tau))),
        "median_abs_error_length_um": float(np.median(abs(res.found_length_um - res.true_length_um))),
    }
    with open("results/inverse_design_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Figure: 3 examples, target vs DEVSIM re-simulation of the found design
    best3 = res.sort_values("devsim_verified_rmse_dec").index[[0, len(res) // 2, -1]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, k, lab in zip(axes, best3, ["Best match", "Median match", "Worst match"]):
        r = rows[k]
        ax.semilogy(bias, 10 ** r["_y_true"], "o", color=ps.TRUTH, ms=5, label="Target curve")
        if r["_y_dev"] is not None:
            ax.semilogy(bias, 10 ** r["_y_dev"], color=ps.BLUE, label="DEVSIM of found design")
        ax.set_title(f"{lab}: error {r['devsim_verified_rmse_dec']:.3f} decades", fontsize=9)
        ax.set_xlabel("Forward bias (V)")
    axes[0].set_ylabel("Current density (A/cm²)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Inverse design, verified by re-running DEVSIM", y=1.03)
    fig.savefig("results/inverse_design.png")
    print("Saved -> results/inverse_design.png")


if __name__ == "__main__":
    main()
