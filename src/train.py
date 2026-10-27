"""
Train the physics-constrained surrogate and a plain-MLP baseline, then evaluate.

Experiments
-----------
1. Full-data training  -> accuracy, physics violations, speed-up over DEVSIM
2. Data-efficiency     -> both models trained on 100 / 200 / 400 / 800 / all
                          devices, to see where the physics constraint helps most

Outputs: models/*.pt, results/metrics.json and figures in results/

Usage:
    python src/train.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import PhysicsSurrogate, PlainMLP, ideality_factor, physics_violations  # noqa: E402
import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

DATA = "data/diode_iv_dataset.csv"
DV = 0.05
EPOCHS = 4000


def load_split(path=DATA, seed=0):
    df = pd.read_csv(path)
    X = torch.tensor(df.iloc[:, :4].values, dtype=torch.float32)
    Y = torch.tensor(df.iloc[:, 4:].values, dtype=torch.float32)
    bias = np.array([float(c.split("_")[1]) for c in df.columns[4:]])
    idx = np.random.default_rng(seed).permutation(len(df))
    n_tr, n_va = int(0.70 * len(df)), int(0.15 * len(df))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return X, Y, bias, tr, va, te


def build(kind, Ytr):
    if kind == "physics":
        return PhysicsSurrogate(n_bias=Ytr.shape[1], dV=DV,
                                y0_mean=Ytr[:, 0].mean().item(),
                                y0_std=Ytr[:, 0].std().item(),
                                step_init=(Ytr[:, 1:] - Ytr[:, :-1]).mean(0))
    return PlainMLP(n_bias=Ytr.shape[1], y_mean=Ytr.mean(0), y_std=Ytr.std(0))


def fit(kind, Xtr, Ytr, Xva, Yva, epochs=EPOCHS):
    model = build(kind, Ytr)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.mean((model(Xtr) - Ytr) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                v = torch.mean((model(Xva) - Yva) ** 2).item()
            if v < best:
                best = v
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


BIAS_GRID = np.round(np.arange(0.05, 0.80 + 1e-9, 0.05), 3)


def evaluate(model, X, Y):
    with torch.no_grad():
        P = model(X)
    err = (P - Y).numpy()
    nonmono, sub_ideal = physics_violations(P.numpy(), BIAS_GRID, DV)
    return {
        "rmse_decades": float(np.sqrt(np.mean(err ** 2))),
        "mean_abs_pct_error_J": float(np.mean(np.abs(10 ** err - 1)) * 100),
        "median_abs_pct_error_J": float(np.median(np.abs(10 ** err - 1)) * 100),
        "p95_abs_pct_error_J": float(np.percentile(np.abs(10 ** err - 1), 95) * 100),
        "curves_non_monotonic": int(nonmono.sum()),
        "curves_ideality_below_1_above_0.15V": int(sub_ideal.sum()),
        "curves_violating_physics_pct": float(100 * np.mean(nonmono | sub_ideal)),
        "n_test_curves": int(len(Y)),
    }, P.numpy()


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    X, Y, bias, tr, va, te = load_split()
    print(f"Dataset: {len(X)} devices  train {len(tr)} / val {len(va)} / test {len(te)}")

    # ---------- Experiment 1: full data ----------
    metrics, preds, models = {}, {}, {}
    for kind in ("physics", "plain"):
        t0 = time.time()
        m = fit(kind, X[tr], Y[tr], X[va], Y[va])
        print(f"  trained {kind:8s} in {time.time() - t0:.0f} s")
        metrics[kind], preds[kind] = evaluate(m, X[te], Y[te])
        models[kind] = m
        torch.save(m.state_dict(), f"models/{kind}_surrogate.pt")

    # Speed: surrogate inference vs DEVSIM
    with torch.no_grad():
        Xb = X[te].repeat(10, 1)
        t0 = time.time()
        for _ in range(20):
            models["physics"](Xb)
        t_nn = (time.time() - t0) / (20 * len(Xb))
    t_devsim = 1.7   # s per device, single core (see results/dataset_generation_log.txt)
    metrics["speed"] = {"devsim_s_per_device": t_devsim,
                        "surrogate_s_per_device": t_nn,
                        "speedup_x": t_devsim / t_nn}

    # ---------- Experiment 2: data efficiency ----------
    sizes = [100, 200, 400, 800, len(tr)]
    eff = {"train_size": sizes, "physics": [], "plain": [],
           "physics_violation_pct": [], "plain_violation_pct": []}
    for n in sizes:
        sub = tr[:n]
        for kind in ("physics", "plain"):
            m = fit(kind, X[sub], Y[sub], X[va], Y[va], epochs=3000)
            r, _ = evaluate(m, X[te], Y[te])
            eff[kind].append(r["rmse_decades"])
            eff[f"{kind}_violation_pct"].append(r["curves_violating_physics_pct"])
        print(f"  n={n:5d}  RMSE physics {eff['physics'][-1]:.4f}  plain {eff['plain'][-1]:.4f} dec"
              f"   violations physics {eff['physics_violation_pct'][-1]:.1f}%  plain {eff['plain_violation_pct'][-1]:.1f}%")
    metrics["data_efficiency"] = eff

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save("results/test_index.npy", te)

    make_figures(X, Y, bias, te, preds, eff)
    print(json.dumps({k: metrics[k] for k in ("physics", "plain", "speed")}, indent=2))


def make_figures(X, Y, bias, te, preds, eff):
    Yte = Y[te].numpy()
    Xte = X[te].numpy()

    # Pick 4 visibly different test devices (spread in current level and lifetime)
    order = np.argsort(Yte[:, 7])
    picks = order[[5, len(order) // 3, 2 * len(order) // 3, -5]]

    # 1) I-V curves
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    for ax, i in zip(axes, picks):
        ax.semilogy(bias, 10 ** Yte[i], "o", color=ps.TRUTH, ms=5, label="DEVSIM (truth)")
        ax.semilogy(bias, 10 ** preds["physics"][i], color=ps.BLUE, label="Physics surrogate")
        ax.semilogy(bias, 10 ** preds["plain"][i], "--", color=ps.ORANGE, label="Plain MLP")
        na, nd, tau, L = Xte[i]
        ax.set_title(f"NA=1e{na:.1f}, ND=1e{nd:.1f}\nτ=1e{tau:.1f} s, L={L:.1f} µm", fontsize=9)
        ax.set_xlabel("Forward bias (V)")
    axes[0].set_ylabel("Current density (A/cm²)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Unseen test devices: surrogate vs DEVSIM", y=1.04)
    fig.savefig("results/iv_predictions.png")
    plt.close(fig)

    # 2) Ideality factor n(V): where the physics shows up
    vmid = (bias[1:] + bias[:-1]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, i in zip(axes, picks):
        ax.plot(vmid, ideality_factor(Yte[i]), "o", color=ps.TRUTH, ms=5, label="DEVSIM")
        ax.plot(vmid, ideality_factor(preds["physics"][i]), color=ps.BLUE, label="Physics surrogate")
        ax.plot(vmid, ideality_factor(preds["plain"][i]), "--", color=ps.ORANGE, label="Plain MLP")
        ax.axhline(1, color=ps.INK_2, lw=1, ls=":")
        ax.set_xlabel("Forward bias (V)")
        ax.set_ylim(0.5, 3.0)
    axes[0].set_ylabel("Ideality factor n")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Ideality factor: SRH recombination (n→2) at low bias, diffusion (n≈1), high injection at high bias", y=1.04)
    fig.savefig("results/ideality_factor.png")
    plt.close(fig)

    # 3) Parity plot
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot(Yte.ravel(), preds["physics"].ravel(), ".", color=ps.BLUE, ms=2, alpha=0.5)
    lo, hi = Yte.min() - 0.5, Yte.max() + 0.5
    ax.plot([lo, hi], [lo, hi], color=ps.INK_2, lw=1)
    ax.set_xlabel("DEVSIM log₁₀ J")
    ax.set_ylabel("Surrogate log₁₀ J")
    ax.set_title("Physics surrogate, all test points")
    fig.savefig("results/parity.png")
    plt.close(fig)

    # 4) Data efficiency (two separate panels, never a dual axis)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    n = eff["train_size"]
    a1.plot(n, eff["physics"], "o-", color=ps.BLUE, ms=6, label="Physics surrogate")
    a1.plot(n, eff["plain"], "s--", color=ps.ORANGE, ms=6, label="Plain MLP")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("Training devices"); a1.set_ylabel("Test RMSE (decades of J)")
    a1.set_title("Accuracy vs training-set size")
    a1.legend()
    a2.plot(n, eff["physics_violation_pct"], "o-", color=ps.BLUE, ms=6, label="Physics surrogate")
    a2.plot(n, eff["plain_violation_pct"], "s--", color=ps.ORANGE, ms=6, label="Plain MLP")
    a2.set_xscale("log")
    a2.set_xlabel("Training devices"); a2.set_ylabel("Test curves violating physics (%)")
    a2.set_title("Non-monotonic or n < 1 predictions")
    a2.legend()
    fig.savefig("results/data_efficiency.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
