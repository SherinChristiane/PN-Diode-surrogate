"""
Sanity check: does the DEVSIM simulation obey textbook diode physics?

Compares DEVSIM J-V curves with the analytic model

    J = J0_diff * (exp(V/Vt) - 1)  +  J0_rec * (exp(V/2Vt) - 1)

    J0_diff = q ni^2 [ Dn/(Ln NA) coth(Wp/Ln) + Dp/(Lp ND) coth(Wn/Lp) ]
              (Shockley diffusion current, valid for long AND short diodes)
    J0_rec  = q ni W / (2 tau)
              (Sah-Noyce-Shockley recombination in the depletion region)

Agreement in the mid-bias range shows the TCAD set-up is correct; the
deviations at low bias (recombination details) and high bias (high-level
injection, ohmic drop) are expected and are exactly the effects the
surrogate has to learn beyond the simple formula.

Usage:
    python src/validate_physics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

q, kT_q, ni = 1.6e-19, 0.025852, 1.0e10
eps = 11.1 * 8.85e-14
mu_n, mu_p = 400.0, 200.0          # same constants as DEVSIM simple_physics


def analytic_J(V, NA, ND, tau, length_um):
    L = length_um * 1e-4
    Dn, Dp = mu_n * kT_q, mu_p * kT_q
    Ln, Lp = np.sqrt(Dn * tau), np.sqrt(Dp * tau)
    Vbi = kT_q * np.log(NA * ND / ni ** 2)
    W = np.sqrt(2 * eps * np.clip(Vbi - V, 0.05, None) / q * (1 / NA + 1 / ND))
    xp, xn = W * ND / (NA + ND), W * NA / (NA + ND)
    Wp, Wn = np.clip(L / 2 - xp, 1e-7, None), np.clip(L / 2 - xn, 1e-7, None)
    J0d = q * ni ** 2 * (Dn / (Ln * NA) / np.tanh(Wp / Ln) + Dp / (Lp * ND) / np.tanh(Wn / Lp))
    J0r = q * ni * W / (2 * tau)
    return J0d * (np.exp(V / kT_q) - 1) + J0r * (np.exp(V / (2 * kT_q)) - 1)


def main():
    devnull = os.open(os.devnull, os.O_WRONLY); saved = os.dup(1); os.dup2(devnull, 1)
    try:
        from simulate import simulate_diode, BIAS
        cases = [
            dict(NA=1e17, ND=1e16, tau=1e-6, length_um=4.0),    # short diode
            dict(NA=1e16, ND=1e18, tau=1e-8, length_um=15.0),   # long diode, short lifetime
            dict(NA=1e15, ND=1e15, tau=1e-7, length_um=10.0),   # lightly doped
        ]
        sims = [simulate_diode(**c) for c in cases]
    finally:
        os.dup2(saved, 1); os.close(devnull); os.close(saved)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, c, J in zip(axes, cases, sims):
        Vf = np.linspace(0.05, 0.8, 100)
        ax.semilogy(BIAS, J, "o", color=ps.TRUTH, ms=5, label="DEVSIM drift-diffusion")
        ax.semilogy(Vf, analytic_J(Vf, **c), color=ps.AQUA, label="Analytic (Shockley + SNS)")
        ax.set_title(f"NA={c['NA']:.0e}, ND={c['ND']:.0e}\nτ={c['tau']:.0e} s, L={c['length_um']} µm", fontsize=9)
        ax.set_xlabel("Forward bias (V)")
        mid = (BIAS >= 0.4) & (BIAS <= 0.6)
        err = np.mean(np.abs(np.log10(J[mid]) - np.log10(analytic_J(BIAS[mid], **c))))
        print(f"{c}: mean |Δlog10 J| at 0.4-0.6 V = {err:.3f} decades")
    axes[0].set_ylabel("Current density (A/cm²)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("DEVSIM vs textbook diode theory", y=1.04)
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/physics_validation.png")
    print("Saved -> results/physics_validation.png")


if __name__ == "__main__":
    main()
