"""
Neural-network surrogates for the PN-diode I-V curve.

PhysicsSurrogate  - physics-constrained network (the main model)
PlainMLP          - ordinary MLP baseline with the same hidden layers

Physics built into PhysicsSurrogate
-----------------------------------
Instead of predicting the 16 log10(J) values directly, the network predicts

    log10 J(V1)                       (the first bias point), and
    Delta_k = log10 J(V_k+1) - log10 J(V_k)   for every later step.

Every increment is forced to be positive (softplus / sigmoid), and from
V_CLAMP = 0.15 V upward it is also bounded:

    0 < Delta_k / Delta_V < S_max ,   S_max = 1 / (ln(10) * kT/q) ≈ 16.8 dec/V

so, by construction, every predicted curve
  * increases monotonically with forward bias, and
  * has an ideality factor n >= 1 once V >> kT/q (no diode at 300 K can be
    steeper than ~60 mV/decade).
The n >= 1 bound is NOT applied below 0.15 V on purpose: there the "-1" in
J0(exp(V/Vt) - 1) makes log J legitimately steeper than 1/Vt.
A plain MLP has neither guarantee.
"""

import math
import torch
import torch.nn as nn

KT_Q = 0.025852               # thermal voltage at 300 K (V)
S_MAX = 1.0 / (math.log(10) * KT_Q)   # max slope, decades per volt (n = 1)

# Parameter bounds used for input normalisation (must match generate_dataset.py)
X_LOW = torch.tensor([15.0, 15.0, -8.0, 1.0])
X_HIGH = torch.tensor([18.0, 18.0, -5.0, 20.0])


def normalise_x(x):
    """Map physical inputs to [-1, 1]."""
    return 2 * (x - X_LOW) / (X_HIGH - X_LOW) - 1


def _mlp(n_in, n_out, hidden=(128, 128, 128)):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


V_CLAMP = 0.15   # V; n >= 1 bound applies to bias steps starting at or above this


class PhysicsSurrogate(nn.Module):
    def __init__(self, n_bias=16, dV=0.05, v_start=0.05, y0_mean=-8.0, y0_std=1.0,
                 step_init=None, hidden=(128, 128, 128)):
        super().__init__()
        self.net = _mlp(4, n_bias, hidden)
        self.dV = dV
        self.max_step = S_MAX * dV       # largest allowed rise per bias step (decades)
        # step k goes from V_k to V_k+1; bounded steps are those with V_k >= V_CLAMP
        v_k = v_start + dV * torch.arange(n_bias - 1)
        self.register_buffer("bounded", (v_k >= V_CLAMP - 1e-9).float())
        self.register_buffer("y0_mean", torch.tensor(float(y0_mean)))
        self.register_buffer("y0_std", torch.tensor(float(y0_std)))
        if step_init is not None:
            # Start the network at the average curve shape (easier, faster training)
            frac = torch.clamp(torch.as_tensor(step_init) / self.max_step, 0.02, 0.98)
            free = torch.log(torch.expm1(torch.as_tensor(step_init).clamp(min=1e-3)))
            init = torch.where(self.bounded > 0, torch.logit(frac), free)
            with torch.no_grad():
                self.net[-1].bias[1:] = init.float()

    def forward(self, x_phys):
        z = self.net(normalise_x(x_phys))
        y0 = self.y0_mean + self.y0_std * z[:, :1]                  # log10 J at first bias
        s = z[:, 1:]
        steps = (self.bounded * self.max_step * torch.sigmoid(s)      # 0 < step < n=1 limit
                 + (1 - self.bounded) * nn.functional.softplus(s))    # 0 < step (monotonic only)
        return torch.cat([y0, y0 + torch.cumsum(steps, dim=1)], dim=1)


class PlainMLP(nn.Module):
    def __init__(self, n_bias=16, y_mean=None, y_std=None, hidden=(128, 128, 128)):
        super().__init__()
        self.net = _mlp(4, n_bias, hidden)
        self.register_buffer("y_mean", torch.zeros(n_bias) if y_mean is None else y_mean)
        self.register_buffer("y_std", torch.ones(n_bias) if y_std is None else y_std)

    def forward(self, x_phys):
        return self.y_mean + self.y_std * self.net(normalise_x(x_phys))


def physics_violations(logJ, bias, dV=0.05, tol=1e-3):
    """Per-curve flags: non-monotonic anywhere, or ideality n < 1 for V >= V_CLAMP."""
    import numpy as np
    slope = np.diff(logJ, axis=1) / dV
    nonmono = (slope <= 0).any(axis=1)
    mask = bias[:-1] >= V_CLAMP - 1e-9
    sub_ideal = (slope[:, mask] > S_MAX * (1 + tol)).any(axis=1)
    return nonmono, sub_ideal


def ideality_factor(logJ, dV=0.05):
    """n(V) = (1 / (ln10 * kT/q)) / (d log10 J / dV), at the midpoints of the bias grid."""
    slope = (logJ[..., 1:] - logJ[..., :-1]) / dV
    return S_MAX / slope
