"""
1D silicon PN diode simulation with DEVSIM (open-source TCAD).

Solves the coupled Poisson + electron/hole continuity equations
(drift-diffusion with SRH recombination) for an abrupt PN junction
and returns the forward I-V curve.

Usage:
    python src/simulate.py            # runs one example device and prints J-V
"""

import os
import sys
import numpy as np

# DEVSIM needs a BLAS/LAPACK library. If it is not found automatically,
# point DEVSIM_MATH_LIBS at one (see README -> "Installation").
import devsim
from devsim.python_packages.simple_physics import (
    GetContactBiasName,
    SetSiliconParameters,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
)
from devsim.python_packages.model_create import CreateSolution, CreateNodeModel

# Forward bias points (V). V = 0 is skipped because J = 0 there (log undefined).
BIAS = np.round(np.arange(0.05, 0.80 + 1e-9, 0.05), 3)

DEVICE = "diode"
REGION = "Si"
T = 300.0  # K

# Keep DEVSIM's solver log quiet
devsim.set_parameter(name="debug_level", value="info")


def _build_mesh(length_um: float):
    """Abrupt junction at the centre, fine grid near the junction."""
    L = length_um * 1e-4  # um -> cm
    xj = L / 2
    mesh = "dio_mesh"
    devsim.create_1d_mesh(mesh=mesh)
    devsim.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=L / 200, tag="anode")
    devsim.add_1d_mesh_line(mesh=mesh, pos=xj, ps=2e-7)  # 2 nm at junction
    devsim.add_1d_mesh_line(mesh=mesh, pos=L, ps=L / 200, tag="cathode")
    devsim.add_1d_contact(mesh=mesh, name="anode", tag="anode", material="metal")
    devsim.add_1d_contact(mesh=mesh, name="cathode", tag="cathode", material="metal")
    devsim.add_1d_region(mesh=mesh, material="Si", region=REGION, tag1="anode", tag2="cathode")
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=DEVICE)
    return mesh, xj


def simulate_diode(NA: float, ND: float, tau: float, length_um: float,
                   bias=BIAS, verbose: bool = False):
    """
    Simulate a 1D PN diode.

    Parameters
    ----------
    NA : acceptor doping of the p-side (cm^-3), anode side
    ND : donor doping of the n-side (cm^-3), cathode side
    tau : SRH carrier lifetime (s), same for electrons and holes
    length_um : total device length (um), junction at the centre

    Returns
    -------
    J : np.ndarray of current density (A/cm^2) at each bias point,
        or None if the solver failed to converge.
    """
    mesh, xj = _build_mesh(length_um)
    try:
        SetSiliconParameters(DEVICE, REGION, T)
        for name in ("taun", "taup"):
            devsim.set_parameter(device=DEVICE, region=REGION, name=name, value=tau)

        # Doping profile
        CreateNodeModel(DEVICE, REGION, "Acceptors", f"{NA:.6e}*step({xj:.6e}-x)")
        CreateNodeModel(DEVICE, REGION, "Donors", f"{ND:.6e}*step(x-{xj:.6e})")
        CreateNodeModel(DEVICE, REGION, "NetDoping", "Donors-Acceptors")

        # 1) Equilibrium: Poisson only
        CreateSolution(DEVICE, REGION, "Potential")
        CreateSiliconPotentialOnly(DEVICE, REGION)
        for c in ("anode", "cathode"):
            devsim.set_parameter(device=DEVICE, name=GetContactBiasName(c), value=0.0)
            CreateSiliconPotentialOnlyContact(DEVICE, REGION, c)
        devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-10,
                     maximum_iterations=50)

        # 2) Full drift-diffusion, initialised from equilibrium carriers
        CreateSolution(DEVICE, REGION, "Electrons")
        CreateSolution(DEVICE, REGION, "Holes")
        devsim.set_node_values(device=DEVICE, region=REGION, name="Electrons",
                               init_from="IntrinsicElectrons")
        devsim.set_node_values(device=DEVICE, region=REGION, name="Holes",
                               init_from="IntrinsicHoles")
        CreateSiliconDriftDiffusion(DEVICE, REGION)
        for c in ("anode", "cathode"):
            CreateSiliconDriftDiffusionAtContact(DEVICE, REGION, c)
        devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10,
                     maximum_iterations=50)

        # 3) Forward bias sweep on the anode, in small steps for stability
        J = []
        v_now = 0.0
        for v_target in bias:
            steps = np.linspace(v_now, v_target, 6)[1:]
            for v in steps:
                devsim.set_parameter(device=DEVICE, name=GetContactBiasName("anode"), value=float(v))
                devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-9,
                             maximum_iterations=40)
            v_now = v_target
            je = devsim.get_contact_current(device=DEVICE, contact="anode",
                                            equation="ElectronContinuityEquation")
            jh = devsim.get_contact_current(device=DEVICE, contact="anode",
                                            equation="HoleContinuityEquation")
            J.append(abs(je + jh))
            if verbose:
                print(f"V = {v_target:.2f} V   J = {abs(je + jh):.3e} A/cm^2")
        return np.array(J)
    except devsim.error as e:
        if verbose:
            print("Solver did not converge:", e)
        return None
    finally:
        devsim.delete_device(device=DEVICE)
        devsim.delete_mesh(mesh=mesh)


if __name__ == "__main__":
    J = simulate_diode(NA=1e17, ND=1e16, tau=1e-6, length_um=4.0, verbose=True)
