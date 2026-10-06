#
# Copyright (c) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare derivative works, distribute copies
# to the public, perform publicly and display publicly, and to permit others
# to do so.

"""Unrestricted SCF, shadow MD, and DeltaSCF excited states.

Spin-resolved counterparts of :mod:`pysmd.closed_shell` for UHF and UKS
references with fixed alpha/beta populations:

    - `shadow_md`: :class:`ShadowMD`, unrestricted shadow MD, with Aufbau,
                   MOM, or IMOM orbital occupations.
    - `newton_raphson`: :class:`NewtonRaphson`, coupled-spin Newton SCF,
                        including IMOM DeltaSCF from a non-Aufbau guess.
    - `spin_correction`: Ziegler two-determinant singlet estimates,
                         :class:`ZieglerDeltaSCF` and :class:`ZieglerShadowMD`.
    - `occupations`: Maximum-overlap occupation tracking.
    - `fermi_dirac`: Fixed-population Fermi occupations for both spins.
    - `jacobian`: Coupled-spin density response and inverse Jacobian actions.
    - `eom_integrators`, `eom_params`: Spin-resolved EOM integration.
    - `simulation_data`: Unrestricted and Ziegler simulation data.

The open-shell drivers currently require the NumPy linear-algebra backend.
"""

from pysmd.open_shell import (
    eom_integrators,
    eom_params,
    fermi_dirac,
    jacobian,
    newton_raphson,
    occupations,
    shadow_md,
    simulation_data,
    spin_correction,
)
from pysmd.open_shell.newton_raphson import NewtonRaphson
from pysmd.open_shell.shadow_md import ShadowMD
from pysmd.open_shell.spin_correction import (
    ZieglerDeltaSCF,
    ZieglerShadowMD,
    singlet_triplet_guesses,
    ziegler_estimate,
)


__all__ = [
    "eom_integrators",
    "eom_params",
    "fermi_dirac",
    "jacobian",
    "newton_raphson",
    "occupations",
    "shadow_md",
    "simulation_data",
    "spin_correction",
    "NewtonRaphson",
    "ShadowMD",
    "ZieglerDeltaSCF",
    "ZieglerShadowMD",
    "singlet_triplet_guesses",
    "ziegler_estimate",
]
