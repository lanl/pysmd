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

"""Recursive update of PySMD spatial-orbital occupations via Fermi operators.

PySMD stores closed-shell occupations in the range ``[0, 1]``. A fully
occupied spatial orbital therefore contributes one to the occupation sum and
two electrons to the system. Backend-specific spin-summed conventions are
handled by the electronic-structure interface, not by this module.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from itertools import count

from pysmd.common import constants
from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

if TYPE_CHECKING:
    from pysmd.closed_shell.newton_raphson import NewtonRaphson
    from pysmd.closed_shell.shadow_md import ShadowMD


def update_fractional_occ(
    method: NewtonRaphson | ShadowMD,
    mo_energy: Any = None,
) -> tuple[Any, float]:
    """Update fractional occupations via the Fermi-Dirac distribution.

    Note
    ----
    Occupations are returned using the PySMD spatial-orbital convention
    ``[0, 1]``. Consequently, the target occupation sum is the number of
    occupied spatial orbitals, ``n_elec / 2``.

    Parameters
    ----------
    method : ShadowMD
        Object that holds attributes such as:
        mo_energy, mo_coeff, n_elec, beta, and temp.
    mo_energy : numpy.ndarray
        Array of orbital energies. Optional, defaults to method.mo_energy.

    Returns
    -------
    mo_occ : numpy.ndarray
        Updated fractional spatial-orbital occupations in ``[0, 1]``.
    mu : float
        Updated chemical potential.
    """
    ### Default to object orbital energies, if not provided
    self: NewtonRaphson | ShadowMD = method
    if mo_energy is None:
        mo_energy = self.mo_energy

    ### Finite temperature
    # Initial chemical potential, midpoint between HOMO and LUMO energies.
    mu: float = 0.5 * (mo_energy[self.n_occ] + mo_energy[self.n_occ - 1])
    # Inverse of kB*T
    beta: float
    if self.beta is not None:
        beta = self.beta
    else:
        beta = self.beta = 1.0 / (constants.KB * method.temp)

    ### Track required number of iterations
    for f_iter in count(start=1):
        log.debug(f" > Iteration #{(f_iter):d}:")

        # Update fractional orbital occupation
        mo_occ = la.expit(-beta * (mo_energy - mu))
        mo_occ_tot: float = la.sum(mo_occ)
        log.debug(
            " -- spatial-orbital occupation update\n"
            f"   = [{' '.join(format(x, '.2f') for x in mo_occ.tolist())}]"
        )

        # Update error
        mo_occ_err: float = la.abs(mo_occ_tot - self.n_occ)
        log.debug(f" -- orbital occupation error\n   = {mo_occ_err:10e}")

        # Check convergence
        if (mo_occ_err <= self.frac_occ_tol):
            log.debug(" > DONE!\n")
            break

        # Compute change in occupancy w.r.t to chemical potential
        d_occ: float = la.sum(beta * mo_occ * (1.0 - mo_occ))

        # Avoid dividing by zero
        if d_occ == 0:
            log.debug((" > delta_occ/delta_mu is zero...\n" +
                       " > Avoiding division by zero.\n"))
            break

        # Update chemical potential
        mu += ((self.n_occ - mo_occ_tot) / d_occ)
        log.debug(f" -- chemical potential update\n   = {mu:8f}\n")

        # Prevent infinite loop, break if not converging
        if f_iter >= 100: # don"t hard code?
            log.error("Unable to converge fractional occupation numbers " +
                     f"after: {f_iter} iterations.")
            break

    return mo_occ, mu
