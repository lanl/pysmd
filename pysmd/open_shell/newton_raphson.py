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

"""Coupled-spin Newton SCF for unrestricted references."""

import numpy as np

from pysmd.common import logger
log: logger.logging.Logger = logger.getLogger(__name__)

from pysmd.open_shell import jacobian
from pysmd.open_shell.occupations import InitialMaximumOverlap
from pysmd.open_shell.shadow_md import ShadowMD

### Default attributes
MAX_CYCLES = 100
ENERGY_DIFF_ERROR_TOL = 1e-10
RESIDUAL_NORM_ERROR_TOL = 1e-9
RELATIVE_RESIDUAL_ERROR_TOL = 1e-6


class NewtonRaphson(ShadowMD):
    """Solve the alpha/beta density fixed point with a damped Newton step.

    The Krylov space contains both spin channels and therefore retains the
    Coulomb and XC coupling. A residual line search stabilizes initial guesses.
    Converged PySCF references can be supplied as starting guesses as well.
    Set occupation_method='imom' for zero-temperature DeltaSCF, using kernel's
    dm0 (or the interface's initial density) to define the occupied subspaces.

    Recognized scf_params: occupation_method, max_cycle, energy_diff_error_tol,
    res_norm_error_tol, rel_res_error_tol, max_rank, temp, frac_occ_tol, and
    initial_dm_ao (the default dm0). n_atom, n_elec, n_mo, and n_occ may be
    given but must agree with the interface.
    """

    def __init__(self, qm_interface, scf_params=None):
        super().__init__(qm_interface)
        params = dict(scf_params or {})
        self.occupation_method = params.pop("occupation_method", "aufbau")
        if self.occupation_method not in ("aufbau", "imom"):
            raise ValueError("occupation_method must be 'aufbau' or 'imom'.")
        self.occupation_reference = None
        if self.occupation_method == "imom":
            self.temp = 0.
        self.max_cycle = params.pop("max_cycle", MAX_CYCLES)
        self.energy_diff_error_tol = params.pop("energy_diff_error_tol", ENERGY_DIFF_ERROR_TOL)
        self.res_norm_error_tol = params.pop("res_norm_error_tol", RESIDUAL_NORM_ERROR_TOL)
        self.rel_res_error_tol = params.pop("rel_res_error_tol", RELATIVE_RESIDUAL_ERROR_TOL)
        self.max_rank = params.pop("max_rank", 2 * self.n_mo ** 2)
        self.initial_dm_ao = params.pop("initial_dm_ao", None)
        for name in ("temp", "frac_occ_tol"):
            if name in params:
                setattr(self, name, params.pop(name))
        for name in ("n_atom", "n_elec", "n_mo", "n_occ"):
            if name in params and not np.array_equal(params.pop(name), getattr(self, name)):
                raise ValueError(f"{name} must agree with the unrestricted reference.")
        if params:
            raise TypeError(f"Unknown unrestricted SCF parameters: {', '.join(params)}")
        self.converged = False

    def kernel(self, dm0=None, *, occupation_reference=None):
        """Converge SCF from dm0, optionally retaining an MD state's reference.

        dm0 defaults to scf_params['initial_dm_ao'], then to the interface's
        initial density. An explicit reference is already represented in the
        current orthonormal AO basis and must remain fixed through every trial
        and correction. Returns the converged total (free) energy.
        """
        if self.max_cycle < 1:
            raise ValueError("max_cycle must be positive.")
        self.converged = self.interface.conv_scf = False
        if self.occupation_method == "imom" and self.temp != 0:
            raise ValueError("IMOM DeltaSCF requires temp=0 and integer occupations.")
        self.update_overlap_matrices()
        if dm0 is None:
            dm0 = self.initial_dm_ao
        if dm0 is None:
            dm0 = self.interface.compute_guess_density_matrix()
        if occupation_reference is not None:
            if (self.occupation_method != "imom"
                    or not isinstance(occupation_reference, InitialMaximumOverlap)
                    or occupation_reference.spin_counts != self.n_occ
                    or occupation_reference.projector.shape != (2, self.n_mo, self.n_mo)):
                raise ValueError("occupation_reference must match the IMOM calculation's basis and spin counts.")
        if self.occupation_method == "imom":
            self.occupation_reference = (occupation_reference if occupation_reference is not None
                                         else InitialMaximumOverlap(dm0, self.Sp12, self.n_occ))
        dm0 = np.asarray(dm0, dtype=float)
        if dm0.shape != (2, self.n_mo, self.n_mo):
            raise ValueError("Initial density must have shape (2, nmo, nmo).")
        self.dynvar_X = ((dm0 + dm0.swapaxes(-1, -2)) * 0.5) @ self.S
        previous_energy = None
        for cycle in range(1, self.max_cycle + 1):
            self.evaluate_density(self.dynvar_X @ self.Sm1)
            residual = self.dm_ao @ self.S - self.dynvar_X
            norm = np.linalg.norm(residual)
            self.update_energies(self.dm_ao, compute_e_nuc=True, compute_e_free=True)
            difference = 0. if previous_energy is None else abs(self.e_tot - previous_energy)
            log.info(f"Unrestricted SCF {cycle}: E = {self.e_tot:.12f} Ha; residual = {norm:.3e}")
            if norm <= self.res_norm_error_tol and difference <= self.energy_diff_error_tol:
                self.dynvar_X = self.dm_ao @ self.S
                self.evaluate_density(self.dm_ao)
                if np.linalg.norm(self.dm_ao @ self.S - self.dynvar_X) > self.res_norm_error_tol:
                    # An occupation switch during the final rebuild is not convergence.
                    previous_energy = self.e_tot
                    continue
                self.update_energies(self.dm_ao, compute_e_nuc=True, compute_e_free=True)
                self.converged = self.interface.conv_scf = True
                self.cycles = cycle
                return self.e_tot
            previous_energy = self.e_tot
            if norm < 1e-14:
                # Already stationary; one more cycle establishes the energy criterion.
                continue
            # The line search below verifies the step, so an inexact direction
            # near an SCF instability is acceptable here.
            direction = jacobian.pseudo_inverse_action(self, residual, allow_inexact=True)
            current = self.dynvar_X.copy()
            for attempt in range(14):
                candidate = current - (0.5 ** attempt) * direction
                self.evaluate_density(candidate @ self.Sm1)
                new_norm = np.linalg.norm(self.dm_ao @ self.S - candidate)
                if new_norm < norm:
                    self.dynvar_X = candidate
                    break
            else:
                # Far from a solution the residual norm can have a nonzero
                # local minimum and Newton directions become nearly singular.
                # A damped fixed-point step leaves that region; convergence
                # still requires both original criteria in a later cycle.
                self.dynvar_X = current + 0.5 * residual
                log.warning(f"Unrestricted SCF cycle {cycle}: using a damped density step after line-search failure.")
        raise RuntimeError(f"Unrestricted SCF did not converge in {self.max_cycle} cycles.")
