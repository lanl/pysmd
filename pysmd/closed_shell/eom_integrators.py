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

"""Provides methods to perform equation-of-motion integration."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from pysmd.closed_shell import fermi_dirac
from pysmd.closed_shell import jacobian
from pysmd.common import constants
from pysmd.common import logger
log: logger.logging.Logger = logger.getLogger(__name__)

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

if TYPE_CHECKING:
    from pysmd.closed_shell.shadow_md import ShadowMD

### Default attributes
# Finite temperature
DEFAULT_TEMP = 300.0
FRAC_OCC_ERROR_TOL = 1e-10
# EOM integration scheme and dissipation
DEFAULT_EOM_SCHEME = "verlet"
DEFAULT_EOM_L_MAX = 2
DEFAULT_DISS_K_MAX = 6
# Simulation time (atomic units, converted from femtoseconds)
DEFAULT_TSTEP_SIZE = (1.0 / constants.AU2FS) # 1 fs -> au
DEFAULT_NUM_TSTEPS = 100
DEFAULT_TOTAL_TIME = DEFAULT_TSTEP_SIZE * DEFAULT_NUM_TSTEPS
# Low-rank kernel
RELATIVE_RESIDUAL_ERROR_TOL = 1e-4
LOW_RANK_NMO_SCALAR = 0.8
PT_RECURSION_DEPTH = 8


def integrate_coeff_split_timestep(
    md: ShadowMD,
    tstep: int,
    total_grad,
    residual_dDS: Any,
) -> tuple[tuple[Any, Any | None], Any, Any]:
    """Integrate one timestep using coefficient-split substeps.

    Args:
        md: ShadowMD object.
        tstep: current timestep
        total_grad: Matrix of gradients/forces.
        residual_dDS: Matrix of dynamical variable residual.

    Returns:
        Residual matrices, the updated gradient, and the timestep X history.
    """
    ### Sub-integration steps
    for lstep in range(md.eom_l_max):

        ### Alias EOM coordinate and velocity coefficent
        a = md.eom_params.coord_coeffs[lstep]
        b = md.eom_params.veloc_coeffs[lstep]

        md.atom_veloc = md.atom_veloc - (b * md._md_timestep) * (total_grad / md.atom_mass.reshape(-1, 1))
        updated_coords = md.atom_coord + (a * md._md_timestep * md.atom_veloc)
        md.atom_coord = md.update_atomic_coords(new_coords=updated_coords)

        ### Update overlap matrices
        md.update_overlap_matrices()

        ### Electronic degrees of freedom
        kern_act = None
        md.X_accel = la.zeros_like(md.dynvar_X)
        if tstep == 0 and lstep == 0:
            log.info("\n>> First timestep! Acceleration matrix set to all zeros.")
        else:
            log.info("\n>> Updating dynamical variable, X...")
            kern_act = jacobian.pseudo_inverse_action(md, residual_dDS)
            md.X_accel -= kern_act

        ### Dissipative friction-like term
        diss_term = sum(md.eom_params.diss_coeffs[i]
                        * md.X_history[(lstep * md.n_mo):(lstep * md.n_mo + md.n_mo),
                                           (i * md.n_mo):(i * md.n_mo + md.n_mo)
                        ] for i in range(md.diss_k_max + 1)
                    )

        ### Update dynamical variable
        md.X_veloc = md.X_veloc + ((md.eom_params.diss_kappa * b) / md._md_timestep) * md.X_accel \
                      + ((md.eom_params.diss_alpha * b) / md._md_timestep) * diss_term
        md.dynvar_X = md.dynvar_X + (a * md._md_timestep * md.X_veloc)

        ### Propagated density matrix, P = X*S^(-1)
        md.dm_prop = la.matmul(md.dynvar_X, md.Sm1)

        ### Update Fock matrix
        md.h1e = md.interface.compute_core_hamiltonian_matrix()
        md.vhf = md.interface.compute_eff_potential_matrix(dm=md.dm_prop)
        md.fock = md.h1e + md.vhf
        md.fock_orth = la.matmul(
            la.matmul(md.Sm12.T.conj(), md.fock), md.Sm12
        )
        md.mo_energy, md.mo_coeff = la.linalg.eigh(md.fock_orth)

        ### Fermi-Dirac operator expansion (finite temperature)
        if md.temp > 0:
            log.info("\n>> Computing fractional occupation...")
            md.mo_occ, md.mu = fermi_dirac.update_fractional_occ(md)

            log.info(
                " -- Updated: Orbital occupations\n"
                f"   = [{' '.join(format(x, '.2f') for x in md.mo_occ.tolist())}]"
            )
            log.info(f" -- Updated: Chemical potential\n   = {md.mu:8f}")

        ### Update density matrices
        md.dm_orth = la.matmul(
            la.matmul(md.mo_coeff, la.diag(md.mo_occ)), md.mo_coeff.T.conj()
        )
        md.dm_ao = la.matmul(
            la.matmul(md.Sm12, md.dm_orth), md.Sm12.T.conj()
        )
        md.dm_lin = (2.0 * md.dm_ao) - md.dm_prop

        ### Residual of dynamical variable
        residual_dDS = la.matmul(md.dm_ao, md.S) - md.dynvar_X

        ### First-level update
        updated_dDS = None
        if md.first_level_dm_update:
            updated_dDS = md.update_first_level_dm(res_mat=residual_dDS)

        # ### Compute new gradients
        total_grad = md.update_gradients(
            fock=md.fock,
            dm_ao=md.dm_ao,
            dm_prop=md.dm_prop,
            dm_lin=md.dm_lin,
        )

        ### Sub-integration history matrices
        if lstep == 0:
            tstep_X_history = la.copy(md.dynvar_X)
        else:
            tstep_X_history = la.vstack((tstep_X_history, md.dynvar_X))

    ### Return residuals as a tuple
    residuals = (residual_dDS, updated_dDS)

    return residuals, total_grad, tstep_X_history


def integrate_verlet_timestep(
    md: ShadowMD,
    tstep: int,
    total_grad,
    residual_dDS: Any | None,
) -> tuple[tuple[Any, Any | None], Any]:
    """Integrate one timestep using leapfrog Verlet.

    Equation of motion equations are integrated following a modified leapfrog Verlet scheme:
    P(t+dt) = 2*P(t) - P(t-dt) + kappa*P''(t) + [weak dissipation term]
    P''(t) = P''(t)/w^2 ; kappa = dt^2*w^2 (~=2)
    """
    ### Update velocity and coordinates
    md.atom_veloc = md.atom_veloc - (0.5 * md._md_timestep) * (total_grad / md.atom_mass.reshape(-1, 1))
    updated_coords = md.atom_coord + (md._md_timestep * md.atom_veloc)
    md.atom_coord = md.update_atomic_coords(new_coords=updated_coords)

    ### Update overlap matrices
    md.update_overlap_matrices()

    ### Electronic degrees of freedom
    kern_act = None
    md.X_accel = la.zeros_like(md.dynvar_X)
    if residual_dDS is None and tstep == 0:
        log.info("\n>> First timestep! Acceleration matrix set to all zeros.")
    else:
        log.info("\n>> Updating dynamical variable, X...")
        kern_act = jacobian.pseudo_inverse_action(md, residual_dDS)
        md.X_accel -= kern_act

    ### Dissipative friction-like term
    diss_term = sum(md.eom_params.diss_coeffs[i]
                    * md.X_history[:, (i * md.n_mo):(i * md.n_mo + md.n_mo)
                    ] for i in range(md.diss_k_max + 1)
                )

    ### Update dynamical variable
    md_X_zero = la.copy(md.X_history[:, :md.n_mo])
    md_X_one = la.copy(md.X_history[:, md.n_mo:md.n_mo + md.n_mo])
    md.dynvar_X = (2.0 * md_X_zero) - md_X_one + (md.eom_params.diss_kappa * md.X_accel) + (md.eom_params.diss_alpha * diss_term)
    del md_X_zero, md_X_one

    ### Propagated density matrix, P = X*S^(-1)
    md.dm_prop = la.matmul(md.dynvar_X, md.Sm1)

    ### Update Fock matrix
    md.h1e = md.interface.compute_core_hamiltonian_matrix()
    md.vhf = md.interface.compute_eff_potential_matrix(dm=md.dm_prop)
    md.fock = md.h1e + md.vhf
    md.fock_orth = la.matmul(
        la.matmul(md.Sm12.T.conj(), md.fock), md.Sm12
    )
    md.mo_energy, md.mo_coeff = la.linalg.eigh(md.fock_orth)

    ### Fermi-Dirac operator expansion (finite temperature)
    if md.temp > 0:
        log.info("\n>> Computing fractional occupation...")
        md.mo_occ, md.mu = fermi_dirac.update_fractional_occ(md)

        log.info(
            " -- Updated: Orbital occupations\n"
            f"   = [{' '.join(format(x, '.2f') for x in md.mo_occ.tolist())}]"
        )
        log.info(f" -- Updated: Chemical potential\n   = {md.mu:8f}")

    ### Update density matrices
    md.dm_orth = la.matmul(
        la.matmul(md.mo_coeff, la.diag(md.mo_occ)), md.mo_coeff.T.conj()
    )
    md.dm_ao = la.matmul(
        la.matmul(md.Sm12, md.dm_orth), md.Sm12.T.conj()
    )
    md.dm_lin = (2.0 * md.dm_ao) - md.dm_prop

    ### Residual of dynamical variable
    residual_dDS = la.matmul(md.dm_ao, md.S) - md.dynvar_X

    ### First-level update
    updated_dDS = None
    if md.first_level_dm_update:
        updated_dDS = md.update_first_level_dm(res_mat=residual_dDS)

    ### Compute new gradients
    total_grad = md.update_gradients(
        fock=md.fock,
        dm_ao=md.dm_ao,
        dm_prop=md.dm_prop,
        dm_lin=md.dm_lin,
    )

    ### Final velocity update
    md.atom_veloc = md.atom_veloc - (0.5 * md._md_timestep) * (total_grad / md.atom_mass.reshape(-1, 1))

    ### Return residuals as a tuple
    residuals = (residual_dDS, updated_dDS)

    return residuals, total_grad
