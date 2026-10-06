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

import numpy as np

from pysmd.open_shell import jacobian

def _acceleration(md, residual):
    md.X_accel = -jacobian.pseudo_inverse_action(md, residual)


def _move_atoms(md, states, coefficient):
    """Update atomic coordinates shared by each state. We then need to update the basis for each state.
    Particularly when using MOM or IMOM, it is important to move the MOs from the previous state to determine
    which orbitals should now be occupied."""
    coords = md.atom_coord + coefficient * md._md_timestep * md.atom_veloc
    for state in states:
        if state.occupation_method == "mom":
            state.capture_occupation_reference()
        state.atom_coord = state.update_atomic_coords(coords)
        state.update_overlap_matrices()
        state.transport_occupation_reference()
    md.atom_coord = states[0].atom_coord


def _electrons_and_gradient(md):
    md.evaluate_density(md.dynvar_X @ md.Sm1)
    residual = md.dm_ao @ md.S - md.dynvar_X
    updated = md.update_first_level_dm(residual) if md.first_level_dm_update else None
    gradient = md.update_gradients(md.fock, md.dm_ao, md.dm_prop, md.dm_lin)
    return (residual, updated), gradient


def _dissipation(md, stage):
    return np.einsum("k,ksij->sij", md.eom_params.diss_coeffs, md.X_history[stage])


def _store_history(md, stage):
    md.X_history[stage, 1:] = md.X_history[stage, :-1].copy()
    md.X_history[stage, 0] = md.dynvar_X


def _electrons_and_weighted_gradient(states, weights):
    results = [_electrons_and_gradient(state) for state in states]
    total_grad = sum(weight * gradient for weight, (_, gradient) in zip(weights, results))
    return [residuals for residuals, _ in results], total_grad


def weighted_verlet_timestep(md, states, weights, total_grad, residuals_dDS):
    """One velocity-Verlet step on a weighted sum of shadow potentials. (Generalization of the Ziegler corrected MD).
    """
    for state, residual in zip(states, residuals_dDS):
        _acceleration(state, residual)
    md.atom_veloc -= 0.5 * md._md_timestep * total_grad / md.atom_mass[:, None]
    _move_atoms(md, states, 1.)
    for state in states:
        state.dynvar_X = (2 * state.X_history[0, 0] - state.X_history[0, 1]
                          + state.eom_params.diss_kappa * state.X_accel
                          + state.eom_params.diss_alpha * _dissipation(state, 0))
    residuals, total_grad = _electrons_and_weighted_gradient(states, weights)
    md.atom_veloc -= 0.5 * md._md_timestep * total_grad / md.atom_mass[:, None]
    for state in states:
        _store_history(state, 0)
    return residuals, total_grad


def weighted_coeff_split_timestep(md, states, weights, total_grad, residuals_dDS):
    """One coefficient-split symplectic step on a weighted sum of shadow potentials."""
    for stage, (a, b) in enumerate(zip(md.eom_params.coord_coeffs, md.eom_params.veloc_coeffs)):
        for state, residual in zip(states, residuals_dDS):
            _acceleration(state, residual)
        md.atom_veloc -= b * md._md_timestep * total_grad / md.atom_mass[:, None]
        _move_atoms(md, states, a)
        for state in states:
            state.X_veloc += b / md._md_timestep * (
                state.eom_params.diss_kappa * state.X_accel
                + state.eom_params.diss_alpha * _dissipation(state, stage))
            state.dynvar_X += a * md._md_timestep * state.X_veloc
        residuals, total_grad = _electrons_and_weighted_gradient(states, weights)
        residuals_dDS = [residual for residual, _ in residuals]
        for state in states:
            _store_history(state, stage)
    return residuals, total_grad


def integrate_verlet_timestep(md, tstep, total_grad, residual_dDS):
    """One complete velocity-Verlet step"""
    residuals, total_grad = weighted_verlet_timestep(md, (md,), (1.,), total_grad, (residual_dDS,))
    return residuals[0], total_grad


def integrate_coeff_split_timestep(md, tstep, total_grad, residual_dDS):
    """One full coefficient-split step"""
    residuals, total_grad = weighted_coeff_split_timestep(md, (md,), (1.,), total_grad, (residual_dDS,))
    return residuals[0], total_grad
