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

"""Coupled-spin density response and low-rank inverse Jacobian actions."""

import numpy as np

from pysmd.common import logger
log = logger.getLogger(__name__)


def density_response(method, potential_response, delta_x):
    """Apply dD/dX with each spin population held fixed.

    For Aufbau occupations this is the derivative of the Fermi matrix
    function, including the chemical-potential response for each spin.
    For IMOM/MOM only orbital rotations contribute and energy differences
    retain their signs when an occupied orbital lies above a virtual one.
    """
    delta_p = delta_x @ method.Sm1
    delta_f = potential_response(delta_p)
    coeff = method.mo_coeff  # orthonormal representation
    delta_f = coeff.swapaxes(-1, -2) @ (method.Sm12 @ delta_f @ method.Sm12) @ coeff
    result = np.empty_like(delta_x)
    for spin in range(2):
        energy, occ = method.mo_energy[spin], method.mo_occ[spin]
        difference = energy[:, None] - energy[None, :]
        occupation_difference = occ[:, None] - occ[None, :]
        close = np.abs(difference) < 1e-10
        slope = np.zeros_like(occ) if method.temp == 0 else -method.beta * occ * (1 - occ)
        if method.temp == 0 and np.any(close & (np.abs(occupation_difference) > 1e-10)):
            if getattr(method, "occupation_method", "aufbau") in ("imom", "mom"):
                raise ValueError("The IMOM/MOM determinant has an occupied/virtual degeneracy; "
                                 "its orbital response is not uniquely defined.")
            raise ValueError("A zero-temperature spin channel has a degenerate Fermi level; use temp > 0.")
        divided = np.divide(occupation_difference, difference,
                            out=np.zeros_like(difference), where=~close)
        divided[close] = np.broadcast_to((slope[:, None] + slope[None, :]) / 2, difference.shape)[close]
        response = divided * delta_f[spin]
        if abs(slope.sum()) > 1e-15:
            delta_mu = np.dot(slope, np.diag(delta_f[spin])) / slope.sum()
            response[np.diag_indices_from(response)] -= slope * delta_mu
        result[spin] = method.Sm12 @ coeff[spin] @ response @ coeff[spin].T @ method.Sm12
    return result


def pseudo_inverse_action(method, resid_mat, allow_inexact=False):
    """Low-rank approximation of Jacobian pseudoinverse kernel acting on residual matrix."""
    residual = np.asarray(resid_mat, dtype=float)
    norm = np.linalg.norm(residual)
    if norm < 1e-14:
        return np.zeros_like(residual)
    if method.max_rank < 1 or not 0 < method.rel_res_error_tol < 1:
        raise ValueError("Require max_rank >= 1 and 0 < rel_res_error_tol < 1.")
    potential_response = method.interface.get_potential_response(method.dm_prop)
    basis, actions = [], []
    vector = residual.ravel() / norm
    for rank in range(min(int(method.max_rank), residual.size)):
        basis.append(vector.copy())
        action = density_response(method, potential_response, vector.reshape(residual.shape))
        action = (action @ method.S - vector.reshape(residual.shape)).ravel()
        actions.append(action)
        matrix = np.column_stack(actions)
        weights = np.linalg.lstsq(matrix, residual.ravel(), rcond=None)[0]
        error = np.linalg.norm(matrix @ weights - residual.ravel()) / norm
        if error <= method.rel_res_error_tol:
            method.converged_rank.append(rank + 1)
            return (np.column_stack(basis) @ weights).reshape(residual.shape)
        # Reorthogonalization prevents loss of orthogonality near convergence.
        vector = action.copy()
        for _ in range(2):
            for previous in basis:
                vector -= np.dot(previous, vector) * previous
        length = np.linalg.norm(vector)
        if length < 1e-13:
            break
        vector /= length
    message = (f"Unrestricted Jacobian did not converge at rank {len(basis)} "
               f"(relative error {error:.3e}; tolerance {method.rel_res_error_tol:.3e}).")
    if not allow_inexact:
        raise RuntimeError(message)
    log.warning(f"{message} Using the best available action.")
    method.converged_rank.append(len(basis))
    return (np.column_stack(basis) @ weights).reshape(residual.shape)
