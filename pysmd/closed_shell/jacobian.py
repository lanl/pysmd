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

"""Functions to compute Jacobian inverses acting on residual matrices."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

if TYPE_CHECKING:
    from pysmd.closed_shell.newton_raphson import NewtonRaphson
    from pysmd.closed_shell.shadow_md import ShadowMD


def pseudo_inverse_action(
    method: NewtonRaphson | ShadowMD,
    resid_mat: Any,
    precond: Any | None = None,
) -> Any:
    """Low-rank approximation of Jacobian pseudoinverse kernel acting on residual matrix."""
    # Initialize V matrix
    vmat = la.copy(resid_mat)

    # Begin low-rank updates
    for m in range(method.max_rank):
        rank_m = m + 1 # Define dimension for creating matrices
        log.debug(f"\n> current rank {rank_m:d}:")

        # Construct V matrix and normalize
        if m == 0:
            v_mat = vmat[None]
        else:
            v_mat = la.stack((*v_mat, w_mat[m-1] / la.linalg.norm(w_mat[m-1])))
            for j in range(rank_m):
                trace_term = la.trace(la.matmul(v_mat[m].T, v_mat[j]))
                v_mat[m] = v_mat[m] - (trace_term * v_mat[j])
        v_mat[m] = v_mat[m] / la.linalg.norm(v_mat[m])

        # Recursive density matrix update
        delta_dm = la.matmul(v_mat[m], method.Sm1)
        p1 = recursive_canonical_dm_pt_response(method, delta_dm)
        wmat = la.matmul(method.Sm12, la.matmul(p1, method.Sp12))

        # Construct W matrix
        if m == 0:
            w_mat = wmat[None]
        else:
            w_mat = la.stack((*w_mat, wmat))
        w_mat[m] = w_mat[m] - v_mat[m]

        # Apply pre-conditioner
        if precond is not None:
            w_mat[m] = la.matmul(precond, w_mat[m])

        # Construct M matrix by inverting O matrix
        o_mat = la.einsum("ipq, jpq-> ij", w_mat, w_mat, optimize = True)
        m_mat = la.linalg.inv(o_mat)

        # Error estimate: resolution of identity acting on residual function
        # Algorithm 3, arXiv:2003.09050
        ident_res = la.zeros((method.n_mo, method.n_mo))
        for i in range(rank_m):
            for j in range(rank_m):
                trace_term = la.trace(la.matmul(w_mat[j].T, resid_mat))
                ident_res = ident_res + w_mat[i] * (m_mat[i,j] * trace_term)

        scf_rank_conv = la.linalg.norm(ident_res - resid_mat) / la.linalg.norm(resid_mat)
        log.debug(f"- updated ROI residual norm:\n {scf_rank_conv:10e}")

        # Check convergence
        if (scf_rank_conv <= method.rel_res_error_tol):
            log.debug(f"* DONE!")
            break
        elif (m + 1 == method.max_rank):
            # Out of rank budget. Fall through and return the rank-max_rank
            # approximation rather than aborting: every matrix needed to build
            # it is complete and valid at this point, an inexact kernel
            # degrades the dynamics gradually, and ending the trajectory
            # outright discards the whole run.
            #
            # Note this is deliberately not an exception. Earlier revisions
            # returned None here, which reached the caller's matrix arithmetic
            # and surfaced as an unrelated dtype error several frames away.
            log.warning(
                f"Low-rank Jacobian kernel did not converge at maximum rank "
                f"{method.max_rank}: relative residual error "
                f"{scf_rank_conv:.6e} exceeds tolerance "
                f"{method.rel_res_error_tol:.6e}. Continuing with the "
                f"rank-{method.max_rank} approximation, which may let the "
                f"dynamics drift; loosen rel_res_error_tol or raise max_rank."
            )
        else: continue

    # Construct K*(D*S-X) matrix to return
    return_mat = la.zeros((method.n_mo, method.n_mo))
    for i in range(rank_m):
        for j in range(rank_m):
            trace_term = la.trace(la.matmul(w_mat[j].T, resid_mat))
            return_mat = return_mat + v_mat[i] * (m_mat[i,j] * trace_term)

    # Store convergence rank
    if hasattr(method, "converged_rank"):
        method.converged_rank.append(rank_m)
    else:
        method.converged_rank = [rank_m]

    return return_mat


def recursive_canonical_dm_pt_response(
    method: NewtonRaphson | ShadowMD,
    delta_dm: Any,
) -> Any:
    """Recursively calculate first-order density matrix perturbation response."""
    # Potential perturbation matrix
    f_perturb = method.interface.compute_eff_potential_matrix(dm=delta_dm)

    # Orthogonalize
    f_perturb = la.matmul(
        la.matmul(method.Sm12.T.conj(), f_perturb), method.Sm12
    )
    f_perturb = la.matmul(
        la.matmul(method.mo_coeff.T.conj(), f_perturb), method.mo_coeff
    )

    # Initialize density matrices
    norm_cnst = 2.0 ** (-2.0 - float(method.pt_recursion_depth)) * method.beta
    p0 = (0.5 - norm_cnst * (method.mo_energy - method.mu))
    p1 = -(norm_cnst * f_perturb)

    # Recursive perturbation
    for __ in range(method.pt_recursion_depth):
        X1_temp = p0 * p1 + (p1 * p0).T
        Y0_temp = 1.0 / (2.0 * ((p0 ** 2) - p0) + 1.0)
        p0 = Y0_temp * (p0 ** 2)

        p1_temp = X1_temp + 2.0 * (p1 - X1_temp) * p0
        p1 = la.matmul(la.diag(Y0_temp), p1_temp)

    # Chemical potential derivative
    dp_dmu = (method.beta * p0) * (1.0 - p0)
    dmu = -(la.trace(p1) / la.sum(dp_dmu))

    # First-order response and orthogonalization
    p1 += la.diag(dp_dmu * dmu)
    p1_orth = la.matmul(
        la.matmul(method.mo_coeff, p1), method.mo_coeff.T.conj()
    )

    return p1_orth
