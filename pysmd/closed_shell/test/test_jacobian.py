"""Tests for the low-rank Jacobian kernel."""

import unittest

import numpy as np
from pyscf import gto, scf

from pysmd.closed_shell import jacobian, shadow_md
from pysmd.common import logger
from pysmd.interface.pyscf import PyscfDriver

JACOBIAN_LOGGER = "pysmd.closed_shell.jacobian"


class TestLowRankKernelNonConvergence(unittest.TestCase):
    """Exhausting the rank budget must warn, not abort.

    The kernel action feeds straight into matrix arithmetic in
    ``NewtonRaphson.kernel`` and ``ShadowMD.update_first_level_dm``, so
    returning a sentinel turned an unmet tolerance into an opaque NumPy dtype
    error several frames from the cause. Raising instead would end an otherwise
    usable trajectory, so the kernel returns its best available approximation
    and says so.
    """

    def setUp(self):
        self.log = logger.getLogger("pysmd")
        self._log_disabled = self.log.disabled
        self.log.disabled = True

    def tearDown(self):
        self.log.disabled = self._log_disabled

    def make_interface(self):
        mol = gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            unit="bohr",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.max_cycle = 0
        mf.kernel()
        return PyscfDriver(pyscf_mf=mf)

    def make_unconvergeable_method(self):
        """Return a ShadowMD primed with an unreachable kernel tolerance."""
        interface = self.make_interface()
        md = shadow_md.ShadowMD(qm_interface=interface)

        # A tolerance below double precision, with only a couple of ranks
        # available, cannot be met.
        md.max_rank = 2
        md.rel_res_error_tol = 1e-16

        md.S, md.Sm1, md.Sp12, md.Sm12 = interface.compute_overlap_matrices()
        md.h1e = interface.compute_core_hamiltonian_matrix()
        dm_ao = interface.compute_guess_density_matrix()
        md.fock = md.h1e + interface.compute_eff_potential_matrix(dm=dm_ao)
        md.fock_orth = np.matmul(
            np.matmul(md.Sm12.T.conj(), md.fock), md.Sm12
        )
        md.mo_energy, md.mo_coeff = np.linalg.eigh(md.fock_orth)
        md.mo_occ = np.zeros(md.n_mo)
        md.mo_occ[:md.n_occ] = 1.0
        md.mu = 0.0
        md.beta = 1.0

        residual = 1e-3 * np.eye(md.n_mo)
        return md, residual

    def test_non_convergence_returns_usable_matrix(self):
        """The best available approximation must come back, never None."""
        md, residual = self.make_unconvergeable_method()

        with self.assertLogs(JACOBIAN_LOGGER, level="WARNING"):
            result = jacobian.pseudo_inverse_action(md, residual)

        self.assertIsNotNone(
            result,
            msg="Returning None here reaches the caller's matrix arithmetic "
                "and surfaces as an unrelated dtype error.",
        )
        self.assertEqual(result.shape, (md.n_mo, md.n_mo))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_non_convergence_warning_reports_the_shortfall(self):
        """The warning must carry enough detail to act on."""
        md, residual = self.make_unconvergeable_method()

        with self.assertLogs(JACOBIAN_LOGGER, level="WARNING") as caught:
            jacobian.pseudo_inverse_action(md, residual)

        message = "\n".join(caught.output)
        self.assertIn("did not converge", message)
        self.assertIn("maximum rank", message)
        self.assertIn("rel_res_error_tol", message)

    def test_simulation_survives_unreachable_tolerance(self):
        """An unreachable tolerance must degrade the run, not end it.

        This is the whole point of warning rather than raising: a slightly
        inexact kernel perturbs the dynamics, while an exception discards the
        trajectory entirely.
        """
        interface = self.make_interface()
        md = shadow_md.ShadowMD(qm_interface=interface)
        md.max_rank = interface.get_num_basis_functions()
        md.rel_res_error_tol = 1e-16
        md.scf_max_rank = interface.get_num_basis_functions()
        md.scf_rel_res_error_tol = 1e-16
        md.md_timestep = 0.05
        md.md_num_tsteps = 2

        md.kernel()

        energies = np.asarray(md.sim_data.energy_tot)
        self.assertEqual(energies.shape, (2,))
        self.assertTrue(np.all(np.isfinite(energies)))

    def test_reachable_tolerance_emits_no_warning(self):
        """The warning must stay silent when the kernel does converge."""
        interface = self.make_interface()
        md = shadow_md.ShadowMD(qm_interface=interface)
        md.max_rank = interface.get_num_basis_functions()
        md.rel_res_error_tol = 1e-6
        md.scf_max_rank = interface.get_num_basis_functions()
        md.scf_rel_res_error_tol = 1e-6
        md.md_timestep = 0.05
        md.md_num_tsteps = 2

        jacobian_log = logger.getLogger(JACOBIAN_LOGGER)
        recorded = []
        original_warning = jacobian_log.warning
        jacobian_log.warning = lambda *args, **kwargs: recorded.append(args)
        try:
            md.kernel()
        finally:
            jacobian_log.warning = original_warning

        self.assertEqual(recorded, [])


if __name__ == "__main__":
    unittest.main()
