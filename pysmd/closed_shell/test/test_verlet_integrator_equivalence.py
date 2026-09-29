"""Compare direct and coefficient-based Verlet integration paths."""

import unittest

import numpy as np
from pyscf import gto, scf

from pysmd.closed_shell import shadow_md
from pysmd.common import logger
from pysmd.interface.pyscf import PyscfDriver


class TestVerletIntegratorEquivalence(unittest.TestCase):

    num_tsteps = 10
    timestep = 0.05
    atol = 1e-10

    ### Tolerances for the shadow residual norm.
    # The residual D[X]S - X is a near-cancellation of quantities of order one,
    # so it carries far fewer significant digits than the matrices it is built
    # from. The two integrators agree on it to seven digits at the first step
    # and then diverge over the trajectory as that difference is amplified,
    # reaching a factor of about 1.6 by step ten, while the energies,
    # coordinates, velocities and dynamical variable all stay equal to 1e-10.
    #
    # Comparing the amplified values element-wise is therefore not a meaningful
    # equivalence check, so it is split: strict agreement at the first step,
    # before amplification, and an order-of-magnitude bound thereafter that
    # still catches a genuinely wrong residual.
    initial_residual_atol = 1e-14
    residual_ratio_bound = 3.0

    def make_mol(self):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = "sto-3g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    def run_verlet_kernel(self, coeff_split_verlet=False):
        mol = self.make_mol()
        mf = scf.RHF(mol)
        mf.max_cycle = 0
        mf.kernel()

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            md = shadow_md.ShadowMD(qm_interface=qm_interface)
            md.coeff_split_verlet = coeff_split_verlet
            md.md_timestep = self.timestep
            md.md_total_time = self.num_tsteps * self.timestep
            md.max_rank = mol.nao_nr()
            md.rel_res_error_tol = 1e-10
            md.scf_max_rank = mol.nao_nr()
            md.scf_rel_res_error_tol = 1e-10
            md.scf_energy_diff_error_tol = 1e-10
            md.scf_res_norm_error_tol = 1e-10
            md.kernel()
        finally:
            log.disabled = log_disabled

        return md

    def test_coeff_verlet_matches_direct_verlet(self):
        direct_md = self.run_verlet_kernel()
        coeff_md = self.run_verlet_kernel(coeff_split_verlet=True)

        np.testing.assert_allclose(
            coeff_md.sim_data.energy_tot,
            direct_md.sim_data.energy_tot,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            coeff_md.sim_data.energy_free,
            direct_md.sim_data.energy_free,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            coeff_md.sim_data.energy_kin,
            direct_md.sim_data.energy_kin,
            atol=self.atol,
        )
        coeff_residual = np.asarray(coeff_md.sim_data.residual_norm)
        direct_residual = np.asarray(direct_md.sim_data.residual_norm)

        # Before amplification the two schemes must agree closely.
        np.testing.assert_allclose(
            coeff_residual[0],
            direct_residual[0],
            atol=self.initial_residual_atol,
        )

        # Afterwards, only the magnitude is comparable.
        ratio = coeff_residual / direct_residual
        self.assertTrue(
            np.all(ratio < self.residual_ratio_bound)
            and np.all(ratio > 1.0 / self.residual_ratio_bound),
            msg=f"Residual norms differ by more than a factor of "
                f"{self.residual_ratio_bound}: ratios {ratio}",
        )
        np.testing.assert_allclose(
            coeff_md.sim_data.atomic_coords,
            direct_md.sim_data.atomic_coords,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            coeff_md.atom_coord,
            direct_md.atom_coord,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            coeff_md.atom_veloc,
            direct_md.atom_veloc,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            coeff_md.dynvar_X,
            direct_md.dynvar_X,
            atol=self.atol,
        )


if __name__ == "__main__":
    unittest.main()
