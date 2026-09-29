"""ShadowMD kernel regression tests."""

import unittest
import numpy as np

from pyscf import dft, gto, scf

from pysmd.closed_shell import shadow_md
from pysmd.interface.pyscf import PyscfDriver
from pysmd.common import logger


class TestShadowMDKernel(unittest.TestCase):

    num_tsteps = 10
    energy_atol = 1e-10
    residual_atol = 1e-13
    rks_num_tsteps = 10
    rks_residual_atol = 1e-10

    def make_mol(self, basis="sto-3g"):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = basis
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    def run_kernel(self, mf, num_tsteps, timestep):
        mol = mf.mol
        mf.max_cycle = 0
        mf.kernel()

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            md = shadow_md.ShadowMD(qm_interface=qm_interface)
            md.md_timestep = timestep
            md.md_total_time = num_tsteps * timestep
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

    def run_rhf_kernel(self):
        mol = self.make_mol()
        return self.run_kernel(
            mf=scf.RHF(mol),
            num_tsteps=self.num_tsteps,
            timestep=0.05,
        )

    def run_rks_kernel(self, xc):
        mol = self.make_mol(basis="6-31g")
        return self.run_kernel(
            mf=dft.RKS(mol, xc=xc),
            num_tsteps=self.rks_num_tsteps,
            timestep=0.5,
        )


    def assert_sim_data_matches(
        self,
        md,
        expected_energy_tot,
        expected_energy_free,
        expected_energy_kin,
        expected_residual_norm,
        residual_atol,
        initial_energy_tot,
        initial_energy_free,
        initial_energy_kin,
    ):
        self.assertEqual(md.sim_data.num_tsteps, md.md_num_tsteps)
        self.assertEqual(len(md.sim_data.energy_tot), md.md_num_tsteps)
        self.assertEqual(md.sim_data.initial_atomic_coords.shape, (md.n_atom, 3))

        self.assertAlmostEqual(md.sim_data.initial_energy_tot, initial_energy_tot,
                               places=12)
        self.assertAlmostEqual(md.sim_data.initial_energy_free, initial_energy_free,
                               places=12)
        self.assertAlmostEqual(md.sim_data.initial_energy_kin, initial_energy_kin,
                               places=12)

        np.testing.assert_allclose(md.sim_data.energy_tot, expected_energy_tot,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.sim_data.energy_free, expected_energy_free,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.sim_data.energy_kin, expected_energy_kin,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.sim_data.residual_norm, expected_residual_norm,
                                   atol=residual_atol)

    def test_rhf_short_trajectory(self):
        md = self.run_rhf_kernel()

        expected_energy_tot = np.array([
            -7.8622463104174045,
            -7.862246310425643,
            -7.862246310420867,
            -7.862246310412388,
            -7.8622463104142595,
            -7.862246310422404,
            -7.862246310423183,
            -7.862246310415966,
            -7.862246310413165,
            -7.862246310418889,
        ])
        expected_energy_free = np.array([
            -7.86224631059506,
            -7.862246311136264,
            -7.862246312019764,
            -7.862246313254871,
            -7.862246314855637,
            -7.862246316817984,
            -7.862246319128273,
            -7.862246321785871,
            -7.8622463248031895,
            -7.862246328184335,
        ])
        expected_energy_kin = np.array([
            1.7765530580657878e-10,
            7.106211201975021e-10,
            1.5988971343587606e-09,
            2.8424828338497617e-09,
            4.441377498208994e-09,
            6.395580200830426e-09,
            8.705089809451177e-09,
            1.1369904986357927e-08,
            1.439002418783556e-08,
            1.776544566380937e-08,
        ])
        expected_residual_norm = np.array([
            7.907302308094838e-10,
            1.7079789297398392e-09,
            1.1760367252311842e-09,
            2.3099204046440327e-10,
            4.3860729832387825e-10,
            1.3479050296743218e-09,
            1.4331541399205933e-09,
            6.295080673140431e-10,
            3.1734616386516426e-10,
            9.552603433516133e-10,
        ])

        self.assert_sim_data_matches(
            md,
            expected_energy_tot,
            expected_energy_free,
            expected_energy_kin,
            expected_residual_norm,
            self.residual_atol,
            -7.862246310410317,
            -7.862246310410317,
            0.0,
        )

    def test_rks_lda_short_trajectory(self):
        md = self.run_rks_kernel("lda,vwn5")

        expected_energy_tot = np.array([
            -7.910199711731511,
            -7.910199711790159,
            -7.910199711790213,
            -7.910199711790292,
            -7.910199711790402,
            -7.9101997117905265,
            -7.910199711790694,
            -7.910199711790866,
            -7.91019971179107,
            -7.910199711791298,
        ])
        expected_energy_free = np.array([
            -7.910199715969919,
            -7.910199728743602,
            -7.910199749934745,
            -7.910199779601014,
            -7.910199817741084,
            -7.910199864353221,
            -7.91019991943536,
            -7.9101999829849845,
            -7.910200054999268,
            -7.910200135474964,
        ])
        expected_energy_kin = np.array([
            4.238408489292101e-09,
            1.6953443137301836e-08,
            3.814453170270237e-08,
            6.781072247475976e-08,
            1.0595068204348378e-07,
            1.5256269513202347e-07,
            2.0764466609425596e-07,
            2.7119411851742796e-07,
            3.432081977065178e-07,
            4.236836656108164e-07,
        ])
        expected_residual_norm = np.array([
            2.2169019517514274e-07,
            5.7818409809727596e-07,
            5.1764140866727339e-07,
            1.3761661773447407e-07,
            1.1287844908428555e-07,
            3.8056965367682690e-07,
            5.9506741858625395e-07,
            3.1252845311154170e-07,
            1.4517843385504661e-07,
            3.2617625420892940e-07,
        ])

        self.assert_sim_data_matches(
            md,
            expected_energy_tot,
            expected_energy_free,
            expected_energy_kin,
            expected_residual_norm,
            self.rks_residual_atol,
            -7.910199711727291,
            -7.910199711727291,
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
