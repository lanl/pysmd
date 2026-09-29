"""Newton-Raphson SCF regression tests."""

import unittest
import numpy as np

from pyscf import dft, gto, scf

from pysmd.closed_shell import newton_raphson
from pysmd.interface.pyscf import PyscfDriver
from pysmd.common import logger


class TestNewtonRaphson(unittest.TestCase):

    basis = "6-31g"
    temperature = 500.0
    energy_atol = 1e-9


    def make_mol(self):
        mol = gto.Mole()
        mol.atom = """
            O        0.000000    0.000000    0.117790
            H        0.000000    0.755453   -0.471161
            H        0.000000   -0.755453   -0.471161
        """
        mol.basis = self.basis
        mol.verbose = 0
        mol.build()
        return mol

    def run_h2o_nrscf(self, mf):
        mf.max_cycle = 0
        mf.kernel()

        mol = mf.mol
        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            nrscf = newton_raphson.NewtonRaphson(
                qm_interface=qm_interface,
                scf_params={
                    "n_atom": mol.natm,
                    "n_elec": mol.nelectron,
                    "n_mo": mol.nao_nr(),
                    "n_occ": mol.nelectron // 2,
                    "temp": self.temperature,
                    "frac_occ_tol": 1e-10,
                    "max_rank": mol.nao_nr(),
                    "rel_res_error_tol": 1e-10,
                    "energy_diff_error_tol": 1e-6,
                    "res_norm_error_tol": 1e-6,
                },
            )
            nrscf.kernel()
            scf_converged = qm_interface.get_scf_convergence()
        finally:
            log.disabled = log_disabled

        self.assertTrue(scf_converged)
        return nrscf

    def test_h2o_rhf_energy(self):
        """Test Newton-Raphson SCF with RHF."""
        mol = self.make_mol()
        nrscf = self.run_h2o_nrscf(scf.RHF(mol))

        expected_energy = -75.98383112063195
        np.testing.assert_allclose(
            nrscf.e_tot,
            expected_energy,
            atol=self.energy_atol,
        )

    def test_h2o_rks_energy(self):
        """Test Newton-Raphson SCF with RKS (LDA)."""
        mol = self.make_mol()
        nrscf = self.run_h2o_nrscf(dft.rks.RKS(mol))

        expected_energy = -75.81783954528726
        np.testing.assert_allclose(
            nrscf.e_tot,
            expected_energy,
            atol=self.energy_atol,
        )


if __name__ == "__main__":
    unittest.main()
