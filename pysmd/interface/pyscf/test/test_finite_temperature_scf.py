"""Finite-temperature SCF checks against PySCF."""

import unittest
import numpy as np

from pyscf import data, dft, gto, scf

from pysmd.closed_shell import newton_raphson
from pysmd.interface.pyscf import PyscfDriver
from pysmd.common import logger


class TestFiniteTemperatureSCF(unittest.TestCase):
    """Compare PySMD NewtonRaphson energies with PySCF smearing."""

    basis = "cc-pvdz"
    temperature = 500.0
    energy_atol = 1e-10


    def make_mol(self):
        mol = gto.Mole()
        mol.atom = """
            H    0.0   0.0   1.5
            H    0.0   0.0   0.0
        """
        mol.basis = self.basis
        mol.verbose = 0
        mol.unit = "bohr"
        mol.build()
        return mol

    def make_reference_mf(self, method, xc=None):
        mol = self.make_mol()

        if method == "rhf":
            mf = scf.RHF(mol)
        elif method == "rks":
            mf = dft.rks.RKS(mol, xc=xc) if xc is not None else dft.rks.RKS(mol)
        else:
            raise ValueError(f"Unsupported method: {method}")

        mf.max_cycle = 150
        mf.conv_tol = 1e-12

        sigma = self.temperature * (data.nist.BOLTZMANN / data.nist.HARTREE2J)
        mf = scf.addons.smearing(
            mf,
            sigma=sigma,
            method="fermi",
            fix_spin=True,
        )
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertIsNotNone(mf.e_free)
        return mf

    def run_pysmd_nrscf(self, method, xc=None):
        mol = self.make_mol()

        if method == "rhf":
            mf = scf.RHF(mol)
        elif method == "rks":
            mf = dft.rks.RKS(mol, xc=xc) if xc is not None else dft.rks.RKS(mol)
        else:
            raise ValueError(f"Unsupported method: {method}")

        mf.max_cycle = 0
        mf.kernel()

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            scf_params = {
                "n_atom": mol.natm,
                "n_elec": mol.nelectron,
                "n_mo": mol.nao_nr(),
                "n_occ": mol.nelectron // 2,
                "temp": self.temperature,
                "max_rank": mol.nao_nr(),
                "rel_res_error_tol": 1e-10,
                "energy_diff_error_tol": 1e-12,
            }

            nrscf = newton_raphson.NewtonRaphson(qm_interface, scf_params)
            nrscf.kernel()
            scf_converged = qm_interface.get_scf_convergence()
        finally:
            log.disabled = log_disabled

        self.assertTrue(scf_converged)
        return nrscf

    def assert_energy_matches_pyscf(self, method, xc=None):
        reference = self.make_reference_mf(method=method, xc=xc)
        nrscf = self.run_pysmd_nrscf(method=method, xc=xc)

        # PySMD's finite-temperature total energy includes the -T S term.
        np.testing.assert_allclose(
            nrscf.e_tot,
            reference.e_free,
            atol=self.energy_atol,
        )

    def test_rhf_energy_matches_pyscf_smearing(self):
        self.assert_energy_matches_pyscf(method="rhf")

    def test_rks_default_energy_matches_pyscf_smearing(self):
        self.assert_energy_matches_pyscf(method="rks")

    def test_rks_pbe_energy_matches_pyscf_smearing(self):
        self.assert_energy_matches_pyscf(method="rks", xc="pbe")


if __name__ == "__main__":
    unittest.main()
