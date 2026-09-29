"""Tests for PySCF convergence and temperature compatibility."""

import unittest

import numpy as np
from pyscf import data, dft, gto, scf

from pysmd.interface.pyscf import PyscfDriver


class TestPyscfSCFConvergence(unittest.TestCase):

    temperature = 500.0


    def make_mol(self):
        return gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            unit="bohr",
            verbose=0,
        )


    def test_unconverged_mean_field(self):
        mf = scf.RHF(self.make_mol())
        mf.max_cycle = 0
        mf.kernel()
        interface = PyscfDriver(pyscf_mf=mf)

        self.assertFalse(interface.get_scf_convergence())
        self.assertFalse(interface.can_reuse_scf(temp=self.temperature))


    def test_converged_mean_field_without_temperature(self):
        mf = scf.RHF(self.make_mol()).run()
        interface = PyscfDriver(pyscf_mf=mf)

        self.assertTrue(interface.get_scf_convergence())
        self.assertFalse(interface.can_reuse_scf(temp=self.temperature))


    def test_converged_mean_field_temperature_match(self):
        sigma = self.temperature * (
            data.nist.BOLTZMANN / data.nist.HARTREE2J
        )
        mf = scf.addons.smearing(
            scf.RHF(self.make_mol()),
            sigma=sigma,
            method="fermi",
            fix_spin=True,
        ).run()
        interface = PyscfDriver(pyscf_mf=mf)

        self.assertTrue(interface.get_scf_convergence())
        self.assertTrue(interface.can_reuse_scf(temp=self.temperature))
        self.assertTrue(interface.can_reuse_scf(
            temp=self.temperature + 1e-7,
        ))
        self.assertFalse(interface.can_reuse_scf(
            temp=self.temperature + 1.0,
        ))


    def test_density_matrix_defaults_to_mean_field_orbitals(self):
        for mf in (
            scf.RHF(self.make_mol()),
            dft.RKS(self.make_mol(), xc="lda,vwn"),
        ):
            with self.subTest(method=type(mf).__name__):
                mf.kernel()
                interface = PyscfDriver(pyscf_mf=mf)

                np.testing.assert_allclose(
                    interface.compute_density_matrix(),
                    0.5 * mf.make_rdm1(),
                )


    def test_density_matrix_requires_available_orbitals(self):
        for mf in (
            scf.RHF(self.make_mol()),
            dft.RKS(self.make_mol(), xc="lda,vwn"),
        ):
            with self.subTest(method=type(mf).__name__):
                interface = PyscfDriver(pyscf_mf=mf)

                with self.assertRaisesRegex(ValueError, "are unavailable"):
                    interface.compute_density_matrix()


if __name__ == "__main__":
    unittest.main()
