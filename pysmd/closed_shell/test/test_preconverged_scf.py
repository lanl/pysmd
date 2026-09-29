"""Tests for initializing ShadowMD from converged mean-field objects."""

import unittest

import numpy as np
from pyscf import data, dft, gto, scf

from pysmd.closed_shell import shadow_md
from pysmd.common import logger
from pysmd.interface.pyscf import PyscfDriver


class TestPreconvergedSCF(unittest.TestCase):

    temperature = 500.0
    energy_atol = 1e-10


    def make_mol(self):
        return gto.M(
            atom="Li 0 0 0; H 0 0 3.0",
            basis="sto-3g",
            unit="bohr",
            verbose=0,
        )


    def make_reference(self, method):
        mol = self.make_mol()
        if method == "rhf":
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol, xc="lda,vwn")

        sigma = self.temperature * (
            data.nist.BOLTZMANN / data.nist.HARTREE2J
        )
        mf = scf.addons.smearing(
            mf,
            sigma=sigma,
            method="fermi",
            fix_spin=True,
        )
        mf.conv_tol = 1e-12
        mf.kernel()
        self.assertTrue(mf.converged)
        return mf


    def make_md(self, mf):
        interface = PyscfDriver(pyscf_mf=mf)
        return shadow_md.ShadowMD(qm_interface=interface)


    def test_reuses_compatible_rhf_and_rks_wavefunctions(self):
        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            for method in ("rhf", "rks"):
                with self.subTest(method=method):
                    mf = self.make_reference(method)
                    md = self.make_md(mf)
                    md.verify_scf_convergence()

                    for key in md._scf_keys:
                        self.assertIsNotNone(getattr(md, key), key)
                    for key in (
                        "dm_ao", "dm_orth", "dynvar_X", "fock",
                        "fock_orth", "mo_energy", "mo_coeff", "mo_occ",
                        "mu", "beta", "e_free", "e_tot",
                    ):
                        self.assertTrue(np.all(np.isfinite(getattr(md, key))), key)
                    np.testing.assert_allclose(
                        md.e_tot,
                        mf.e_free,
                        atol=self.energy_atol,
                    )
        finally:
            log.disabled = log_disabled


    def test_temperature_mismatch_runs_newton_raphson(self):
        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            mf = self.make_reference("rhf")
            md = self.make_md(mf)
            md.temp = 600.0
            md.scf_max_rank = md.n_mo
            md.scf_rel_res_error_tol = 1e-10
            md.scf_energy_diff_error_tol = 1e-12
            md.scf_res_norm_error_tol = 1e-10

            md.verify_scf_convergence()
        finally:
            log.disabled = log_disabled

        self.assertTrue(md.interface.can_reuse_scf(temp=md.temp))
        self.assertEqual(md.interface.temp, md.temp)
        for key in md._scf_keys:
            self.assertIsNotNone(getattr(md, key), key)


if __name__ == "__main__":
    unittest.main()
