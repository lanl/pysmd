"""ShadowMD.update_energies regression tests."""

import unittest
import numpy as np

from pyscf import dft, gto, scf

from pysmd.closed_shell import shadow_md
from pysmd.interface.pyscf import PyscfDriver
from pysmd.common import logger


class TestShadowMDUpdateEnergies(unittest.TestCase):

    energy_atol = 1e-10

    def make_mol(self):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = "sto-3g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    def assert_energy_matches_pyscf(self, mf):
        mf.kernel()
        self.assertTrue(mf.converged)

        dm = mf.make_rdm1()
        h1e = mf.get_hcore()
        vhf = mf.get_veff(mf.mol, dm)
        ref_e_elec, ref_e2 = mf.energy_elec(dm=dm, h1e=h1e, vhf=vhf)

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            dm = qm_interface.compute_density_matrix()
            h1e = qm_interface.compute_core_hamiltonian_matrix()
            vj = qm_interface.compute_coulomb_matrix(dm=dm)
            vk = qm_interface.compute_exchange_matrix(dm=dm)
            ref_shadow_e2 = (
                np.einsum("ij, ji->", vj, dm, optimize=True)
                - 0.5 * np.einsum("ij, ji->", vk, dm, optimize=True)
            )
            ref_e_xc, _ = qm_interface.compute_xc_terms(dm=dm)

            md = shadow_md.ShadowMD(qm_interface=qm_interface)
            md.h1e = h1e

            md.update_energies(
                dm_ao=dm,
                dm_prop=dm,
                dm_lin=dm,
                compute_e_nuc=True,
            )
        finally:
            log.disabled = log_disabled

        np.testing.assert_allclose(md.e1, ref_e_elec - ref_e2,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.e2, ref_shadow_e2,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.e_xc, ref_e_xc,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.e2 + md.e_xc, ref_e2,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.e_elec, ref_e_elec,
                                   atol=self.energy_atol)
        np.testing.assert_allclose(md.e_tot, ref_e_elec + mf.energy_nuc(),
                                   atol=self.energy_atol)

    def test_rhf_energy_matches_pyscf(self):
        self.assert_energy_matches_pyscf(scf.RHF(self.make_mol()))

    def test_rks_lda_energy_matches_pyscf(self):
        self.assert_energy_matches_pyscf(dft.RKS(self.make_mol(), xc="lda,vwn5"))

    def test_rks_pbe_energy_matches_pyscf(self):
        self.assert_energy_matches_pyscf(dft.RKS(self.make_mol(), xc="pbe"))


if __name__ == "__main__":
    unittest.main()
