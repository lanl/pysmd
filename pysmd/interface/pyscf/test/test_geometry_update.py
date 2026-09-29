"""PySCF interface geometry-update regression tests."""

import unittest
import numpy as np

from pyscf import gto, scf

from pysmd.interface.pyscf import PyscfDriver
from pysmd.common import logger


class TestPyscfGeometryUpdate(unittest.TestCase):

    atol = 1e-12


    def test_rhf_integrals_follow_updated_geometry(self):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = "6-31g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.max_cycle = 0
        mf.kernel()

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            dm = qm_interface.compute_density_matrix()
            coords = qm_interface.get_atomic_coords()
            coords[1, 2] = 1.4
            qm_interface.get_updated_atomic_coords(coords)

            dm_pyscf = 2.0 * dm
            ref_vj, ref_vk = scf.hf.get_jk(qm_interface.mol, dm_pyscf)
            ref_vhf = scf.hf.get_veff(qm_interface.mol, dm_pyscf)

            test_vhf = qm_interface.compute_eff_potential_matrix(dm)
            test_vj = qm_interface.compute_coulomb_matrix(dm)
            test_vk = qm_interface.compute_exchange_matrix(dm)
        finally:
            log.disabled = log_disabled

        np.testing.assert_allclose(
            test_vhf,
            ref_vhf,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            test_vj,
            ref_vj,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            test_vk,
            ref_vk,
            atol=self.atol,
        )


if __name__ == "__main__":
    unittest.main()
