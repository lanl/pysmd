"""Tests for the exchange-correlation functional support gate of the RKS interface.

Support is restricted to pure LDA and GGA functionals, i.e. the first two rungs
of the Jacob's ladder classification. Everything else must be rejected when the
interface is constructed rather than silently producing wrong physics.
"""

import unittest

import numpy as np
from pyscf import dft, gto, scf

from pysmd.common import logger
from pysmd.interface.pyscf import GPU4RHF, GPU4RKS, PyscfDriver, RKS
from pysmd.interface.pyscf.gpu import _GPU4PySCFBase
from pysmd.interface.pyscf.rks import verify_supported_xc
from pysmd.lib import linalg_helper


class TestXCFunctionalSupport(unittest.TestCase):

    ### Functionals that must be accepted
    supported_xc = (
        "lda,vwn5",
        "lda,vwn",
        "svwn",
        "pbe",
        "blyp",
        "pbe,pbe",
    )

    ### Global hybrids: nonzero fraction of exact exchange
    hybrid_xc = ("pbe0", "b3lyp", "b3lyp5")

    ### Range-separated hybrids: nonzero range-separation parameter
    range_separated_xc = ("camb3lyp", "wb97x")

    ### Meta-GGAs: rung 3, depend on the kinetic energy density
    meta_gga_xc = ("tpss", "scan", "m06")

    ### Non-local correlation
    nlc_xc = ("wb97x-v",)

    def setUp(self):
        self.log = logger.getLogger("pysmd")
        self._log_disabled = self.log.disabled
        self.log.disabled = True

    def tearDown(self):
        self.log.disabled = self._log_disabled

    def make_mol(self):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = "sto-3g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    ### Rejected functionals ------------------------------------------------

    def assert_rejected(self, xc):
        """The functional must be rejected through both construction paths."""
        # Path 1: molecule plus functional string.
        with self.assertRaises(NotImplementedError):
            RKS(pyscf_mol=self.make_mol(), xc=xc)

        # Path 2: pre-built mean-field object, where the string bypasses the
        # constructor argument entirely and must be read back off the object.
        with self.assertRaises(NotImplementedError):
            RKS(pyscf_mf=dft.RKS(self.make_mol(), xc=xc))

        # Path 3: the factory, which dispatches on the presence of 'xc'.
        with self.assertRaises(NotImplementedError):
            PyscfDriver(pyscf_mol=self.make_mol(), xc=xc)

    def test_hybrid_functionals_rejected(self):
        for xc in self.hybrid_xc:
            with self.subTest(xc=xc):
                self.assert_rejected(xc)

    def test_range_separated_functionals_rejected(self):
        for xc in self.range_separated_xc:
            with self.subTest(xc=xc):
                self.assert_rejected(xc)

    def test_meta_gga_functionals_rejected(self):
        for xc in self.meta_gga_xc:
            with self.subTest(xc=xc):
                self.assert_rejected(xc)

    def test_nlc_functionals_rejected(self):
        for xc in self.nlc_xc:
            with self.subTest(xc=xc):
                self.assert_rejected(xc)

    def test_pure_hartree_fock_rejected(self):
        """A pure 'HF' functional string belongs to the RHF interface."""
        with self.assertRaises(NotImplementedError):
            RKS(pyscf_mol=self.make_mol(), xc="hf")

    def test_unrecognized_functional_raises_value_error(self):
        """An unparseable name is a user error, not an unsupported feature."""
        with self.assertRaises(ValueError):
            RKS(pyscf_mol=self.make_mol(), xc="not_a_functional")

    ### Accepted functionals ------------------------------------------------

    def test_supported_functionals_accepted(self):
        for xc in self.supported_xc:
            with self.subTest(xc=xc):
                qm_interface = RKS(pyscf_mol=self.make_mol(), xc=xc)
                self.assertEqual(qm_interface.xc, xc)

    def test_default_functional_is_vwn5(self):
        """The default is spelled with literal libxc keys.

        PySCF's 'VWN' has no literal entry in its functional table and resolves
        to LDA_C_VWN, which is the VWN5 parameterization. Naming VWN5 directly
        avoids relying on that fuzzy lookup.
        """
        qm_interface = RKS(pyscf_mol=self.make_mol())
        self.assertEqual(qm_interface.xc, "LDA,VWN5")

    def test_default_matches_pyscf_default_energy(self):
        """'LDA,VWN5' and PySCF's 'LDA,VWN' are the same functional."""
        reference = dft.RKS(self.make_mol(), xc="LDA,VWN")
        reference.verbose = 0
        reference.kernel()

        test = dft.RKS(self.make_mol(), xc="LDA,VWN5")
        test.verbose = 0
        test.kernel()

        self.assertAlmostEqual(test.e_tot, reference.e_tot, places=12)

    ### Absence of exact exchange -------------------------------------------

    def test_rks_exchange_matrix_is_zero(self):
        """Pure functionals carry no exact exchange."""
        qm_interface = RKS(pyscf_mol=self.make_mol(), xc="pbe")
        nao = qm_interface.get_num_basis_functions()
        dm = qm_interface.compute_guess_density_matrix()

        vk = qm_interface.compute_exchange_matrix(dm=dm)
        self.assertEqual(vk.shape, (nao, nao))
        np.testing.assert_allclose(vk, np.zeros((nao, nao)))

    def test_rks_exchange_gradient_is_zero(self):
        qm_interface = RKS(pyscf_mol=self.make_mol(), xc="pbe")
        nao = qm_interface.get_num_basis_functions()
        dm = qm_interface.compute_guess_density_matrix()

        exchange_grad = qm_interface.compute_exchange_gradient(dm=dm)
        self.assertEqual(exchange_grad.shape, (3, nao, nao))
        np.testing.assert_allclose(exchange_grad, np.zeros((3, nao, nao)))

    def test_rhf_exchange_matrix_is_nonzero(self):
        """The zero-exchange default must not leak into Hartree-Fock."""
        mf = scf.RHF(self.make_mol())
        mf.verbose = 0
        qm_interface = PyscfDriver(pyscf_mf=mf)
        dm = qm_interface.compute_guess_density_matrix()

        vk = qm_interface.compute_exchange_matrix(dm=dm)
        self.assertGreater(np.linalg.norm(vk), 1e-6)

        exchange_grad = qm_interface.compute_exchange_gradient(dm=dm)
        self.assertGreater(np.linalg.norm(exchange_grad), 1e-6)

    ### Grid response --------------------------------------------------------

    def test_grid_response_enabled(self):
        """Forces must be the exact derivative of the quadrature energy."""
        qm_interface = RKS(pyscf_mol=self.make_mol(), xc="pbe")
        self.assertTrue(qm_interface.grad.grid_response)


class TestSharedXCGate(unittest.TestCase):
    """The functional gate must be backend-independent.

    Every Kohn-Sham interface applies the same restriction through the shared
    :func:`verify_supported_xc`. These tests exercise it directly so that the
    GPU4PySCF interface is covered on machines without a GPU: the gate runs
    before any device work, so the restriction is verifiable without CUDA.
    """

    def setUp(self):
        self.log = logger.getLogger("pysmd")
        self._log_disabled = self.log.disabled
        self.log.disabled = True

    def tearDown(self):
        self.log.disabled = self._log_disabled

    def make_mol(self):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = "sto-3g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    def test_gate_rejects_unsupported_functionals(self):
        for xc in ("pbe0", "b3lyp", "camb3lyp", "wb97x", "tpss", "scan", "hf"):
            with self.subTest(xc=xc):
                mf = dft.RKS(self.make_mol(), xc=xc)
                with self.assertRaises(NotImplementedError):
                    verify_supported_xc(mf.xc, mf)

    def test_gate_accepts_supported_functionals(self):
        for xc, expected in (
            ("lda,vwn5", "LDA"),
            ("pbe", "GGA"),
            ("blyp", "GGA"),
        ):
            with self.subTest(xc=xc):
                mf = dft.RKS(self.make_mol(), xc=xc)
                self.assertEqual(verify_supported_xc(mf.xc, mf), expected)

    def test_gpu_rks_rejects_hybrid_before_touching_array_backend(self):
        """A rejected functional must not leave the array backend switched.

        ``GPU4RKS`` validates before ``_initialize_gpu``, which mutates the
        process-wide array backend. This mirrors the existing guarantee for a
        failed GPU4PySCF import.
        """
        backend_before = linalg_helper.get_linalg_backend_name()

        # Stand in for a GPU4PySCF RKS object. The gate only needs the
        # functional string and do_nlc(), so a CPU object exercises the same
        # code path that GPU4RKS.__init__ runs before any device work.
        mf = dft.RKS(self.make_mol(), xc="pbe0")
        with self.assertRaises(NotImplementedError):
            verify_supported_xc(mf.xc, mf)

        self.assertEqual(
            linalg_helper.get_linalg_backend_name(), backend_before
        )

    def test_gpu_interfaces_do_not_define_exchange_or_dead_methods(self):
        """GPU RKS must not reintroduce removed machinery.

        ``GPU4RKS`` must not define its own exchange matrix: pure functionals
        carry no exact exchange, so it inherits the zero-returning default.
        The two dead gradient hooks must not reappear on either GPU class.
        """
        self.assertNotIn("compute_exchange_matrix", vars(GPU4RKS))

        for cls in (GPU4RHF, GPU4RKS, _GPU4PySCFBase):
            for name in (
                "compute_2e_gradient",
                "compute_xc_grid_response_gradient",
            ):
                with self.subTest(cls=cls.__name__, method=name):
                    self.assertNotIn(name, vars(cls))

    def test_gpu_rks_default_functional_matches_cpu(self):
        import inspect

        gpu_default = inspect.signature(
            GPU4RKS.__init__
        ).parameters["xc"].default
        cpu_default = inspect.signature(
            RKS.__init__
        ).parameters["xc"].default
        self.assertEqual(gpu_default, cpu_default)
        self.assertEqual(gpu_default, "LDA,VWN5")


if __name__ == "__main__":
    unittest.main()
