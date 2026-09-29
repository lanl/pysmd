"""Backend resolution and GPU4PySCF integration tests."""

import importlib.util
import sys
import unittest
from unittest import mock

import numpy as np
from pyscf import dft, gto, scf

from pysmd.closed_shell import shadow_md
import pysmd.interface.pyscf as pyscf_interface
from pysmd.common import logger
from pysmd.interface.pyscf import GPU4RHF, GPU4RKS, PyscfDriver, RHF, RKS
from pysmd.lib import linalg_helper


class TestBackendResolution(unittest.TestCase):

    def setUp(self):
        linalg_helper.set_linalg_backend("numpy")
        self.mol = gto.M(
            atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0
        )


    def tearDown(self):
        linalg_helper.set_linalg_backend("numpy")


    def test_cpu_default_and_auto_resolution(self):
        with mock.patch.object(
            pyscf_interface,
            "_load_gpu4pyscf",
            side_effect=AssertionError("GPU import attempted"),
        ):
            default = PyscfDriver(pyscf_mf=scf.RHF(self.mol))
            automatic = PyscfDriver(
                pyscf_mf=dft.RKS(self.mol), backend="auto"
            )
            molecule_only = PyscfDriver(pyscf_mol=self.mol, backend="auto")

        self.assertIsInstance(default, RHF)
        self.assertIsInstance(automatic, RKS)
        self.assertIsInstance(molecule_only, RHF)
        self.assertEqual(default.electronic_backend, "cpu")
        self.assertEqual(default.gradient_backend, "cpu")
        self.assertEqual(default.array_backend, "numpy")


    def test_mean_field_takes_precedence_over_molecule(self):
        other = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        interface = PyscfDriver(
            pyscf_mf=scf.RHF(self.mol), pyscf_mol=other
        )
        self.assertIs(interface.mol, self.mol)


    def test_invalid_backend_and_cpu_gpu_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Unsupported electronic backend"):
            PyscfDriver(pyscf_mol=self.mol, backend="invalid")
        with self.assertRaisesRegex(TypeError, "backend must be"):
            PyscfDriver(pyscf_mol=self.mol, backend=None)
        with self.assertRaisesRegex(TypeError, "expects GPU mean-field objects"):
            PyscfDriver(pyscf_mf=scf.RHF(self.mol), backend="gpu")

        open_shell = gto.M(
            atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0
        )
        with self.assertRaisesRegex(NotImplementedError, "ROHF"):
            PyscfDriver(pyscf_mf=scf.ROHF(open_shell))
        with self.assertRaisesRegex(TypeError, "scf.hf.RHF"):
            RHF(pyscf_mf=scf.ROHF(open_shell))
        with self.assertRaisesRegex(NotImplementedError, "UHF"):
            PyscfDriver(pyscf_mf=scf.UHF(open_shell))



    def test_failed_gpu_import_does_not_change_array_backend(self):
        with mock.patch.dict(sys.modules, {"gpu4pyscf": None}):
            with self.assertRaisesRegex(ImportError, "gpu4pyscf is not importable"):
                PyscfDriver(pyscf_mol=self.mol, backend="gpu")
        self.assertEqual(linalg_helper.get_linalg_backend_name(), "numpy")


def _gpu_is_available():
    if not importlib.util.find_spec("cupy"):
        return False
    if not importlib.util.find_spec("gpu4pyscf"):
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(_gpu_is_available(), "GPU4PySCF/CuPy/CUDA unavailable")
class TestGPU4PySCFBackend(unittest.TestCase):
    """Device-resident electronic structure and shadow gradients.

    The GPU interfaces evaluate both the electronic-structure operations and
    the full shadow-gradient contract on the device. References are taken from
    independently constructed CPU interfaces rather than from any delegate.
    """

    ### Tolerance for GPU-vs-CPU agreement of matrices and gradients.
    # Loose enough for the reduced-precision quadrature GPU4PySCF uses by
    # default, tight enough that a wrong term or factor cannot hide.
    comparison_atol = 1e-8

    ### Tolerance for translational invariance of the assembled gradient.
    translation_atol = 1e-8

    ### Tolerances for the end-to-end trajectory comparison.
    # Both backends run in double precision, so agreement is limited by
    # summation and quadrature ordering rather than by precision. For scale,
    # two *identical* CPU runs of these cases reproduce each other to about
    # 7e-14 in the energies and 1e-20 in the coordinates, while the nuclei
    # travel around 1e-6 bohr over the trajectory. A force error large enough
    # to matter physically therefore shows up far above these thresholds; the
    # superseded one-sided two-electron gradient, for instance, produced a net
    # force of order 1 Hartree/bohr.
    #
    # If one of these fails only marginally, the staged assertions below say
    # whether the disagreement sits in the initial SCF state (a summation-order
    # difference, benign) or in the trajectory alone (a genuine gradient
    # discrepancy, which the per-term gradient tests also catch).
    md_energy_atol = 1e-8
    md_coordinate_atol = 1e-9
    md_residual_atol = 5e-8

    ### Convergence tolerance for the low-rank Jacobian kernel.
    # See run_shadow_md for why this is not tightened further.
    low_rank_tol = 1e-4

    def setUp(self):
        import cupy
        from gpu4pyscf import dft as gpu_dft
        from gpu4pyscf import scf as gpu_scf

        self.cupy = cupy
        self.gpu_dft = gpu_dft
        self.gpu_scf = gpu_scf
        self.mol = gto.M(
            atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0
        )
        self.log = logger.getLogger("pysmd")
        self._log_disabled = self.log.disabled
        self.log.disabled = True

    def tearDown(self):
        self.log.disabled = self._log_disabled
        linalg_helper.set_linalg_backend("numpy")

    def make_water(self, coords=None):
        """Larger asymmetric molecule for the full gradient comparisons.

        Returns a fresh object on every call. That matters for any test that
        runs a trajectory: the MD kernel displaces the nuclei through
        ``Mole.set_geom_(inplace=True)``, so a shared molecule would leave the
        second run starting from the first run's final geometry.
        """
        if coords is None:
            coords = np.array(
                [[0.0, 0.0, 0.12], [0.0, 1.43, -0.99], [0.0, -1.43, -0.99]]
            )
        atom = "; ".join(
            f"{symbol} {xyz[0]} {xyz[1]} {xyz[2]}"
            for symbol, xyz in zip(("O", "H", "H"), coords)
        )
        return gto.M(
            atom=atom,
            basis="sto-3g",
            unit="bohr",
            verbose=0,
        )

    def make_diatomic(self):
        """Smallest heteronuclear molecule used by these tests.

        LiH rather than H2: a homonuclear diatomic makes the spurious net force
        of an incorrect two-electron gradient cancel by symmetry, so such a
        molecule cannot detect that class of error. See :meth:`make_water` on
        why a fresh object is returned each call.
        """
        return gto.M(
            atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0
        )

    def make_unconverged_mf(self, on_gpu, xc=None, molecule="water"):
        """Build a deliberately unconverged mean-field on the chosen backend.

        Leaving the mean field unconverged forces ``verify_scf_convergence`` to
        run the Newton-Raphson SCF, so an end-to-end comparison covers the SCF
        path as well as the MD kernel.
        """
        mol = self.make_water() if molecule == "water" else self.make_diatomic()

        if xc is None:
            mf = self.gpu_scf.RHF(mol) if on_gpu else scf.RHF(mol)
        else:
            mf = (
                self.gpu_dft.RKS(mol, xc=xc) if on_gpu
                else dft.RKS(mol, xc=xc)
            )
            mf.grids.level = 3
            mf.grids.prune = None

        mf.verbose = 0
        mf.max_cycle = 0
        mf.kernel()
        return mf

    def make_interfaces(self, xc=None, mol=None):
        """Return a matched (gpu_interface, cpu_interface) pair."""
        if mol is None:
            mol = self.mol

        if xc is None:
            gpu_mf = self.gpu_scf.RHF(mol)
            cpu_mf = scf.RHF(mol)
        else:
            gpu_mf = self.gpu_dft.RKS(mol, xc=xc)
            cpu_mf = dft.RKS(mol, xc=xc)
            for mean_field in (gpu_mf, cpu_mf):
                mean_field.grids.level = 3
                mean_field.grids.prune = None

        cpu_interface = PyscfDriver(pyscf_mf=cpu_mf)
        gpu_interface = PyscfDriver(pyscf_mf=gpu_mf, backend="gpu")
        return gpu_interface, cpu_interface

    def assert_matches(self, gpu_value, cpu_value):
        self.assertIsInstance(gpu_value, self.cupy.ndarray)
        np.testing.assert_allclose(
            self.cupy.asnumpy(gpu_value),
            self.cupy.asnumpy(cpu_value)
            if isinstance(cpu_value, self.cupy.ndarray)
            else np.asarray(cpu_value),
            atol=self.comparison_atol,
        )

    ### Backend wiring -----------------------------------------------------

    def test_gpu_factory_dispatch_and_backend_flags(self):
        rhf = PyscfDriver(pyscf_mf=self.gpu_scf.RHF(self.mol), backend="gpu")
        rks = PyscfDriver(pyscf_mf=self.gpu_dft.RKS(self.mol), backend="auto")
        molecule_rhf = PyscfDriver(pyscf_mol=self.mol, backend="gpu")
        molecule_rks = PyscfDriver(
            pyscf_mol=self.mol, backend="gpu", xc="pbe"
        )

        self.assertIsInstance(rhf, GPU4RHF)
        self.assertIsInstance(rks, GPU4RKS)
        self.assertIsInstance(molecule_rhf, GPU4RHF)
        self.assertIsInstance(molecule_rks, GPU4RKS)

        # Both the electronic structure and the gradients are device-resident.
        self.assertEqual(rhf.electronic_backend, "gpu")
        self.assertEqual(rhf.gradient_backend, "gpu")
        self.assertEqual(rhf.array_backend, "cupy")

        with self.assertRaisesRegex(TypeError, "expects CPU mean-field objects"):
            PyscfDriver(pyscf_mf=self.gpu_scf.RHF(self.mol), backend="cpu")

    def test_no_cpu_delegate_remains(self):
        """The mixed CPU/GPU fallback has been removed."""
        gpu, _ = self.make_interfaces()
        self.assertFalse(hasattr(gpu, "_cpu_delegate"))

    def test_grid_response_forced_for_rks(self):
        """A fixed-grid force is non-conservative and assumes an idempotent
        density, which a propagated shadow density is not."""
        gpu, _ = self.make_interfaces(xc="pbe")
        self.assertTrue(gpu.grad.grid_response)

    def test_rejected_type_leaves_array_backend_unchanged(self):
        open_shell = gto.M(
            atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0
        )
        linalg_helper.set_linalg_backend("numpy")
        with self.assertRaisesRegex(TypeError, "GPU4PySCF RHF"):
            GPU4RHF(pyscf_mf=self.gpu_scf.ROHF(open_shell))
        self.assertEqual(linalg_helper.get_linalg_backend_name(), "numpy")

    def test_rejected_functional_leaves_array_backend_unchanged(self):
        """GPU4RKS validates before switching the process-wide backend."""
        linalg_helper.set_linalg_backend("numpy")
        with self.assertRaises(NotImplementedError):
            GPU4RKS(pyscf_mol=self.mol, xc="pbe0")
        self.assertEqual(linalg_helper.get_linalg_backend_name(), "numpy")

    def test_restore_array_backend(self):
        linalg_helper.set_linalg_backend("numpy")
        gpu, _ = self.make_interfaces()
        self.assertEqual(linalg_helper.get_linalg_backend_name(), "cupy")
        gpu.restore_array_backend()
        self.assertEqual(linalg_helper.get_linalg_backend_name(), "numpy")

    def test_derivative_matrix_hooks_raise(self):
        """GPU4PySCF exposes no derivative matrices; the hooks must say so."""
        gpu, _ = self.make_interfaces()
        dm = gpu.compute_guess_density_matrix()
        for call in (
            lambda: gpu.compute_1e_gradient(),
            lambda: gpu.compute_coulomb_gradient(dm),
            lambda: gpu.compute_exchange_gradient(dm),
            lambda: gpu.compute_xc_gradient(dm, dm),
        ):
            with self.assertRaises(NotImplementedError):
                call()

    ### Electronic-structure matrices --------------------------------------

    def test_rhf_matrices_match_cpu(self):
        gpu, cpu = self.make_interfaces()
        dm = gpu.compute_guess_density_matrix()
        dm_numpy = self.cupy.asnumpy(dm)

        for gpu_value, cpu_value in (
            (gpu.compute_overlap_matrix(), cpu.compute_overlap_matrix()),
            (
                gpu.compute_core_hamiltonian_matrix(),
                cpu.compute_core_hamiltonian_matrix(),
            ),
            (gpu.compute_coulomb_matrix(dm), cpu.compute_coulomb_matrix(dm_numpy)),
            (
                gpu.compute_exchange_matrix(dm),
                cpu.compute_exchange_matrix(dm_numpy),
            ),
            (
                gpu.compute_eff_potential_matrix(dm),
                cpu.compute_eff_potential_matrix(dm_numpy),
            ),
            (gpu.compute_nuclear_gradient(), cpu.compute_nuclear_gradient()),
        ):
            self.assert_matches(gpu_value, self.cupy.asnumpy(cpu_value))

    def test_rks_xc_terms_match_cpu(self):
        for xc in ("lda,vwn5", "pbe"):
            with self.subTest(xc=xc):
                gpu, cpu = self.make_interfaces(xc=xc)
                dm = gpu.compute_guess_density_matrix()
                dm_numpy = self.cupy.asnumpy(dm)

                gpu_exc, gpu_vxc = gpu.compute_xc_terms(dm)
                cpu_exc, cpu_vxc = cpu.compute_xc_terms(dm_numpy)
                np.testing.assert_allclose(
                    float(gpu_exc), float(cpu_exc), atol=self.comparison_atol
                )
                self.assert_matches(gpu_vxc, self.cupy.asnumpy(cpu_vxc))

    def test_rks_exchange_is_zero(self):
        """Pure functionals carry no exact exchange on any backend."""
        gpu, _ = self.make_interfaces(xc="pbe")
        dm = gpu.compute_guess_density_matrix()
        vk = gpu.compute_exchange_matrix(dm)
        np.testing.assert_allclose(
            self.cupy.asnumpy(vk), np.zeros_like(self.cupy.asnumpy(vk))
        )

    def test_geometry_update_resets_cached_integrals(self):
        gpu, _ = self.make_interfaces(xc="pbe")
        old_optimizer = gpu.mf._opt_gpu
        displacement = self.cupy.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]])
        gpu.get_updated_atomic_coords(gpu.get_atomic_coords() + displacement)
        self.assertIsNot(old_optimizer, gpu.mf._opt_gpu)

    ### Shadow gradients ---------------------------------------------------

    def build_shadow_state(self, interface, dm_prop=None, mixing=0.5):
        """Return a genuinely non-self-consistent shadow state.

        Mirrors the CPU test helper: the propagated density is displaced from
        the converged ground state so that ``dm_prop != dm_ao``, which is the
        regime where the two-electron symmetrization matters.
        """
        backend = linalg_helper.get_linalg_backend()

        ovlp, inv_ovlp, _, inv_sqrt_ovlp = interface.compute_overlap_matrices()
        h1e = interface.compute_core_hamiltonian_matrix()

        if dm_prop is None:
            converged = interface.compute_guess_density_matrix()
            guess = interface.compute_guess_density_matrix()
            dm_prop = mixing * converged + (1.0 - mixing) * guess

        fock = h1e + interface.compute_eff_potential_matrix(dm=dm_prop)
        fock_orth = backend.matmul(
            backend.matmul(inv_sqrt_ovlp.T.conj(), fock), inv_sqrt_ovlp
        )
        _, mo_coeff = backend.linalg.eigh(fock_orth)

        n_occ = interface.get_num_electrons() // 2
        mo_occ = backend.zeros(interface.get_num_basis_functions())
        mo_occ[:n_occ] = 1.0
        dm_orth = backend.matmul(mo_coeff * mo_occ, mo_coeff.T.conj())
        dm_ao = backend.matmul(
            backend.matmul(inv_sqrt_ovlp, 2.0 * dm_orth),
            inv_sqrt_ovlp.T.conj(),
        )
        dm_lin = (2.0 * dm_ao) - dm_prop

        return fock, dm_ao, dm_prop, dm_lin

    def shadow_potential_energy(self, interface, fock, dm_ao, dm_prop, dm_lin):
        """Evaluate the implemented shadow potential on the active backend."""
        backend = linalg_helper.get_linalg_backend()
        vj = interface.compute_coulomb_matrix(dm=dm_prop)
        vk = interface.compute_exchange_matrix(dm=dm_prop)
        exc, vxc = interface.compute_xc_terms(dm=dm_prop)
        e_elec = backend.einsum(
            "ij,ji->", interface.compute_core_hamiltonian_matrix(), dm_ao
        )
        e_elec += 0.5 * backend.einsum("ij,ji->", vj, dm_lin)
        e_elec -= 0.25 * backend.einsum("ij,ji->", vk, dm_lin)
        e_elec += exc + backend.einsum(
            "ij,ji->", vxc, dm_ao - dm_prop
        )
        return float(backend.to_numpy(e_elec + interface.get_nuclear_energy()))

    def shadow_gradient_and_energy(self, interface, dm_prop):
        fock, dm_ao, dm_prop, dm_lin = self.build_shadow_state(
            interface, dm_prop=dm_prop
        )
        gradient = (
            interface.compute_one_electron_gradient(dm_ao=dm_ao, fock=fock)
            + interface.compute_two_electron_gradient(
                dm_bra=dm_prop, dm_ket=dm_lin
            )
            + interface.compute_xc_gradient_per_atom(
                dm_prop=dm_prop, dm_ao=dm_ao
            )
            + interface.compute_nuclear_gradient()
        )
        energy = self.shadow_potential_energy(
            interface, fock, dm_ao, dm_prop, dm_lin
        )
        return gradient, energy

    def finite_difference_shadow_gradient(self, on_gpu, xc=None):
        """Differentiate the shadow energy with a frozen propagated density."""
        if not on_gpu:
            linalg_helper.set_linalg_backend("numpy")
        coords = np.array(
            [[0.0, 0.0, 0.12], [0.0, 1.43, -0.99], [0.0, -1.43, -0.99]]
        )

        def make_interface(displaced_coords):
            mol = self.make_water(displaced_coords)
            if xc is None:
                mf = self.gpu_scf.RHF(mol) if on_gpu else scf.RHF(mol)
            else:
                mf = (
                    self.gpu_dft.RKS(mol, xc=xc)
                    if on_gpu else dft.RKS(mol, xc=xc)
                )
                mf.grids.level = 3
                mf.grids.prune = None
            return PyscfDriver(pyscf_mf=mf, backend="gpu" if on_gpu else "cpu")

        base = make_interface(coords)
        dm_prop = base.compute_guess_density_matrix()
        gradient, _ = self.shadow_gradient_and_energy(base, dm_prop)
        fd_gradient = np.zeros((coords.shape[0], 3))
        step = 1e-4
        for atom in range(coords.shape[0]):
            for direction in range(3):
                plus = coords.copy()
                minus = coords.copy()
                plus[atom, direction] += step
                minus[atom, direction] -= step
                plus_interface = make_interface(plus)
                minus_interface = make_interface(minus)
                displaced_dm_prop = (
                    dm_prop
                    if on_gpu
                    else np.asarray(dm_prop)
                )
                plus_state = self.build_shadow_state(
                    plus_interface, dm_prop=displaced_dm_prop
                )
                minus_state = self.build_shadow_state(
                    minus_interface, dm_prop=displaced_dm_prop
                )
                e_plus = self.shadow_potential_energy(
                    plus_interface, *plus_state
                )
                e_minus = self.shadow_potential_energy(
                    minus_interface, *minus_state
                )
                fd_gradient[atom, direction] = (e_plus - e_minus) / (2.0 * step)
        return gradient, fd_gradient

    def assert_gradient_terms_match_cpu(self, xc=None):
        """Calibration at self-consistency.

        With a single density everywhere, every factor and sign convention of
        the per-atom GPU4PySCF API must reproduce the validated CPU result.
        This is the test that pins down j_factor/k_factor and the
        ``_hcore_energy`` conventions.
        """
        mol = self.make_water()
        gpu, cpu = self.make_interfaces(xc=xc, mol=mol)

        dm_gpu = gpu.compute_guess_density_matrix()
        dm_cpu = self.cupy.asnumpy(dm_gpu)
        fock_gpu = (
            gpu.compute_core_hamiltonian_matrix()
            + gpu.compute_eff_potential_matrix(dm=dm_gpu)
        )
        fock_cpu = self.cupy.asnumpy(fock_gpu)

        self.assert_matches(
            gpu.compute_one_electron_gradient(dm_ao=dm_gpu, fock=fock_gpu),
            cpu.compute_one_electron_gradient(dm_ao=dm_cpu, fock=fock_cpu),
        )
        self.assert_matches(
            gpu.compute_two_electron_gradient(dm_bra=dm_gpu, dm_ket=dm_gpu),
            cpu.compute_two_electron_gradient(dm_bra=dm_cpu, dm_ket=dm_cpu),
        )
        self.assert_matches(
            gpu.compute_xc_gradient_per_atom(dm_prop=dm_gpu, dm_ao=dm_gpu),
            cpu.compute_xc_gradient_per_atom(dm_prop=dm_cpu, dm_ao=dm_cpu),
        )

    def test_rhf_gradient_terms_match_cpu(self):
        self.assert_gradient_terms_match_cpu()

    def test_rks_lda_gradient_terms_match_cpu(self):
        self.assert_gradient_terms_match_cpu(xc="lda,vwn5")

    def test_rks_pbe_gradient_terms_match_cpu(self):
        self.assert_gradient_terms_match_cpu(xc="pbe")

    def assert_shadow_gradient_matches_cpu(self, xc=None):
        """The decisive test: agreement away from self-consistency.

        Exercises the polarization identity for the two-electron term and the
        linearized exchange-correlation response, neither of which reduces to
        the self-consistent case.
        """
        mol = self.make_water()
        gpu, cpu = self.make_interfaces(xc=xc, mol=mol)

        fock, dm_ao, dm_prop, dm_lin = self.build_shadow_state(gpu)
        self.assertGreater(
            float(self.cupy.linalg.norm(dm_ao - dm_prop)), 1e-6
        )

        gpu_grad = (
            gpu.compute_one_electron_gradient(dm_ao=dm_ao, fock=fock)
            + gpu.compute_two_electron_gradient(dm_bra=dm_prop, dm_ket=dm_lin)
            + gpu.compute_xc_gradient_per_atom(dm_prop=dm_prop, dm_ao=dm_ao)
            + gpu.compute_nuclear_gradient()
        )

        to_numpy = self.cupy.asnumpy
        cpu_grad = (
            cpu.compute_one_electron_gradient(
                dm_ao=to_numpy(dm_ao), fock=to_numpy(fock)
            )
            + cpu.compute_two_electron_gradient(
                dm_bra=to_numpy(dm_prop), dm_ket=to_numpy(dm_lin)
            )
            + cpu.compute_xc_gradient_per_atom(
                dm_prop=to_numpy(dm_prop), dm_ao=to_numpy(dm_ao)
            )
            + cpu.compute_nuclear_gradient()
        )

        np.testing.assert_allclose(
            to_numpy(gpu_grad),
            to_numpy(cpu_grad)
            if isinstance(cpu_grad, self.cupy.ndarray)
            else np.asarray(cpu_grad),
            atol=self.comparison_atol,
        )

        # The exact gradient of the shadow potential has no net force, because
        # the potential is built from translationally invariant integrals
        # contracted with matrices held fixed with respect to the nuclei.
        np.testing.assert_allclose(
            to_numpy(gpu_grad).sum(axis=0),
            np.zeros(3),
            atol=self.translation_atol,
        )

    def test_rhf_shadow_gradient_matches_cpu(self):
        self.assert_shadow_gradient_matches_cpu()

    def test_rhf_shadow_gradient_matches_finite_difference_on_gpu(self):
        analytical, finite_difference = self.finite_difference_shadow_gradient(
            on_gpu=True
        )
        np.testing.assert_allclose(
            self.cupy.asnumpy(analytical),
            finite_difference,
            atol=5e-6,
        )

    def test_rhf_shadow_gradient_matches_finite_difference_on_cpu(self):
        analytical, finite_difference = self.finite_difference_shadow_gradient(
            on_gpu=False
        )
        np.testing.assert_allclose(analytical, finite_difference, atol=5e-6)

    def test_rks_lda_shadow_gradient_matches_cpu(self):
        self.assert_shadow_gradient_matches_cpu(xc="lda,vwn5")

    def test_rks_pbe_shadow_gradient_matches_cpu(self):
        self.assert_shadow_gradient_matches_cpu(xc="pbe")

    def test_rks_pbe_shadow_gradient_matches_finite_difference_on_gpu(self):
        analytical, finite_difference = self.finite_difference_shadow_gradient(
            on_gpu=True, xc="pbe"
        )
        np.testing.assert_allclose(
            self.cupy.asnumpy(analytical),
            finite_difference,
            atol=5e-6,
        )

    def test_polarization_identity_reduces_at_self_consistency(self):
        """Three-evaluation identity must collapse to one when densities match.

        Guards the shortcut branch against the general path.
        """
        gpu, _ = self.make_interfaces(xc="pbe", mol=self.make_water())
        dm = gpu.compute_guess_density_matrix()

        shortcut = gpu.compute_two_electron_gradient(dm_bra=dm, dm_ket=dm)
        # A distinct object with identical contents takes the general path.
        general = gpu.compute_two_electron_gradient(
            dm_bra=dm, dm_ket=self.cupy.array(dm, copy=True)
        )
        np.testing.assert_allclose(
            self.cupy.asnumpy(shortcut),
            self.cupy.asnumpy(general),
            atol=self.comparison_atol,
        )

    ### End-to-end ---------------------------------------------------------

    def run_shadow_md(self, interface, num_tsteps=5):
        """Run a short shadow MD trajectory from an unconverged mean field.

        The mean-field object is deliberately left unconverged so that
        ``verify_scf_convergence`` runs the Newton-Raphson SCF. That way the
        comparison covers the SCF path as well as the MD kernel.

        Notes
        -----
        ``rel_res_error_tol`` is left at a value the low-rank Jacobian kernel
        can actually reach within ``max_rank`` for every method tested here.
        Exceeding it is not fatal, since the kernel warns and continues with
        its best available approximation, but a run that silently fell back to
        an inexact kernel would be comparing the wrong thing.

        A comfortable margin also keeps the comparison meaningful: the rank at
        which the expansion terminates is a discrete decision, so a tolerance
        sitting right at the achievable limit could be met at different ranks
        on the two backends and produce trajectories that differ by far more
        than the numerical noise being tested for.
        """
        md = shadow_md.ShadowMD(qm_interface=interface)
        md.md_timestep = 0.05
        md.md_num_tsteps = num_tsteps
        md.max_rank = interface.get_num_basis_functions()
        md.rel_res_error_tol = self.low_rank_tol
        md.scf_max_rank = interface.get_num_basis_functions()
        md.scf_rel_res_error_tol = self.low_rank_tol
        md.scf_energy_diff_error_tol = 1e-10
        md.scf_res_norm_error_tol = 1e-10
        md.kernel()
        return md

    def assert_shadow_md_matches_cpu(self, xc=None, num_tsteps=5):
        """Run the identical calculation on both backends and compare.

        Uses an asymmetric molecule and several timesteps. Both matter: a
        symmetric molecule hides two-electron gradient errors because the
        spurious net force cancels, and a single timestep never exercises the
        accumulated trajectory.

        Assertions are staged so that a failure localizes itself. If the
        initial values disagree, the Newton-Raphson SCF is at fault; if only
        the trajectory disagrees, the gradient is.
        """
        # The CPU run must come first: constructing a GPU interface switches
        # the process-wide array backend to CuPy, which would otherwise make
        # the CPU run allocate device arrays. Each run gets its own molecule,
        # since the kernel displaces the nuclei in place.
        linalg_helper.set_linalg_backend("numpy")
        cpu_md = self.run_shadow_md(
            PyscfDriver(pyscf_mf=self.make_unconverged_mf(False, xc=xc)),
            num_tsteps=num_tsteps,
        )
        gpu_md = self.run_shadow_md(
            PyscfDriver(
                pyscf_mf=self.make_unconverged_mf(True, xc=xc), backend="gpu"
            ),
            num_tsteps=num_tsteps,
        )

        to_numpy = self.cupy.asnumpy
        cpu_data = cpu_md.sim_data
        gpu_data = gpu_md.sim_data

        ### Stage 1: the converged SCF starting point
        self.assertAlmostEqual(
            float(cpu_data.initial_energy_tot),
            float(gpu_data.initial_energy_tot),
            delta=self.md_energy_atol,
            msg="Initial SCF energies differ; the Newton-Raphson SCF, not the "
                "gradient, is the likely cause.",
        )
        self.assertAlmostEqual(
            float(cpu_data.initial_energy_free),
            float(gpu_data.initial_energy_free),
            delta=self.md_energy_atol,
        )
        np.testing.assert_allclose(
            np.asarray(cpu_data.initial_atomic_coords),
            to_numpy(gpu_data.initial_atomic_coords),
            atol=self.md_coordinate_atol,
        )

        ### Stage 2: per-timestep energies
        for name, atol in (
            ("energy_tot", self.md_energy_atol),
            ("energy_free", self.md_energy_atol),
            ("energy_kin", self.md_energy_atol),
        ):
            with self.subTest(quantity=name):
                np.testing.assert_allclose(
                    np.asarray(getattr(cpu_data, name)),
                    to_numpy(getattr(gpu_data, name)),
                    atol=atol,
                    err_msg=f"{name} diverges between backends",
                )

        ### Stage 3: the trajectory itself
        # Coordinates integrate the forces, so this is the most sensitive
        # check of gradient agreement.
        np.testing.assert_allclose(
            np.asarray(cpu_data.atomic_coords),
            to_numpy(gpu_data.atomic_coords),
            atol=self.md_coordinate_atol,
            err_msg="Nuclear trajectories diverge between backends",
        )

        ### Stage 4: the electronic degrees of freedom
        # The residual norm ||D[X]S - X|| gauges the shadow dynamics, so
        # agreement here means the propagated density matched at every step.
        np.testing.assert_allclose(
            np.asarray(cpu_data.residual_norm),
            to_numpy(gpu_data.residual_norm),
            atol=self.md_residual_atol,
            err_msg="Shadow residual norms diverge between backends",
        )

    def test_rhf_shadow_md_matches_cpu(self):
        self.assert_shadow_md_matches_cpu()

    def test_rks_lda_shadow_md_matches_cpu(self):
        self.assert_shadow_md_matches_cpu(xc="lda,vwn5")

    def test_rks_pbe_shadow_md_matches_cpu(self):
        self.assert_shadow_md_matches_cpu(xc="pbe")

    def test_rhf_shadow_md_matches_cpu_on_diatomic(self):
        """Smallest end-to-end case, useful for first-run debugging.

        Retained as a fast smoke test on top of the water comparisons above.
        LiH is heteronuclear, so unlike H2 it does not mask two-electron
        gradient errors, but the water cases remain the stronger check.
        """
        linalg_helper.set_linalg_backend("numpy")
        cpu_md = self.run_shadow_md(
            PyscfDriver(
                pyscf_mf=self.make_unconverged_mf(False, molecule="diatomic")
            ),
            num_tsteps=1,
        )
        gpu_md = self.run_shadow_md(
            PyscfDriver(
                pyscf_mf=self.make_unconverged_mf(True, molecule="diatomic"),
                backend="gpu",
            ),
            num_tsteps=1,
        )

        np.testing.assert_allclose(
            np.asarray(cpu_md.sim_data.energy_tot),
            self.cupy.asnumpy(gpu_md.sim_data.energy_tot),
            atol=self.md_energy_atol,
        )


if __name__ == "__main__":
    unittest.main()
