"""ShadowMD gradient regression tests."""

import unittest

import numpy as np
from pyscf import dft, gto, scf

from pysmd.closed_shell.shadow_md import ShadowMD
from pysmd.common import logger
from pysmd.interface.pyscf import PyscfDriver
from pysmd.lib import linalg_helper


class TestShadowMDGradients(unittest.TestCase):

    gradient_atol = 1e-9
    xc_fd_atol = 1e-7
    shadow_fd_atol = 1e-6
    translation_atol = 1e-9

    def make_mol(self):
        mol = gto.Mole()
        mol.atom = "Li 0 0 0; H 0 0 3.0"
        mol.basis = "sto-3g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    def make_water_mol(self, coords):
        mol = gto.Mole()
        mol.atom = (
            f"O {coords[0, 0]} {coords[0, 1]} {coords[0, 2]}; "
            f"H {coords[1, 0]} {coords[1, 1]} {coords[1, 2]}; "
            f"H {coords[2, 0]} {coords[2, 1]} {coords[2, 2]}"
        )
        mol.basis = "sto-3g"
        mol.unit = "bohr"
        mol.verbose = 0
        mol.build()
        return mol

    def assert_gradient_matches_pyscf(self, mf):
        # Converge tightly. The comparison is between two gradient assemblies
        # evaluated at the same density, so any residual SCF error enters both
        # sides differently and would otherwise set the achievable agreement
        # (around 1e-8 at PySCF's default conv_tol).
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-10
        mf.kernel()
        self.assertTrue(mf.converged)

        dm = mf.make_rdm1()
        fock = mf.get_hcore() + mf.get_veff(mf.mol, dm)

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            qm_interface = PyscfDriver(pyscf_mf=mf)
            dm = qm_interface.compute_density_matrix()
            md = ShadowMD(qm_interface=qm_interface)
            test_grad = md.update_gradients(
                fock=fock,
                dm_ao=dm,
                dm_prop=dm,
                dm_lin=dm,
            )
        finally:
            log.disabled = log_disabled

        grad = mf.nuc_grad_method()
        if isinstance(mf, dft.rks.RKS):
            grad.grid_response = True
        ref_grad = grad.kernel()

        np.testing.assert_allclose(test_grad, ref_grad,
                                   atol=self.gradient_atol)

    def assert_rks_gradient_matches_pyscf(self, xc):
        self.assert_gradient_matches_pyscf(dft.RKS(self.make_mol(), xc=xc))

    def make_rks_interface(self, xc, coords):
        mf = dft.RKS(self.make_water_mol(coords), xc=xc)
        mf.grids.level = 3
        mf.grids.prune = None
        mf.verbose = 0
        return PyscfDriver(pyscf_mf=mf)

    def linearized_xc_energy(self, xc, coords, dm_prop, dm_ao):
        qm_interface = self.make_rks_interface(xc, coords)
        e_xc, vxc = qm_interface.compute_xc_terms(dm=dm_prop)
        return e_xc + 2.0 * np.einsum(
            "ij,ji->", vxc, dm_ao - dm_prop, optimize=True
        )

    def linearized_xc_gradient(
        self,
        xc,
        coords,
        dm_prop,
        dm_ao,
    ):
        qm_interface = self.make_rks_interface(xc, coords)
        xc_grad, grid_grad = qm_interface.compute_xc_gradient(
            dm_prop=dm_prop,
            dm_ao=dm_ao,
        )
        aoslices = qm_interface.compute_ao_slices_by_atom()

        la = linalg_helper.get_linalg_backend()
        total_grad = la.copy(grid_grad)
        for k in range(qm_interface.get_num_atoms()):
            p0, p1 = aoslices[k, 2:]
            total_grad[k] += 4.0 * np.einsum(
                "xij,ij->x",
                xc_grad[:, p0:p1],
                dm_ao[p0:p1],
                optimize=True,
            )

        return total_grad

    def assert_linearized_xc_gradient_matches_finite_difference(self, xc):
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.4, 0.2],
            [1.2, 0.0, -0.1],
        ])
        mf = dft.RKS(self.make_water_mol(coords), xc=xc)
        mf.grids.level = 3
        mf.grids.prune = None
        mf.kernel()
        self.assertTrue(mf.converged)
        interface = PyscfDriver(pyscf_mf=mf)
        dm_ao = interface.compute_density_matrix()
        dm_prop = 0.50 * dm_ao + 0.50 * interface.compute_guess_density_matrix()
        self.assertGreater(np.linalg.norm(dm_ao - dm_prop), 1e-6)

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            test_grad = self.linearized_xc_gradient(
                xc,
                coords,
                dm_prop,
                dm_ao,
            )

            fd_grad = np.zeros_like(test_grad)
            step = 1e-4
            for ia in range(coords.shape[0]):
                for ix in range(coords.shape[1]):
                    coords_plus = coords.copy()
                    coords_minus = coords.copy()
                    coords_plus[ia, ix] += step
                    coords_minus[ia, ix] -= step
                    e_plus = self.linearized_xc_energy(
                        xc, coords_plus, dm_prop, dm_ao
                    )
                    e_minus = self.linearized_xc_energy(
                        xc, coords_minus, dm_prop, dm_ao
                    )
                    fd_grad[ia, ix] = (e_plus - e_minus) / (2.0 * step)
        finally:
            log.disabled = log_disabled

        np.testing.assert_allclose(test_grad, fd_grad, atol=self.xc_fd_atol)

    ### Shadow-potential gradient tests -----------------------------------
    # These exercise update_gradients away from self-consistency, where the
    # propagated (dm_prop) and linearized (dm_lin) density matrices differ.
    # That is the regime the shadow dynamics actually operates in, and it is
    # where the two-electron symmetrization of Eq. 43 matters.

    def make_shadow_mf(self, coords, xc=None):
        """Build a mean-field object for water at the given geometry."""
        mol = self.make_water_mol(coords)
        if xc is None:
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol, xc=xc)
            mf.grids.level = 3
            mf.grids.prune = None
        mf.verbose = 0
        return mf

    def build_shadow_state(self, coords, dm_prop, xc=None):
        r"""Return the SCF-free shadow state at a geometry for frozen ``dm_prop``.

        Performs the single non-iterative step of Eqs. 18-21 of Niklasson,
        *J. Chem. Theory Comput.* **2020**, *16*, 3628: the Fock matrix is
        built from the propagated density :math:`\mathbf{P}` and
        :math:`\mathbf{D}[\mathbf{P}]` follows from a single diagonalization.

        The propagated density is held fixed as the nuclei move, matching
        Eq. 43, which defines the two-electron derivative as
        :math:`\mathbf{G}_{R_I} = \partial\mathbf{G}(\mathbf{R},\mathbf{P})/
        \partial R_I|_{\mathbf{P}}`. See
        :meth:`assert_shadow_gradient_matches_finite_difference` for why this
        matters.

        Returns
        -------
        tuple
            ``(md, fock, dm_ao, dm_prop, dm_lin)``
        """
        qm_interface = PyscfDriver(pyscf_mf=self.make_shadow_mf(coords, xc=xc))
        md = ShadowMD(qm_interface=qm_interface)
        md.temp = 0.0

        md.S, md.Sm1, md.Sp12, md.Sm12 = qm_interface.compute_overlap_matrices()
        md.h1e = qm_interface.compute_core_hamiltonian_matrix()

        fock = md.h1e + qm_interface.compute_eff_potential_matrix(dm=dm_prop)
        fock_orth = np.matmul(
            np.matmul(md.Sm12.T.conj(), fock), md.Sm12
        )
        mo_energy, mo_coeff = np.linalg.eigh(fock_orth)

        # Zero electronic temperature: integer occupations.
        mo_occ = np.zeros(md.n_mo)
        mo_occ[:md.n_occ] = 1.0
        dm_orth = np.matmul(mo_coeff * mo_occ, mo_coeff.T.conj())
        dm_ao = np.matmul(
            np.matmul(md.Sm12, 2.0 * dm_orth), md.Sm12.T.conj()
        )
        dm_lin = (2.0 * dm_ao) - dm_prop

        md.mo_energy, md.mo_coeff, md.mo_occ = mo_energy, mo_coeff, mo_occ
        md.fock = fock
        md.dm_ao, md.dm_prop, md.dm_lin = dm_ao, dm_prop, dm_lin
        md.dynvar_X = np.matmul(0.50 * dm_ao, md.S)

        return md, fock, dm_ao, dm_prop, dm_lin

    def make_frozen_dm_prop(self, coords, xc=None):
        r"""Build a frozen :math:`\mathbf{P}` giving a genuine shadow residual.

        The propagated density is deliberately displaced from the converged
        ground state by mixing in the superposition-of-atomic-densities guess,
        so that ``dm_prop != dm_ao`` once the shadow state is rebuilt. That
        non-self-consistency is what makes these tests sensitive to the
        two-electron symmetrization.
        """
        mf = self.make_shadow_mf(coords, xc=xc)
        mf.kernel()
        self.assertTrue(mf.converged)
        interface = PyscfDriver(pyscf_mf=mf)
        return 0.50 * interface.compute_density_matrix() + 0.50 * interface.compute_guess_density_matrix()

    def assert_shadow_gradient_translationally_invariant(self, xc=None):
        r"""The exact shadow gradient must have zero net force.

        The shadow potential of Eq. 14 is built from translationally
        invariant integrals contracted with matrices held fixed with respect
        to the nuclear positions, so its exact gradient satisfies
        :math:`\sum_I \partial\mathcal{U}/\partial R_I = 0`. Every term except
        the two-electron one is individually translation invariant, so a
        nonzero net force isolates the one-sided two-electron contraction.

        This is far cheaper than a finite-difference comparison and needs no
        reference implementation, which makes it the primary guard.
        """
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.4, 0.2],
            [1.2, 0.0, -0.1],
        ])

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            dm_prop = self.make_frozen_dm_prop(coords, xc=xc)
            md, fock, dm_ao, dm_prop, dm_lin = self.build_shadow_state(
                coords, dm_prop, xc=xc
            )

            # The shadow state must be genuinely non-self-consistent,
            # otherwise the test degenerates into the symmetric case.
            self.assertGreater(np.linalg.norm(dm_ao - dm_prop), 1e-6)

            grad = md.update_gradients(
                fock=fock, dm_ao=dm_ao, dm_prop=dm_prop, dm_lin=dm_lin
            )
            # Control: at self-consistency both forms must agree exactly and
            # both must be translationally invariant.
            grad_sc = md.update_gradients(
                fock=fock, dm_ao=dm_ao, dm_prop=dm_ao, dm_lin=dm_ao
            )
        finally:
            log.disabled = log_disabled

        np.testing.assert_allclose(
            grad.sum(axis=0), np.zeros(3), atol=self.translation_atol
        )

        np.testing.assert_allclose(
            grad_sc.sum(axis=0), np.zeros(3), atol=self.translation_atol
        )

    def shadow_potential_energy(self, coords, dm_prop, xc=None):
        r"""Return :math:`\mathcal{U}(\mathbf{R}, \mathbf{P})` less kinetic energy.

        This is the shadow potential of Eq. 14 evaluated at zero electronic
        temperature, so the entropy term vanishes and ``e_free == e_elec``.
        """
        md, _, dm_ao, dm_prop, dm_lin = self.build_shadow_state(
            coords, dm_prop, xc=xc
        )
        md.update_energies(
            dm_ao=dm_ao,
            dm_prop=dm_prop,
            dm_lin=dm_lin,
            compute_e_nuc=True,
            compute_e_free=True,
        )
        return md.e_nuc + md.e_free

    def assert_shadow_gradient_matches_finite_difference(self, xc=None):
        r"""Compare the full shadow gradient against finite differences.

        The propagated density :math:`\mathbf{P}` is held fixed as the nuclei
        move, while :math:`\mathbf{D}[\mathbf{P}]` is re-solved at every
        displaced geometry. The finite-difference reference is therefore
        :math:`\mathcal{U}(\mathbf{R}, \mathbf{P})`, whose exact gradient is
        the complete ``update_gradients`` result including the Pulay term.

        Fixing :math:`\mathbf{P}` rather than :math:`\mathbf{X} = \mathbf{PS}`
        is deliberate: Eq. 43 defines the two-electron derivative with
        :math:`\mathbf{P}` held fixed. The two conventions differ, because
        :math:`\mathbf{P} = \mathbf{X}\mathbf{S}^{-1}` carries its own
        geometry dependence and
        :math:`\partial\mathcal{U}/\partial\mathbf{P} = 2\mathbf{G}
        (\mathbf{D} - \mathbf{P})` does not vanish away from
        self-consistency. The difference is first order in the shadow
        residual, so it is only visible in a test such as this one that
        deliberately uses a large residual.
        """
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.4, 0.2],
            [1.2, 0.0, -0.1],
        ])

        log = logger.getLogger("pysmd")
        log_disabled = log.disabled
        log.disabled = True
        try:
            dm_prop = self.make_frozen_dm_prop(coords, xc=xc)
            md, fock, dm_ao, dm_prop, dm_lin = self.build_shadow_state(
                coords, dm_prop, xc=xc
            )
            self.assertGreater(np.linalg.norm(dm_ao - dm_prop), 1e-6)

            test_grad = md.update_gradients(
                fock=fock, dm_ao=dm_ao, dm_prop=dm_prop, dm_lin=dm_lin
            )

            fd_grad = np.zeros_like(test_grad)
            step = 1e-4
            for ia in range(coords.shape[0]):
                for ix in range(coords.shape[1]):
                    coords_plus = coords.copy()
                    coords_minus = coords.copy()
                    coords_plus[ia, ix] += step
                    coords_minus[ia, ix] -= step
                    e_plus = self.shadow_potential_energy(
                        coords_plus, dm_prop, xc=xc
                    )
                    e_minus = self.shadow_potential_energy(
                        coords_minus, dm_prop, xc=xc
                    )
                    fd_grad[ia, ix] = (e_plus - e_minus) / (2.0 * step)
        finally:
            log.disabled = log_disabled

        np.testing.assert_allclose(
            test_grad, fd_grad, atol=self.shadow_fd_atol
        )

    def test_rhf_gradient_matches_pyscf(self):
        self.assert_gradient_matches_pyscf(scf.RHF(self.make_mol()))

    def test_rks_lda_gradient_matches_pyscf(self):
        self.assert_rks_gradient_matches_pyscf("lda,vwn5")

    def test_rks_lda_linearized_xc_gradient_matches_finite_difference(self):
        self.assert_linearized_xc_gradient_matches_finite_difference("lda,vwn5")

    def test_rks_pbe_linearized_xc_gradient_matches_finite_difference(self):
        self.assert_linearized_xc_gradient_matches_finite_difference("pbe")

    def test_rhf_shadow_gradient_translationally_invariant(self):
        self.assert_shadow_gradient_translationally_invariant()

    def test_rks_lda_shadow_gradient_translationally_invariant(self):
        self.assert_shadow_gradient_translationally_invariant(xc="lda,vwn5")

    def test_rks_pbe_shadow_gradient_translationally_invariant(self):
        self.assert_shadow_gradient_translationally_invariant(xc="pbe")

    def test_rhf_shadow_gradient_matches_finite_difference(self):
        self.assert_shadow_gradient_matches_finite_difference()

    def test_rks_lda_shadow_gradient_matches_finite_difference(self):
        self.assert_shadow_gradient_matches_finite_difference(xc="lda,vwn5")

    def test_rks_pbe_shadow_gradient_matches_finite_difference(self):
        self.assert_shadow_gradient_matches_finite_difference(xc="pbe")


if __name__ == "__main__":
    unittest.main()
