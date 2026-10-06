#
# Copyright (c) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare derivative works, distribute copies
# to the public, perform publicly and display publicly, and to permit others
# to do so.

"""PySMD adapter for PySCF unrestricted Hartree--Fock calculations.

Densities, Fock matrices, and potentials carry a leading alpha/beta axis and
have shape ``(2, nao, nao)``; occupations have shape ``(2, nmo)`` with values
in ``[0, 1]``. This is already PySCF's unrestricted convention, so unlike the
restricted adapters no factor-of-two conversion is applied at the PySCF
boundary.

The supplied PySCF molecule and mean-field objects are copied, so moving the
nuclei during dynamics never changes the caller's objects, and interfaces
created with :meth:`UHF.with_spin` keep independent spin populations,
orbitals, and grids.

As for the restricted adapters, the gradient methods return per-atom
contractions. The two-electron term is built from both supplied densities,
which gives the exact shadow two-electron force when the propagated and
linearized densities differ; see
:meth:`pysmd.interface.pyscf.base.PyscfBase.compute_two_electron_gradient`.
"""

from typing import Any, Callable

import numpy
from pyscf import dft, gto, scf

from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.interface.pyscf.base import PyscfBase

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()


class UHF(PyscfBase):
    """Interface class for PySCF unrestricted Hartree-Fock (UHF) calculations.

    Densities are shaped ``(2, nao, nao)`` and ordered alpha, beta. Spin
    occupations lie in ``[0, 1]``; no closed-shell factor of two is used.
    The class can be initialized with either a PySCF molecule object or a
    UHF mean-field object, which is copied.
    """

    spin_channels = 2

    def __init__(
        self,
        pyscf_mol: gto.MoleBase | None = None,
        pyscf_mf: scf.uhf.UHF | None = None,
    ) -> None:
        """Initialize UHF interface with PySCF objects.

        Parameters
        ----------
        pyscf_mol : gto.MoleBase, optional
            PySCF molecule object. Used to create a UHF mean-field object if
            pyscf_mf is not provided. ``mol.spin`` is N_alpha - N_beta.
        pyscf_mf : scf.uhf.UHF, optional
            PySCF UHF mean-field object (may be pre-converged or unconverged).
            If provided, takes precedence over pyscf_mol.

        Raises
        ------
        TypeError
            If neither object is provided, if either has the wrong type, or if
            pyscf_mf is a Kohn-Sham object (use :class:`UKS`).
        """
        log.info(">> Instantiating PySCF-backend UHF interface...")

        if pyscf_mf is not None:
            if isinstance(pyscf_mf, dft.rks.KohnShamDFT):
                raise TypeError("Use UKS or PyscfDriver for a DFT mean-field object.")
            if not isinstance(pyscf_mf, scf.uhf.UHF):
                raise TypeError(f"pyscf_mf must be an instance of scf.uhf.UHF, "
                                f"got {type(pyscf_mf).__name__}")
            mf = pyscf_mf
        elif pyscf_mol is not None:
            if not isinstance(pyscf_mol, gto.MoleBase):
                raise TypeError(f"pyscf_mol must be an instance of gto.MoleBase, "
                                f"got {type(pyscf_mol).__name__}")
            mf = scf.uhf.UHF(pyscf_mol)
        else:
            raise TypeError("Must provide either pyscf_mf (scf.uhf.UHF) "
                            "or pyscf_mol (gto.MoleBase).")

        self._initialize_unrestricted(mf)

        return


    def _initialize_unrestricted(self, mf: scf.uhf.UHF) -> None:
        """Take ownership of copies of mf and its molecule, in atomic units."""
        mf = mf.copy()
        mol = mf.mol.copy()
        mol.set_geom_(mol.atom_coords(unit="Bohr"), unit="Bohr")
        for name in ("grids", "nlcgrids"):
            grids = getattr(mf, name, None)
            if grids is not None:
                setattr(mf, name, grids.copy())
        mf.reset(mol)
        mf.verbose = mol.verbose = 0

        PyscfBase.__init__(self, pyscf_mf=mf)
        self.grad = mf.nuc_grad_method()

        return


    @staticmethod
    def _to_pyscf_density(dm: Any) -> Any:
        """PySMD and PySCF share the per-spin unrestricted density convention."""
        return la.to_numpy(dm)


    @staticmethod
    def _from_pyscf_density(dm: Any) -> Any:
        """PySMD and PySCF share the per-spin unrestricted density convention."""
        return la.asarray(dm)


    def get_spin_electron_counts(self) -> tuple[int, int]:
        """Return the fixed (N_alpha, N_beta) populations."""
        return tuple(int(count) for count in self.mf.nelec)


    def with_spin(self, spin: int) -> "UHF":
        """Return an independent interface at the same geometry with new N_alpha - N_beta.

        Settings such as the basis, functional, and grids are retained; orbitals
        and convergence information are discarded.
        """
        mol = self.mol.copy()
        mol.spin = spin
        counts = mol.nelec  # also validates electron/spin parity
        if min(counts) < 0 or max(counts) > mol.nao_nr():
            raise ValueError("Requested spin populations do not fit this basis.")
        mf = self.mf.copy()
        for name in ("grids", "nlcgrids"):
            grids = getattr(mf, name, None)
            if grids is not None:
                setattr(mf, name, grids.copy())
        mf.reset(mol)
        mf.nelec = counts
        mf.mo_coeff = mf.mo_occ = mf.mo_energy = None
        mf.converged = False
        mf.chkfile = None
        return type(self)(pyscf_mf=mf)


    def compute_overlap_matrix(self) -> Any:
        r"""Return the overlap matrix :math:`\mathbf{S}` with shape ``(nao, nao)``."""
        return la.asarray(scf.hf.get_ovlp(mol=self.mol))


    def compute_density_matrix(
        self,
        mo_coeff: Any = None,
        mo_occ: Any = None,
    ) -> Any:
        """Return per-spin AO densities with shape ``(2, nao, nao)``.

        Coefficients and occupations default to those stored on the mean-field
        object. Occupations use the per-spin ``[0, 1]`` convention.
        """
        if mo_coeff is None: mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_occ is None: mo_occ = getattr(self.mf, "mo_occ", None)
        if mo_coeff is None or mo_occ is None:
            raise ValueError(
                "Molecular orbital coefficients and occupations are unavailable."
            )

        result = scf.uhf.make_rdm1(la.to_numpy(mo_coeff), la.to_numpy(mo_occ))
        return self._from_pyscf_density(result)


    def compute_guess_density_matrix(
        self,
        mo_coeff: Any = None,
        mo_occ: Any = None,
    ) -> Any:
        """Return an initial per-spin density guess.

        Unlike the restricted adapters, orbitals *and* occupations stored on the
        mean-field object are used when available. A converged, or deliberately
        modified (non-Aufbau), reference therefore supplies the initial guess.
        Otherwise PySCF's ``init_guess`` (default: 'minao') is used.
        """
        if mo_coeff is None: mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_occ is None: mo_occ = getattr(self.mf, "mo_occ", None)
        if mo_coeff is None or mo_occ is None:
            init_guess: str = getattr(self.mf, "init_guess", "minao")
            return self._from_pyscf_density(
                self.mf.get_init_guess(mol=self.mol, key=init_guess)
            )
        return self.compute_density_matrix(mo_coeff=mo_coeff, mo_occ=mo_occ)


    def compute_core_hamiltonian_matrix(self) -> Any:
        """Return the spin-independent core Hamiltonian with shape ``(nao, nao)``."""
        return la.asarray(scf.hf.get_hcore(mol=self.mol))


    def compute_eff_potential_matrix(self, dm: Any) -> Any:
        r"""Return per-spin effective potentials, :math:`J[D_\alpha + D_\beta] - K[D_s]`."""
        result = self.mf.get_veff(mol=self.mol, dm=self._to_pyscf_density(dm))
        return la.asarray(result)


    def compute_coulomb_matrix(self, dm: Any) -> Any:
        """Return the shared Coulomb matrix of the total (alpha + beta) density."""
        dm_numpy = self._to_pyscf_density(dm)
        return la.asarray(self.mf.get_j(mol=self.mol, dm=dm_numpy[0] + dm_numpy[1]))


    def compute_exchange_matrix(self, dm: Any) -> Any:
        """Return per-spin exchange matrices with shape ``(2, nao, nao)``."""
        return la.asarray(self.mf.get_k(mol=self.mol, dm=self._to_pyscf_density(dm)))


    def compute_xc_terms(self, dm: Any) -> tuple[float, Any]:
        """Return UHF exchange-correlation scalar and potential matrices (zeros)."""
        return 0.0, la.zeros_like(la.asarray(dm))


    def get_potential_response(self, dm: Any) -> Callable[[Any], Any]:
        r"""Return the operator applying :math:`\partial V_{\mathrm{eff}}/\partial P`.

        The UHF potential is linear, so the response is independent of ``dm``:
        :math:`\delta V_s = J[\delta D_\alpha + \delta D_\beta] - K[\delta D_s]`.
        """
        def response(delta_dm):
            return self.compute_eff_potential_matrix(delta_dm)
        return response


    # =========================================================================
    # Gradient components
    # =========================================================================

    def compute_nuclear_gradient(self) -> Any:
        """Return the nuclear repulsion gradient with shape ``(natom, 3)``."""
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        return la.asarray(self.grad.grad_nuc(mol=self.mol))


    def compute_overlap_gradient(self) -> Any:
        """Return the overlap derivative with shape ``(3, nao, nao)``."""
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        return la.asarray(self.grad.get_ovlp(mol=self.mol))


    def compute_1e_gradient(self) -> Callable[[int], Any]:
        """Return a generator of per-atom core-Hamiltonian derivatives."""
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        generator = self.grad.hcore_generator(mol=self.mol)

        def backend_generator(atom_index):
            return la.asarray(generator(atom_index))

        return backend_generator


    def compute_coulomb_gradient(self, dm: Any) -> Any:
        """Return the Coulomb derivative build of the total density, ``(3, nao, nao)``."""
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        dm_numpy = self._to_pyscf_density(dm)
        return la.asarray(self.grad.get_j(mol=self.mol, dm=dm_numpy[0] + dm_numpy[1]))


    def compute_exchange_gradient(self, dm: Any) -> Any:
        """Return per-spin exchange derivative builds, ``(2, 3, nao, nao)``."""
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        return la.asarray(self.grad.get_k(mol=self.mol, dm=self._to_pyscf_density(dm)))


    def _jk_derivatives(self, dms: Any) -> tuple[Any, Any]:
        """Return Coulomb and exchange derivative builds for stacked densities.

        A single pass over the derivative integrals serves every density in
        ``dms`` (shape ``(n, nao, nao)``); both results have shape
        ``(n, 3, nao, nao)``.
        """
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        return self.grad.get_jk(mol=self.mol, dm=dms)


    def _veff_derivative_builds(self, dms: tuple[Any, ...]) -> list[Any]:
        r"""Return :math:`\mathbf{G}'_s = \mathbf{J}'[X_\alpha + X_\beta] - \mathbf{K}'[X_s]`
        for each per-spin density ``X`` in ``dms``, from one integral pass."""
        stacked = numpy.concatenate([la.to_numpy(dm) for dm in dms])
        vj, vk = self._jk_derivatives(stacked)
        builds = []
        for i in range(len(dms)):
            coulomb = vj[2 * i] + vj[2 * i + 1]
            builds.append(la.asarray(coulomb[None] - vk[2 * i:2 * i + 2]))
        return builds


    def compute_xc_gradient(self, dm_prop: Any, dm_ao: Any) -> tuple[Any, Any]:
        """Return zero exchange-correlation gradient components for UHF."""
        nao = self.mol.nao_nr()
        return la.zeros((2, 3, nao, nao)), la.zeros((self.get_num_atoms(), 3))


    def compute_pulay_gradient(self, weight: Any) -> Any:
        r"""Return :math:`-2\,\mathrm{Tr}[\mathbf{S}_{R_I}\mathbf{W}]` per atom.

        ``weight`` is a spin-summed, energy-weighted density-like matrix.
        Only the AO rows of each atom are differentiated; the factor of two
        accounts for the corresponding columns.
        """
        s_grad = self.compute_overlap_gradient()
        aoslices = self.compute_ao_slices_by_atom()
        total_grad = la.zeros((self.get_num_atoms(), 3))

        for k in range(self.get_num_atoms()):
            p0, p1 = aoslices[k, 2:]
            total_grad[k] -= 2.0 * la.einsum(
                "xij, ij-> x", s_grad[:, p0:p1], weight[p0:p1], optimize=True
            )

        return total_grad


    def compute_one_electron_gradient(self, dm_ao: Any, fock: Any) -> Any:
        r"""Return the one-electron and Pulay gradient contributions per atom.

        The Pulay term uses :math:`\sum_s \mathbf{S}^{-1}\mathbf{F}_s\mathbf{D}_s`,
        the unrestricted shadow generalization of the energy-weighted density.
        """
        hcore_grad = self.compute_1e_gradient()
        inv_ovlp = self.compute_inv_overlap_matrix()
        dm_tot = dm_ao[0] + dm_ao[1]
        fock_dm = sum(
            la.matmul(la.matmul(inv_ovlp, fock[s]), dm_ao[s]) for s in range(2)
        )

        total_grad = self.compute_pulay_gradient(fock_dm)
        for k in range(self.get_num_atoms()):
            total_grad[k] += la.einsum(
                "xij, ij-> x", hcore_grad(k), dm_tot, optimize=True
            )

        return total_grad


    def compute_two_electron_gradient(self, dm_bra: Any, dm_ket: Any) -> Any:
        r"""Return the symmetrized two-electron gradient contribution per atom.

        With :math:`\mathbf{G}'_s[\mathbf{X}] = \mathbf{J}'[X_\alpha + X_\beta]
        - \mathbf{K}'[X_s]`, the result is
        :math:`\sum_s T[\mathbf{G}'_s[\mathrm{bra}], \mathrm{ket}_s]
        + T[\mathbf{G}'_s[\mathrm{ket}], \mathrm{bra}_s]`, which at
        self-consistency reduces to PySCF's ``2 sum_s G'_s[D] D_s``.
        """
        if dm_ket is dm_bra:
            # Self-consistent case; the two builds coincide.
            two_e_grad_bra = two_e_grad_ket = self._veff_derivative_builds((dm_bra,))[0]
        else:
            two_e_grad_bra, two_e_grad_ket = self._veff_derivative_builds((dm_bra, dm_ket))

        aoslices = self.compute_ao_slices_by_atom()
        total_grad = la.zeros((self.get_num_atoms(), 3))

        for k in range(self.get_num_atoms()):
            p0, p1 = aoslices[k, 2:]
            total_grad[k] += la.einsum(
                "sxij, sij-> x",
                two_e_grad_bra[:, :, p0:p1],
                dm_ket[:, p0:p1],
                optimize=True,
            )
            total_grad[k] += la.einsum(
                "sxij, sij-> x",
                two_e_grad_ket[:, :, p0:p1],
                dm_bra[:, p0:p1],
                optimize=True,
            )

        return total_grad


    def compute_xc_gradient_per_atom(self, dm_prop: Any, dm_ao: Any) -> Any:
        """Return zero exchange-correlation gradient contributions for UHF."""
        return la.zeros((self.get_num_atoms(), 3))
