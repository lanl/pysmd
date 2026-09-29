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

"""Optional GPU4PySCF restricted mean-field interfaces.

Electronic-structure operations and the full shadow-gradient contract are
evaluated on the device. GPU4PySCF exposes no derivative matrices -- its
gradient classes set ``get_j``, ``get_k`` and ``get_veff`` to
``NotImplemented`` and contract the density inside the CUDA kernels -- so the
gradient is assembled from per-atom energy-derivative routines instead. See
:meth:`_GPU4PySCFBase.compute_two_electron_gradient` for how the shadow
two-electron term is recovered from that API. No CPU gradient delegate is used;
the only host transfer in the RKS exchange-correlation gradient is the final
conversion of the assembled result returned by the backend.

PySMD uses spatial-orbital occupations in the range ``[0, 1]``. GPU4PySCF
restricted mean-field routines follow PySCF's spin-summed occupation
convention, ``[0, 2]``. Occupations supplied by PySMD are multiplied by two
before GPU4PySCF calls, while occupations read from GPU4PySCF are divided by
two before being used or returned to PySMD. The corresponding density matrices
are converted at the same interface boundary.
"""

from typing import Any

from pyscf import gto

from pysmd.interface.pyscf.rhf import RHF
from pysmd.interface.pyscf.rks import DEFAULT_XC, RKS, verify_supported_xc

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()


def _load_gpu4pyscf() -> tuple[Any, Any]:
    try:
        from gpu4pyscf import dft as gpu_dft
        from gpu4pyscf import scf as gpu_scf
    except ImportError as exc:
        raise ImportError(
            "GPU4PySCF backend requested, but gpu4pyscf is not importable. "
            "Install GPU4PySCF and the CuPy distribution matching the CUDA "
            "runtime, or add their source trees to PYTHONPATH."
        ) from exc
    return gpu_scf, gpu_dft


def _strip_density_tags(dm: Any) -> Any:
    """Return a plain device array view of a density matrix.

    GPU4PySCF attaches metadata such as ``mo_coeff``/``mo_occ`` to density
    matrices, and its density-fitting code paths reconstruct the density from
    those tags in preference to the array contents. A shadow density is not
    representable that way, so any incoming tags must be discarded before the
    array is handed to a GPU4PySCF gradient routine.
    """
    import cupy

    return cupy.asarray(dm, order="C").view(cupy.ndarray)


class _GPU4PySCFBase:
    """Shared device-resident implementation for GPU4PySCF interfaces.

    This base supplies the common GPU4PySCF implementation used by
    :class:`GPU4RHF` and :class:`GPU4RKS`. Electronic matrices, energies, and
    gradient contractions remain on the device through the backend calls.
    It does not delegate gradient work to the CPU PySCF interfaces. The CPU
    RHF/RKS classes remain in the inheritance chain to provide the common
    PySMD interface and genuinely shared non-computational operations; the
    GPU classes override the electronic and gradient operations that require
    backend-specific implementations.
    """

    ### Coefficient of exact exchange in the two-electron energy.
    # Hartree-Fock carries a full exchange term; pure density functionals carry
    # none. Subclasses set this to select the corresponding GPU4PySCF
    # ``k_factor``.
    _exchange_factor = 1.0

    @staticmethod
    def _to_pyscf_density(dm: Any) -> Any:
        return 2.0 * la.asarray(dm)

    @staticmethod
    def _from_pyscf_density(dm: Any) -> Any:
        return 0.5 * la.asarray(dm)

    def _initialize_gpu(self, mf: Any) -> None:
        # The array backend is process-wide state. Record what it was so a
        # caller can put it back; otherwise constructing a GPU interface would
        # silently leave unrelated NumPy code running on CuPy arrays.
        self._previous_array_backend = linalg_helper.get_linalg_backend_name()
        linalg_helper.set_linalg_backend("cupy")

        mf.verbose = mf.mol.verbose = 0
        if hasattr(mf.mol, "unit") and not gto.is_au(mf.mol.unit):
            mf.mol.unit = "au"
            mf.mol.set_geom_(
                atoms_or_coords=mf.mol.atom_coords(unit=mf.mol.unit),
                unit=mf.mol.unit,
                inplace=True,
            )

        self.mf = mf
        self.mol = mf.mol
        self.mf_type = mf.__class__.__name__
        self.temp = None
        self.conv_scf = False
        self.electronic_backend = "gpu"
        self.gradient_backend = "gpu"
        self.array_backend = "cupy"
        self._extract_pyscf_attributes()

        ### Device-resident gradient object
        # Grid response is always included: a gradient evaluated on a fixed
        # quadrature grid is not the exact derivative of the
        # quadrature-approximated energy, so such forces are not conservative.
        # GPU4PySCF has a second reason to require it here, namely that its
        # fixed-grid path evaluates the density from molecular orbitals and so
        # assumes an idempotent density, which a propagated shadow density is
        # not.
        self.grad = mf.Gradients()
        self.grad.grid_response = True

    def restore_array_backend(self) -> None:
        """Restore the array backend that was active before construction.

        Constructing a GPU interface switches the process-wide PySMD array
        backend to CuPy. Call this once the interface is no longer needed so
        that unrelated NumPy-based code is not left running on CuPy arrays.
        """
        previous = getattr(self, "_previous_array_backend", None)
        if previous is not None:
            linalg_helper.set_linalg_backend(previous)

    def _ensure_cupy_backend(self) -> None:
        if linalg_helper.get_linalg_backend_name() != "cupy":
            raise RuntimeError(
                "GPU4PySCF interfaces require the process-wide PySMD array "
                "backend to remain set to cupy."
            )

    def get_atomic_coords(self) -> Any:
        self._ensure_cupy_backend()
        return super().get_atomic_coords()

    def get_atomic_masses(self) -> Any:
        self._ensure_cupy_backend()
        return super().get_atomic_masses()

    def get_updated_atomic_coords(self, new_coords: Any) -> Any:
        self._ensure_cupy_backend()
        coords = la.to_numpy(new_coords)

        self.mol.set_geom_(coords, unit=self.mol.unit, inplace=True)
        # Cached two-electron integrals and quadrature grids are
        # geometry-dependent and must not survive an MD step.
        self.mf.reset(self.mol)
        if hasattr(self.grad, "reset"):
            self.grad.reset(self.mol)
        return self.get_atomic_coords()

    def compute_overlap_matrix(self) -> Any:
        self._ensure_cupy_backend()
        return la.asarray(self.mf.get_ovlp(mol=self.mol))

    def compute_density_matrix(self, mo_coeff: Any = None, mo_occ: Any = None) -> Any:
        self._ensure_cupy_backend()
        if mo_coeff is None:
            mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_occ is None:
            mo_occ = getattr(self.mf, "mo_occ", None)
            if mo_occ is not None:
                mo_occ = 0.5 * la.asarray(mo_occ)
        if mo_coeff is None or mo_occ is None:
            raise ValueError(
                "Molecular orbital coefficients and occupations are unavailable."
            )
        mo_occ = 2.0 * la.asarray(mo_occ)
        return self._from_pyscf_density(
            self.mf.make_rdm1(
                mo_coeff=la.asarray(mo_coeff),
                mo_occ=mo_occ,
            )
        )

    def compute_guess_density_matrix(self, mo_coeff: Any = None, mo_occ: Any = None) -> Any:
        self._ensure_cupy_backend()
        if mo_coeff is None:
            mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_coeff is not None and mo_occ is not None:
            return self.compute_density_matrix(mo_coeff, mo_occ)
        key = getattr(self.mf, "init_guess", "minao")
        return self._from_pyscf_density(
            self.mf.get_init_guess(mol=self.mol, key=key)
        )

    def compute_core_hamiltonian_matrix(self) -> Any:
        self._ensure_cupy_backend()
        return la.asarray(self.mf.get_hcore(mol=self.mol))

    def compute_eff_potential_matrix(self, dm: Any) -> Any:
        self._ensure_cupy_backend()
        return la.asarray(
            self.mf.get_veff(mol=self.mol, dm=self._to_pyscf_density(dm))
        )

    def compute_coulomb_matrix(self, dm: Any) -> Any:
        self._ensure_cupy_backend()
        return la.asarray(
            self.mf.get_j(mol=self.mol, dm=self._to_pyscf_density(dm))
        )

    def get_electronic_energy(self, dm: Any, h1e: Any, vhf: Any) -> tuple[Any, Any, Any]:
        self._ensure_cupy_backend()
        e_tot, e2 = self.mf.energy_elec(
            dm=self._to_pyscf_density(dm),
            h1e=la.asarray(h1e),
            vhf=la.asarray(vhf),
        )
        return e_tot - e2, e2, e_tot

    def compute_nuclear_gradient(self) -> Any:
        self._ensure_cupy_backend()
        return la.asarray(self.grad.grad_nuc(mol=self.mol))

    def compute_overlap_gradient(self):
        """Return the overlap derivative matrix.

        Not used by the gradient assembly on this backend, which obtains the
        Pulay term from ``_hcore_energy``. Retained because the matrix itself
        is occasionally useful for diagnostics.
        """
        self._ensure_cupy_backend()
        return la.asarray(self.grad.get_ovlp(self.mol))

    def compute_1e_gradient(self):
        raise NotImplementedError(
            "GPU4PySCF does not expose a core-Hamiltonian derivative "
            "generator. Use compute_one_electron_gradient, which returns the "
            "per-atom one-electron and Pulay contributions together."
        )

    def compute_coulomb_gradient(self, dm):
        raise NotImplementedError(
            "GPU4PySCF does not expose Coulomb derivative matrices; its "
            "gradient classes set get_j to NotImplemented and contract the "
            "density inside the device kernels. Use "
            "compute_two_electron_gradient instead."
        )

    def compute_exchange_gradient(self, dm):
        raise NotImplementedError(
            "GPU4PySCF does not expose exchange derivative matrices; its "
            "gradient classes set get_k to NotImplemented and contract the "
            "density inside the device kernels. Use "
            "compute_two_electron_gradient instead."
        )

    def compute_xc_gradient(self, dm_prop: Any, dm_ao: Any) -> tuple[Any, Any]:
        raise NotImplementedError(
            "GPU4PySCF does not expose exchange-correlation derivative "
            "matrices. Use compute_xc_gradient_per_atom instead."
        )


    # =========================================================================
    # Per-atom gradient contributions
    # =========================================================================
    # GPU4PySCF does not expose derivative matrices at all: its gradient
    # classes set get_j/get_k/get_veff to NotImplemented, and the density
    # contraction is performed inside the CUDA kernels. Every term below is
    # therefore assembled from the per-atom energy-derivative API instead.

    def compute_one_electron_gradient(self, dm_ao: Any, fock: Any) -> Any:
        r"""Return the one-electron and Pulay gradient contributions per atom.

        Contracts GPU4PySCF's core-Hamiltonian and overlap derivative tensors
        directly. The energy-weighted density is
        :math:`\mathbf{S}^{-1}\mathbf{F}\mathbf{D}`, the shadow generalization
        of ``make_rdm1e``; the two coincide at self-consistency.
        """
        self._ensure_cupy_backend()

        inv_ovlp = self.compute_inv_overlap_matrix()
        energy_weighted_dm = la.matmul(la.matmul(inv_ovlp, fock), dm_ao)
        import cupy

        hcore_gradient = la.asarray(self.grad.get_hcore(mol=self.mol))
        overlap_gradient = la.asarray(self.grad.get_ovlp(mol=self.mol))

        from gpu4pyscf.df import int3c2e
        from gpu4pyscf.grad.rhf import contract_h1e_dm

        hcore_term = contract_h1e_dm(
            self.mol, hcore_gradient, _strip_density_tags(dm_ao), hermi=1
        )
        # The shadow energy-weighted density is not generally symmetric.
        # Match PySMD's CPU convention by selecting the AO columns belonging
        # to each atom before contracting, rather than using the Hermitian
        # contraction helper, which exchanges the two density indices.
        pulay_term = cupy.zeros((self.mol.natm, 3))
        ao_slices = self.mol.aoslice_by_atom()
        for atom_index, (_, _, p0, p1) in enumerate(ao_slices):
            pulay_term[atom_index] = 2.0 * cupy.einsum(
                "xij,ij->x",
                overlap_gradient[:, p0:p1],
                _strip_density_tags(energy_weighted_dm)[p0:p1],
            )
        nuclear_attraction_term = int3c2e.get_dh1e(
            self.mol, _strip_density_tags(dm_ao)
        )
        return la.asarray(hcore_term) - la.asarray(pulay_term) + la.asarray(
            nuclear_attraction_term
        )


    def compute_two_electron_gradient(self, dm_bra: Any, dm_ket: Any) -> Any:
        r"""Return the two-electron gradient contribution per atom.

        GPU4PySCF exposes only ``jk_energy_per_atom``, which differentiates the
        two-electron energy for a *single* density appearing on both sides of
        the contraction. Writing the shadow two-electron energy as a symmetric
        bilinear form,

        .. math::

            b(X, Y) = \tfrac12 \mathrm{Tr}[J(X) Y]
                    - \tfrac{k}{4} \mathrm{Tr}[K(X) Y],

        that routine returns the derivative of the diagonal
        :math:`Q(D) = b(D, D)`. Expanding
        :math:`Q(X + Y) = Q(X) + 2 b(X, Y) + Q(Y)` gives the required
        off-diagonal element from three diagonal evaluations,

        .. math::

            \frac{\partial b(X, Y)}{\partial R}
                = \tfrac12 \left[
                    \frac{\partial Q(X + Y)}{\partial R}
                  - \frac{\partial Q(X)}{\partial R}
                  - \frac{\partial Q(Y)}{\partial R}
                  \right].

        This is an exact algebraic identity, not an approximation, and it is
        automatically the symmetrized form required by Eq. 43 of the reference.
        It collapses to a single evaluation when the two densities coincide.
        """
        self._ensure_cupy_backend()

        def diagonal_derivative(dm):
            # jk_energy_per_atom returns j_factor * dE_J/dR - k_factor * dE_K/dR.
            # j_factor = 1 corresponds to the one-half coefficient of the
            # Coulomb energy, matching the convention GPU4PySCF itself uses
            # when assembling a pure-functional two-electron gradient.
            return self.grad.jk_energy_per_atom(
                _strip_density_tags(dm),
                j_factor=1.0,
                k_factor=self._exchange_factor,
                hermi=1,
            )

        if dm_ket is dm_bra:
            # Self-consistent case; the identity reduces to the diagonal.
            return la.asarray(diagonal_derivative(dm_bra))

        combined = diagonal_derivative(la.asarray(dm_bra) + la.asarray(dm_ket))
        return la.asarray(
            0.5 * (
                combined
                - diagonal_derivative(dm_bra)
                - diagonal_derivative(dm_ket)
            )
        )


class GPU4RHF(_GPU4PySCFBase, RHF):
    """Device-resident GPU4PySCF restricted Hartree--Fock interface.

    The interface uses GPU4PySCF for electronic quantities and for all
    gradient contributions. It inherits the PySMD-facing method names from
    the CPU RHF adapter without delegating calculations to that adapter.
    """

    def __init__(self, pyscf_mol: gto.MoleBase | None = None, pyscf_mf: Any = None) -> None:
        gpu_scf, _ = _load_gpu4pyscf()
        if pyscf_mf is not None:
            if (
                not isinstance(pyscf_mf, gpu_scf.hf.RHF)
                or isinstance(pyscf_mf, gpu_scf.rohf.ROHF)
            ):
                raise TypeError(
                    "pyscf_mf must be a GPU4PySCF RHF object, got "
                    f"{type(pyscf_mf).__module__}.{type(pyscf_mf).__name__}"
                )
            mf = pyscf_mf
        elif pyscf_mol is not None:
            if not isinstance(pyscf_mol, gto.MoleBase):
                raise TypeError("pyscf_mol must be a PySCF Mole object.")
            mf = gpu_scf.RHF(pyscf_mol)
        else:
            raise TypeError("Must provide either pyscf_mf or pyscf_mol.")
        self._initialize_gpu(mf)

    def compute_exchange_matrix(self, dm: Any) -> Any:
        self._ensure_cupy_backend()
        return la.asarray(self.mf.get_k(mol=self.mol, dm=la.asarray(dm)))

    def compute_xc_terms(self, dm: Any) -> tuple[float, Any]:
        self._ensure_cupy_backend()
        return 0.0, la.zeros_like(dm)

    def compute_xc_gradient_per_atom(self, dm_prop: Any, dm_ao: Any) -> Any:
        """Return zero exchange-correlation gradient for Hartree-Fock."""
        self._ensure_cupy_backend()
        return la.zeros((self.get_num_atoms(), 3))


class GPU4RKS(_GPU4PySCFBase, RKS):
    """Device-resident GPU4PySCF restricted Kohn--Sham interface.

    Only pure LDA and GGA functionals are supported, matching the CPU RKS
    interface. Electronic quantities and the linearized exchange-correlation
    gradient are evaluated with GPU4PySCF and CuPy; no CPU gradient delegate
    is used. See :func:`pysmd.interface.pyscf.rks.verify_supported_xc`.
    """

    ### Pure density functionals carry no exact exchange.
    _exchange_factor = 0.0

    def __init__(self, pyscf_mol: gto.MoleBase | None = None, pyscf_mf: Any = None, xc: str = DEFAULT_XC) -> None:
        _, gpu_dft = _load_gpu4pyscf()
        if pyscf_mf is not None:
            if not isinstance(pyscf_mf, gpu_dft.rks.RKS):
                raise TypeError(
                    "pyscf_mf must be a GPU4PySCF RKS object, got "
                    f"{type(pyscf_mf).__module__}.{type(pyscf_mf).__name__}"
                )
            mf = pyscf_mf
        elif pyscf_mol is not None:
            if not isinstance(pyscf_mol, gto.MoleBase):
                raise TypeError("pyscf_mol must be a PySCF Mole object.")
            mf = gpu_dft.RKS(pyscf_mol, xc=xc)
        else:
            raise TypeError("Must provide either pyscf_mf or pyscf_mol.")

        ### Verify the requested functional is supported
        # Deliberately performed before _initialize_gpu, which mutates the
        # process-wide array backend. Validating first guarantees that a
        # rejected functional leaves the backend untouched, matching the
        # behaviour already required of a failed GPU4PySCF import.
        verify_supported_xc(mf.xc, mf)
        self.xc = mf.xc

        self._initialize_gpu(mf)

    def compute_xc_terms(self, dm: Any) -> tuple[Any, Any]:
        self._ensure_cupy_backend()
        grids = self.mf.grids
        if getattr(grids, "coords", None) is None:
            grids.build()
        _, exc, vxc = self.mf._numint.nr_rks(
            self.mol,
            grids,
            self.mf.xc,
            la.asarray(dm),
            max_memory=self.mf.max_memory,
            verbose=self.mf.verbose,
        )
        return exc, la.asarray(vxc)

    def compute_xc_gradient_per_atom(self, dm_prop: Any, dm_ao: Any) -> Any:
        r"""Return all exchange-correlation gradient contributions per atom.

        Linearized analogue of GPU4PySCF's
        :func:`gpu4pyscf.grad.rks.get_exc_full_response`. The loop structure,
        AO derivative orders, quadrature helpers and grid-weight response
        machinery are kept deliberately close to that routine.

        The shadow functional is

        .. math::

            E_{xc}^{lin} = E_{xc}[\rho_{prop}]
                + \int v_{xc}[\rho_{prop}]\,(\rho_{ao} - \rho_{prop}),

        so ``dm_prop`` builds :math:`\rho`, :math:`v_{xc}` and :math:`f_{xc}`
        while ``dm_ao`` is the target density. Differentiating produces three
        contributions, all collected here:

        1. :math:`\int v_{xc}[\rho_{prop}]\,\partial_R \rho_{ao}`
        2. :math:`\int f_{xc}[\rho_{prop}]\,(\rho_{ao} - \rho_{prop})\,
           \partial_R \rho_{prop}`
        3. the response of the quadrature weights, driven by the first-order
           energy density
           :math:`\epsilon_{xc}\rho_{prop} + v_{xc}\!\cdot\!
           (\rho_{ao} - \rho_{prop})`.

        Grid response is always included, so the result is the exact
        derivative of the quadrature-approximated energy and the forces are
        conservative.

        Notes
        -----
        Only LDA and GGA are handled, which the constructor already enforces.
        That restriction is what keeps this routine short: no kinetic-energy
        density, no Laplacian and no non-local correlation terms are required.
        """
        self._ensure_cupy_backend()

        import cupy
        from gpu4pyscf.dft import numint as gpu_numint
        from gpu4pyscf.grad import rhf as gpu_rhf_grad
        from gpu4pyscf.grad import rks as gpu_rks_grad
        from gpu4pyscf.hessian.rks import get_dweight_dA
        from gpu4pyscf.lib.cupy_helper import get_avail_mem, take_last2d

        from pysmd.interface.pyscf.rks import SUPPORTED_XC_TYPES

        ### Build a private numerical integrator
        # Mirrors get_exc_full_response: the mean-field object's integrator is
        # left untouched so its cached grid sorting is not disturbed.
        ni = gpu_numint.NumInt()
        xctype = ni._xc_type(self.mf.xc)
        if xctype not in SUPPORTED_XC_TYPES:
            raise NotImplementedError(
                f"Linearized RKS gradients are not implemented for XC type "
                f"{xctype!r}."
            )

        grids = self.grad.grids if self.grad.grids is not None else self.mf.grids
        grids = grids.copy()
        grids.build(sort_grids_of_each_atom=True)
        ngrids = grids.coords.shape[0]

        ni.gdftopt = None
        ni.build(self.mol, grids.coords)
        opt = ni.gdftopt
        sorted_mol = opt._sorted_mol
        nao = sorted_mol.nao
        natm = self.mol.natm

        dm_prop_sorted = opt.sort_orbitals(
            _strip_density_tags(dm_prop), axis=[0, 1]
        )
        dm_ao_sorted = opt.sort_orbitals(
            _strip_density_tags(dm_ao), axis=[0, 1]
        )

        if xctype == "LDA":
            ao_deriv = 0
            ncomp = 1
        else:
            ao_deriv = 1
            ncomp = 4

        ### First pass: evaluate both densities on the full grid
        rho_prop = cupy.empty([ncomp, ngrids])
        rho_ao = cupy.empty([ncomp, ngrids])
        # Separate masking buffers for the two densities: both are live within
        # a single loop iteration.
        dm_prop_buf = cupy.empty(nao * nao)
        dm_ao_buf = cupy.empty(nao * nao)
        grid_end = 0
        for ao, idx, weight, _ in ni.block_loop(
            sorted_mol, grids, deriv=ao_deriv, strict_grid_order=True
        ):
            grid_start, grid_end = grid_end, grid_end + weight.size
            rho_prop[:, grid_start:grid_end] = gpu_numint.eval_rho(
                sorted_mol,
                ao,
                take_last2d(dm_prop_sorted, idx, out=dm_prop_buf),
                xctype=xctype,
                hermi=1,
            )
            rho_ao[:, grid_start:grid_end] = gpu_numint.eval_rho(
                sorted_mol,
                ao,
                take_last2d(dm_ao_sorted, idx, out=dm_ao_buf),
                xctype=xctype,
                hermi=1,
            )
        assert grid_end == ngrids

        ### Exchange-correlation potential and kernel at the propagated density
        exc, vxc, fxc = ni.eval_xc_eff(
            self.mf.xc, rho_prop, 2, xctype=xctype, spin=0
        )[:3]
        delta_rho = rho_ao - rho_prop

        # Weighted potential, contracted below against dm_ao.
        weighted_vxc = grids.weights * vxc

        # Weighted kernel contribution, contracted below against dm_prop.
        if xctype == "LDA":
            kernel_vxc = fxc[0, 0] * delta_rho[0]
            weighted_kernel = grids.weights * kernel_vxc[None, :]
            first_order_energy_density = (
                exc * rho_prop[0] + vxc[0] * delta_rho[0]
            )
        else:
            kernel_vxc = cupy.einsum("abr,br->ar", fxc, delta_rho)
            weighted_kernel = grids.weights * kernel_vxc
            first_order_energy_density = (
                exc * rho_prop[0]
                + cupy.einsum("nr,nr->r", vxc, delta_rho)
            )

        nonzero_weight_mask = cupy.abs(grids.weights) > 1e-14
        del rho_prop, rho_ao, delta_rho, fxc, vxc

        ### Response of the quadrature weights
        # Uses the first-order energy density in place of the ground-state
        # energy density that get_exc_full_response contracts here.
        grid_weight_response = cupy.zeros((natm, 3))
        available_memory = int(get_avail_mem() * 0.1)
        bytes_per_grid = (2 * 3 * natm) * 8
        grids_per_batch = int(available_memory / bytes_per_grid)
        if grids_per_batch < 16:
            raise MemoryError(
                "Out of GPU memory for the linearized XC gradient; "
                f"available memory = {get_avail_mem()} bytes, nao = {nao}, "
                f"natm = {natm}, ngrids = {ngrids}."
            )
        grids_per_batch = (grids_per_batch + 15) // 16 * 16

        for grid_start in range(0, ngrids, grids_per_batch):
            grid_stop = min(grid_start + grids_per_batch, ngrids)
            dweight_dA = get_dweight_dA(
                sorted_mol, grids, (grid_start, grid_stop)
            )
            grid_weight_response += cupy.einsum(
                "Adg->Ad",
                dweight_dA
                * first_order_energy_density[grid_start:grid_stop],
            )
        del first_order_energy_density

        ### Second pass: orbital and grid-coordinate response
        potential_vmat = cupy.zeros((3, nao, nao))
        kernel_vmat = cupy.zeros((3, nao, nao))
        grid_coordinate_response = cupy.zeros((natm, 3))

        grid_start = 0
        for ao, idx, weight, _ in ni.block_loop(
            sorted_mol, grids, nao, ao_deriv + 1, strict_grid_order=True
        ):
            grid_stop = grid_start + weight.shape[0]
            block_mask = nonzero_weight_mask[grid_start:grid_stop]
            ao = ao[:, :, block_mask]
            if ao.size == 0:
                grid_start = grid_stop
                continue

            # Grids are sorted by atom, so every point in this block belongs to
            # the same nucleus.
            atom_index = int(grids.atm_idx[grid_start])

            block_vxc = cupy.ascontiguousarray(
                weighted_vxc[:, grid_start:grid_stop][:, block_mask]
            )
            block_kernel = cupy.ascontiguousarray(
                weighted_kernel[:, grid_start:grid_stop][:, block_mask]
            )

            if xctype == "LDA":
                potential_tmp = gpu_rks_grad._d1_dot_(
                    ao[1:4],
                    gpu_numint._scale_ao(ao[0], block_vxc[0]).T,
                )
                kernel_tmp = gpu_rks_grad._d1_dot_(
                    ao[1:4],
                    gpu_numint._scale_ao(ao[0], block_kernel[0]).T,
                )
            else:
                block_vxc[0] *= 0.5
                block_kernel[0] *= 0.5
                potential_tmp = gpu_rks_grad._gga_grad_sum_(
                    ao, block_vxc[:4]
                )
                kernel_tmp = gpu_rks_grad._gga_grad_sum_(
                    ao, block_kernel[:4]
                )

            potential_vmat[:, idx[:, None], idx] += potential_tmp
            kernel_vmat[:, idx[:, None], idx] += kernel_tmp

            # Grid-coordinate response: the potential term pairs with dm_ao and
            # the kernel term with dm_prop, matching the linearized functional.
            grid_coordinate_response[atom_index] += 2.0 * cupy.einsum(
                "xij,ji->x",
                potential_tmp,
                take_last2d(dm_ao_sorted, idx, out=dm_ao_buf),
            )
            grid_coordinate_response[atom_index] += 2.0 * cupy.einsum(
                "xij,ji->x",
                kernel_tmp,
                take_last2d(dm_prop_sorted, idx, out=dm_prop_buf),
            )

            grid_start = grid_stop
        assert grid_start == ngrids

        ### Assemble
        total_grad = (grid_weight_response + grid_coordinate_response).get()

        # Negative sign because nabla_X = -nabla_x. contract_h1e_dm already
        # accounts for the factor of two of a symmetric density matrix.
        total_grad -= gpu_rhf_grad.contract_h1e_dm(
            sorted_mol, potential_vmat, dm_ao_sorted, hermi=1
        )
        total_grad -= gpu_rhf_grad.contract_h1e_dm(
            sorted_mol, kernel_vmat, dm_prop_sorted, hermi=1
        )

        return la.asarray(total_grad)
