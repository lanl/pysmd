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

"""Shared implementation for PySCF-backed PySMD interfaces.

The base adapter delegates ordinary molecule properties, integral-related
operations, and energy components to the attached PySCF objects. It also
converts arrays through :mod:`pysmd.lib.linalg_helper` and supplies common
overlap-matrix and geometry-update helpers. Specialized RHF and RKS classes
implement method-specific electronic and gradient operations.

PySMD uses spatial-orbital occupations and one-spin density matrices. PySCF's
restricted routines use spin-summed occupations and densities, so the concrete
PySCF adapters convert PySMD quantities before calling PySCF and convert
restricted densities back before returning them to PySMD.
"""

import math
from typing import Any

from pyscf import gto, scf
from pyscf.data import elements, nist

from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.interface import qm_software

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()


class PyscfBase(qm_software.QMSoftware):
    """Abstract base adapter for PySCF quantum-chemistry objects.
    
    This class cannot be instantiated as a complete calculation interface.
    Use a concrete adapter such as :class:`RHF` or :class:`RKS`.
    
    It handles molecule and mean-field ownership, common system properties,
    geometry updates, overlap transformations, and energy bookkeeping. Where
    a method corresponds directly to a PySCF operation, the implementation
    preserves PySCF's result and converts it to the active PySMD array
    backend.
    """

    def __init__(
        self,
        pyscf_mf: scf.hf.SCF,
    ) -> None:
        """Initialize common attributes for PySCF interfaces.
        
        Parameters
        ----------
        pyscf_mf : scf.hf.SCF
            PySCF mean-field object. The molecule object is extracted from this.
            
        Raises
        ------
        TypeError
            If pyscf_mf is not an instance of scf.hf.SCF or its subclasses
            
        Notes
        -----
        - Subclasses are responsible for creating the appropriate mean-field object
        - The molecule object is always extracted from the mean-field object
        """
        
        # Validate mean-field object type
        if not isinstance(pyscf_mf, scf.hf.SCF):
            raise TypeError("pyscf_mf must be an instance of "
                            "scf.hf.SCF or its subclasses.")
        
        # Initialize attributes from mean-field object
        self.mf: scf.hf.SCF = pyscf_mf
        self.mol: gto.MoleBase = self.mf.mol
        self.mf_type: str = self.mf.__class__.__name__
        self.temp: (float | None) = None
        self.conv_scf: bool = False
        self.electronic_backend = "cpu"
        self.gradient_backend = "cpu"
        self.array_backend = linalg_helper.get_linalg_backend_name()
        
        # Extract common attributes from PySCF objects
        self._extract_pyscf_attributes()

        return


    @staticmethod
    def _to_pyscf_density(dm: Any) -> Any:
        """Convert a PySMD one-spin density to PySCF's spin-summed density.

        PySCF restricted calls receive this converted density; PySMD-facing
        methods must not expose the converted representation.
        """
        return 2.0 * la.to_numpy(dm)


    @staticmethod
    def _from_pyscf_density(dm: Any) -> Any:
        """Convert a PySCF restricted density to a PySMD one-spin density.

        This inverse conversion is applied before a density returned by PySCF
        is used by the PySMD closed-shell implementation.
        """
        return 0.5 * la.asarray(dm)


    def _extract_pyscf_attributes(self) -> None:
        """Extract common attributes from PySCF mean-field object.
        
        This method extracts:
        - SCF convergence status (conv_scf)
        - Finite temperature value (temp), if available
        """
        # Extract SCF convergence flag from mean-field object
        if hasattr(self.mf, 'converged') and self.mf.converged is not None:
            self.conv_scf = bool(self.mf.converged)
        
        # Extract finite temperature from mean-field object (if available)
        if hasattr(self.mf, 'sigma') and self.mf.sigma is not None:
            # Convert from sigma (kb*T in Hartree) to temperature in Kelvin
            self.temp = float(self.mf.sigma / (nist.BOLTZMANN / nist.HARTREE2J))
            log.info(f"\n>> Mean-field object contains finite temperature attributes...")
            log.info(f" -- Value of sigma (kb*T): {self.mf.sigma:10e}\n"
                     f" -- Inherited temperature (Kelvin): {self.temp}")

        return


    def _diagonalize_overlap(self, tol: float = 1e-14) -> tuple[Any, Any]:
        """Diagonalize overlap matrix and filter small eigenvalues.

        Helper method to avoid redundant diagonalization when computing
        multiple overlap matrix variants.

        Parameters
        ----------
        tol : float
            Stability threshold for filtering eigenvalues; default = 1e-14

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - eigenvalues (filtered, 1D array)
            - eigenvectors (filtered columns, 2D array)
        """
        ovlp = self.compute_overlap_matrix()
        e, v = la.linalg.eigh(ovlp)
        idx = e > tol

        # Log if eigenvalues were filtered
        n_filtered = len(e) - la.sum(idx)
        if n_filtered > 0:
            log.info(f" -- Overlap: Filtered {n_filtered} "
                     f"eigenvalue(s) below tolerance {tol:.2e}")

        return e[idx], v[:, idx]


    # =========================================================================
    # System properties
    # =========================================================================

    def get_scf_convergence(self) -> bool:
        """Return SCF convergence status."""
        return self.conv_scf


    def can_reuse_scf(self, temp: float) -> bool:
        """Return whether converged SCF results are reusable at a temperature."""
        if not self.get_scf_convergence():
            return False
        if self.temp is None:
            return False
        return math.isclose(
            temp,
            self.temp,
            rel_tol=1e-9,
            abs_tol=1e-8,
        )


    def get_mean_field_type(self) -> str:
        """Return the type of PySCF mean-field object."""
        return self.mf_type


    def get_num_atoms(self) -> int:
        """Return the number of atoms."""
        return self.mol.natm


    def get_num_electrons(self) -> int:
        """Return the number of electrons."""
        return self.mol.nelectron


    def get_num_basis_functions(self) -> int:
        """Return the number of basis functions."""
        return self.mol.nao_nr()


    # =========================================================================
    # Atomic properties
    # =========================================================================

    def get_atomic_coords(self) -> Any:
        """Return current atomic coordinates."""
        return la.asarray(self.mol.atom_coords(unit=self.mol.unit))


    def get_updated_atomic_coords(self, new_coords: Any) -> Any:
        """Return updated atomic coordinates."""
        self.mol.set_geom_(
            atoms_or_coords=la.to_numpy(new_coords),
            unit=self.mol.unit,
            inplace=True
        )
        # PySCF mean-field objects may cache AO two-electron integrals.
        # Those integrals are geometry-dependent and must not survive an MD step.
        if hasattr(self.mf, "_eri"):
            self.mf._eri = None
        for grids in (
            getattr(self.mf, "grids", None),
            getattr(getattr(self, "grad", None), "grids", None),
        ):
            if hasattr(grids, "reset"):
                grids.reset(mol=self.mol)
        return self.get_atomic_coords()


    def get_atomic_masses(self) -> Any:
        """Return an array of atomic masses."""
        return la.array([elements.COMMON_ISOTOPE_MASSES[m]
                        for m in self.mol.atom_charges()]) * nist.AMU2AU


    def compute_ao_slices_by_atom(self) -> Any:
        """Return AO index slices grouped by atom."""
        return gto.aoslice_by_atom(mol=self.mol)


    # =========================================================================
    # Energy calculations
    # =========================================================================

    def get_nuclear_energy(self) -> float:
        """Return current nuclear repulsion energy."""
        return self.mol.energy_nuc()


    def get_electronic_energy(
        self,
        dm,
        h1e,
        vhf,
    ) -> tuple[float, float, float]:
        """Return one- and two-electron contributions to the total energy.

        The PySMD density is converted to PySCF's spin-summed restricted
        density before the PySCF energy call.
        """
        dm_numpy = self._to_pyscf_density(dm)
        h1e_numpy = la.to_numpy(h1e)
        vhf_numpy = la.to_numpy(vhf)

        e_tot, e2 = self.mf.energy_elec(
            dm=dm_numpy,
            h1e=h1e_numpy,
            vhf=vhf_numpy,
        )
        e1 = e_tot - e2

        return e1, e2, e_tot


    # =========================================================================
    # Matrix computations
    # =========================================================================

    def compute_inv_overlap_matrix(
        self,
        tol: float = 1e-14,
        e: Any = None,
        v: Any = None,
    ) -> Any:
        r"""Return the inverse of the overlap matrix in the AO basis.

        Computes :math:`\mathbf{S}^{-1}` after filtering small eigenvalues.

        Parameters
        ----------
        tol : float
            optional, stability threshold; default = 1e-14
        e : np.ndarray, optional
            Pre-computed eigenvalues (already filtered). If not provided,
            will diagonalize overlap matrix.
        v : np.ndarray, optional
            Pre-computed eigenvectors (already filtered). If not provided,
            will diagonalize overlap matrix.

        Returns
        -------
        numpy.ndarray
            Inverse overlap matrix :math:`\mathbf{S}^{-1}`.
        """
        if e is None or v is None:
            e, v = self._diagonalize_overlap(tol)
        return la.matmul(v / e, v.T.conj())


    def compute_sqrt_overlap_matrix(
        self,
        tol: float = 1e-14,
        e: Any = None,
        v: Any = None,
    ) -> Any:
        r"""Return the square root of the overlap matrix in the AO basis.

        Computes :math:`\mathbf{S}^{1/2}` after filtering small eigenvalues.

        Parameters
        ----------
        tol : float
            optional, stability threshold; default = 1e-14
        e : np.ndarray, optional
            Pre-computed eigenvalues (already filtered). If not provided,
            will diagonalize overlap matrix.
        v : np.ndarray, optional
            Pre-computed eigenvectors (already filtered). If not provided,
            will diagonalize overlap matrix.

        Returns
        -------
        numpy.ndarray
            Square root of overlap matrix :math:`\mathbf{S}^{1/2}`.
        """
        if e is None or v is None:
            e, v = self._diagonalize_overlap(tol)
        return la.matmul(v * la.sqrt(e), v.T.conj())


    def compute_inv_sqrt_overlap_matrix(
        self,
        tol: float = 1e-14,
        e: Any = None,
        v: Any = None,
    ) -> Any:
        r"""Return the inverse square root of the overlap matrix in the AO basis.

        Computes :math:`\mathbf{S}^{-1/2}` after filtering small eigenvalues.

        Parameters
        ----------
        tol : float
            optional, stability threshold; default = 1e-14
        e : np.ndarray, optional
            Pre-computed eigenvalues (already filtered). If not provided,
            will diagonalize overlap matrix.
        v : np.ndarray, optional
            Pre-computed eigenvectors (already filtered). If not provided,
            will diagonalize overlap matrix.

        Returns
        -------
        numpy.ndarray
            Inverse square root of overlap matrix :math:`\mathbf{S}^{-1/2}`.
        """
        if e is None or v is None:
            e, v = self._diagonalize_overlap(tol)
        return la.matmul(v / la.sqrt(e), v.T.conj())


    def compute_overlap_matrices(self, tol: float = 1e-14) -> tuple[Any, Any, Any, Any]:
        r"""Return the overlap matrices in the AO basis.

        Parameters
        ----------
        tol : float
            optional, stability threshold; default = 1e-14

        Returns
        -------
        tuple[np.ndarray, ...]
            The tuple ``(S, S_inv, S_sqrt, S_inv_sqrt)`` corresponding to
            :math:`(\mathbf{S}, \mathbf{S}^{-1}, \mathbf{S}^{1/2},
            \mathbf{S}^{-1/2})`, in that order.
        """
        ### Get overlap matrix and diagonalize once
        ovlp = self.compute_overlap_matrix()
        e, v = self._diagonalize_overlap(tol)

        ### Use individual methods with pre-computed eigendecomposition
        sm1 = self.compute_inv_overlap_matrix(tol=tol, e=e, v=v)
        sp12 = self.compute_sqrt_overlap_matrix(tol=tol, e=e, v=v)
        sm12 = self.compute_inv_sqrt_overlap_matrix(tol=tol, e=e, v=v)

        return ovlp, sm1, sp12, sm12


    def compute_coulomb_matrix(
        self,
        dm: Any,
    ) -> Any:
        """Return the Coulomb matrix in the AO basis."""
        result = self.mf.get_j(
            mol=self.mol,
            dm=self._to_pyscf_density(dm),
        )
        return la.asarray(result)


    def compute_exchange_matrix(
        self,
        dm,
    ):
        """Return the exact-exchange matrix in the AO basis.

        Pure local and semi-local density functionals carry no exact-exchange
        contribution, so this default returns a zero matrix. Hartree-Fock-like
        subclasses override it with a genuine exchange build.

        Parameters
        ----------
        dm : np.ndarray
            Density matrix in AO basis with shape (nao, nao)

        Returns
        -------
        np.ndarray
            Zero matrix with shape (nao, nao)
        """
        return la.zeros_like(la.asarray(dm))


    def compute_exchange_gradient(
        self,
        dm,
    ):
        """Return the exact-exchange gradient in the AO basis.

        Pure local and semi-local density functionals carry no exact-exchange
        contribution, so this default returns a zero tensor. Hartree-Fock-like
        subclasses override it with a genuine exchange-gradient build.

        Parameters
        ----------
        dm : np.ndarray
            Density matrix in AO basis with shape (nao, nao)

        Returns
        -------
        np.ndarray
            Zero tensor with shape (3, nao, nao)
        """
        nao = self.mol.nao_nr()
        return la.zeros((3, nao, nao))


    # =========================================================================
    # Per-atom gradient contributions
    # =========================================================================
    # These assemble the shadow-potential force of Eq. 40 of Niklasson,
    # J. Chem. Theory Comput. 2020, 16, 3628. Each returns a (natom, 3) array
    # with the density contraction already performed, because some backends
    # never expose the derivative matrices themselves.

    def compute_one_electron_gradient(self, dm_ao, fock):
        r"""Return the one-electron and Pulay gradient contributions per atom.

        The core-Hamiltonian derivative carries no factor of two and no AO
        slicing because the generator already returns a symmetrized
        derivative. The Pulay term uses
        :math:`\mathbf{S}^{-1}\mathbf{F}\mathbf{D}`, which is the shadow
        generalization of the energy-weighted density matrix.

        Parameters
        ----------
        dm_ao : np.ndarray
            Density matrix in AO basis with shape (nao, nao)
        fock : np.ndarray
            Fock matrix built from the propagated density, AO basis

        Returns
        -------
        np.ndarray
            Gradient contribution with shape (natom, 3)
        """
        hcore_grad = self.compute_1e_gradient()
        s_grad = self.compute_overlap_gradient()

        inv_ovlp = self.compute_inv_overlap_matrix()
        dm_ao_pyscf = 2.0 * dm_ao
        fock_dm = la.matmul(la.matmul(inv_ovlp, fock), dm_ao_pyscf)

        aoslices = self.compute_ao_slices_by_atom()
        total_grad = la.zeros((self.get_num_atoms(), 3))

        for k in range(self.get_num_atoms()):
            p0, p1 = aoslices[k, 2:]

            # One-electron contribution
            total_grad[k] += la.einsum(
                "xij, ij-> x", hcore_grad(k), dm_ao_pyscf, optimize=True
            )

            # Pulay force term
            total_grad[k] -= 2.0 * la.einsum(
                "xij, ij-> x",
                s_grad[:, p0:p1],
                fock_dm[p0:p1],
                optimize=True,
            )

        return total_grad


    def compute_two_electron_gradient(self, dm_bra, dm_ket):
        r"""Return the two-electron gradient contribution per atom.

        Builds the derivative from both densities and contracts each against
        the other, which is the symmetrized form required by Eq. 43 of
        Niklasson, *J. Chem. Theory Comput.* **2020**, *16*, 3628. The
        backend derivative builders differentiate only one AO index, and the
        usual compensating factor of two is valid only when the density
        inside the build equals the density contracted outside it.

        Parameters
        ----------
        dm_bra : np.ndarray
            First density matrix in AO basis with shape (nao, nao)
        dm_ket : np.ndarray
            Second density matrix in AO basis with shape (nao, nao)

        Returns
        -------
        np.ndarray
            Gradient contribution with shape (natom, 3)
        """
        two_e_grad_bra = (
            self.compute_coulomb_gradient(dm=dm_bra)
            - (0.5 * self.compute_exchange_gradient(dm=dm_bra))
        )
        dm_bra_pyscf = 2.0 * dm_bra
        dm_ket_pyscf = 2.0 * dm_ket
        if dm_ket is dm_bra:
            # Self-consistent case; the two builds coincide.
            two_e_grad_ket = two_e_grad_bra
        else:
            two_e_grad_ket = (
                self.compute_coulomb_gradient(dm=dm_ket)
                - (0.5 * self.compute_exchange_gradient(dm=dm_ket))
            )

        aoslices = self.compute_ao_slices_by_atom()
        total_grad = la.zeros((self.get_num_atoms(), 3))

        for k in range(self.get_num_atoms()):
            p0, p1 = aoslices[k, 2:]
            total_grad[k] += la.einsum(
                "xij, ij-> x",
                two_e_grad_bra[:, p0:p1],
                dm_ket_pyscf[p0:p1],
                optimize=True,
            )
            total_grad[k] += la.einsum(
                "xij, ij-> x",
                two_e_grad_ket[:, p0:p1],
                dm_bra_pyscf[p0:p1],
                optimize=True,
            )

        return total_grad


    def compute_xc_gradient_per_atom(self, dm_prop, dm_ao):
        """Return all exchange-correlation gradient contributions per atom.

        Parameters
        ----------
        dm_prop : np.ndarray
            Propagated density matrix in AO basis with shape (nao, nao)
        dm_ao : np.ndarray
            Density matrix in AO basis with shape (nao, nao)

        Returns
        -------
        np.ndarray
            Gradient contribution with shape (natom, 3)
        """
        xc_grad, xc_grid_grad = self.compute_xc_gradient(
            dm_prop=dm_prop,
            dm_ao=dm_ao,
        )

        aoslices = self.compute_ao_slices_by_atom()
        total_grad = la.copy(xc_grid_grad)

        for k in range(self.get_num_atoms()):
            p0, p1 = aoslices[k, 2:]
            total_grad[k] += 2.0 * la.einsum(
                "xij, ij-> x",
                xc_grad[:, p0:p1],
                (2.0 * dm_ao)[p0:p1],
                optimize=True,
            )

        return total_grad
