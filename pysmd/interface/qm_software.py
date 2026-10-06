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

"""Define the generic contract for quantum-chemistry software adapters.

``QMSoftware`` is intentionally package-agnostic and is not a usable
quantum-chemistry implementation. A backend subclass must translate its
electronic-structure objects, basis-matrix operations, energies, and
gradients into the methods declared here. The current PySMD contract uses
atomic-orbital-like basis matrices, but does not require a particular
quantum-chemistry package or array library. The abstract methods raise
``NotImplementedError`` so that an incomplete adapter fails explicitly.

The PySCF implementation is available from :mod:`pysmd.interface.pyscf`.
"""

import abc
from typing import Any


class QMSoftware(abc.ABC):
    """Abstract interface required by PySMD quantum-chemistry backends.

    Implementations provide access to system properties, AO-like basis
    matrices, electronic energies, and the gradient components needed by
    PySMD. Unless a method states otherwise, arrays use the backend's current
    basis and units, and the caller is responsible for preserving the
    backend's array type. The generic class does not prescribe a particular
    software package, storage type, or numerical implementation.

    ``spin_channels`` is one for restricted adapters and two for unrestricted
    adapters, whose densities carry a leading alpha/beta axis. The optional,
    non-abstract hooks below are required only by :mod:`pysmd.open_shell`;
    their defaults raise ``NotImplementedError``.
    """

    spin_channels: int = 1


    def get_spin_electron_counts(self) -> tuple[int, int]:
        """Return fixed alpha/beta populations for an unrestricted reference."""
        raise NotImplementedError("This interface does not provide spin populations.")


    def get_potential_response(self, dm: Any) -> Any:
        """Return an operator applying the potential derivative at ``dm``."""
        raise NotImplementedError("This interface does not provide a response operator.")


    def capture_overlap_basis(self) -> Any:
        """Return an independent basis/geometry snapshot for orbital tracking."""
        raise NotImplementedError("This interface does not provide cross-geometry overlaps.")


    def compute_cross_overlap(self, reference_basis: Any) -> Any:
        """Return ``<current AO | reference AO>`` for a captured basis snapshot."""
        raise NotImplementedError("This interface does not provide cross-geometry overlaps.")


    @abc.abstractmethod
    def __init__(self) -> None:
        """Instantiate instance of QMSoftware class."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_mean_field_type(self) -> str:
        """Return the backend-specific electronic-structure method type."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_atomic_coords(self) -> Any:
        """Return current atomic coordinates."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_updated_atomic_coords(self, new_coords: Any) -> Any:
        r"""Update and return the atomic coordinates.

        Implementations should document whether the underlying molecular
        object is mutated. The PySCF adapter updates its molecule in place.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_atomic_masses(self) -> Any:
        """Return atomic masses in the coordinate and dynamics units."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_num_atoms() -> int:
        """Return the number of atoms."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_num_electrons() -> int:
        """Return the number of electrons."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_num_basis_functions() -> int:
        """Return the number of functions in the one-particle basis."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_nuclear_energy() -> float:
        """Return the nuclear--nuclear repulsion energy."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_electronic_energy(
        self, dm: Any, h1e: Any, vhf: Any
    ) -> tuple[float, float, float]:
        """Return one-electron, two-electron, and total electronic energies.

        The returned tuple is ``(e1, e2, e_electronic)``. The energy units are
        those used by the backend.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def get_scf_convergence(self) -> bool:
        """Return SCF convergence status."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def can_reuse_scf(self, temp: float) -> bool:
        """Return whether the current converged state is reusable at ``temp``."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_overlap_matrix(self) -> Any:
        """Return overlap matrix."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_inv_overlap_matrix(self, tol: float = 1e-14, e: Any = None, v: Any = None) -> Any:
        r"""Return the filtered inverse overlap matrix :math:`\mathbf{S}^{-1}`.

        ``tol`` controls the treatment of small overlap eigenvalues. The
        optional ``e`` and ``v`` arguments may provide a previously computed
        eigendecomposition.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_sqrt_overlap_matrix(self, tol: float = 1e-14, e: Any = None, v: Any = None) -> Any:
        r"""Return the filtered overlap square root :math:`\mathbf{S}^{1/2}`."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_inv_sqrt_overlap_matrix(self, tol: float = 1e-14, e: Any = None, v: Any = None) -> Any:
        r"""Return the filtered inverse square root :math:`\mathbf{S}^{-1/2}`."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_overlap_matrices(self, tol: float = 1e-14) -> tuple[Any, Any, Any, Any]:
        r"""Return ``(S, S^{-1}, S^{1/2}, S^{-1/2})`` in that order."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_guess_density_matrix(self, mo_coeff: Any = None, mo_occ: Any = None) -> Any:
        """Return an initial AO-like, single-particle density matrix guess.

        For restricted closed-shell adapters, occupations supplied by PySMD
        use the spatial-orbital convention ``[0, 1]``. Any backend-specific
        occupation conversion belongs inside the concrete adapter and must not
        change the PySMD-facing convention.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_density_matrix(self, mo_coeff: Any = None, mo_occ: Any = None) -> Any:
        """Return the current single-particle density matrix in the basis contract.

        Occupations supplied by PySMD use the spatial-orbital convention
        ``[0, 1]``. Conversion to a backend-specific convention is the
        responsibility of the concrete adapter.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_core_hamiltonian_matrix(self) -> Any:
        """Return the one-electron core Hamiltonian matrix."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_eff_potential_matrix(self, dm: Any) -> Any:
        """Return the method-dependent effective potential matrix for ``dm``."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_coulomb_matrix(self, dm: Any) -> Any:
        """Return the Coulomb contribution generated from ``dm``."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_exchange_matrix(self, dm: Any) -> Any:
        """Return exact-exchange matrix.

        Implementations without exact exchange may return a zero matrix.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_xc_terms(self, dm: Any) -> tuple[Any, Any]:
        """Return the backend-defined exchange-correlation energy and potential."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_nuclear_gradient(self) -> Any:
        """Return the nuclear--nuclear contribution to the energy gradient."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_overlap_gradient(self) -> Any:
        """Return derivatives of the basis-overlap matrix."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_1e_gradient(self) -> Any:
        """Return the one-electron derivative representation used by the backend."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_coulomb_gradient(self, dm: Any) -> Any:
        """Return the Coulomb contribution to the electronic gradient."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_exchange_gradient(self, dm: Any) -> Any:
        """Return exact-exchange component of electronic gradient.

        Implementations without exact exchange may return a zero tensor.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_xc_gradient(self, dm_prop: Any, dm_ao: Any) -> tuple[Any, Any]:
        """Return backend-defined exchange-correlation gradient components."""
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_one_electron_gradient(self, dm_ao: Any, fock: Any) -> Any:
        """Return contracted one-electron and basis-response gradients per atom.

        The returned quantity is an energy gradient, not a force. The
        implementation may combine several one-electron and basis-response
        terms because some backends expose them only as a contracted
        per-atom quantity.

        Parameters
        ----------
        dm_ao : array
            Density matrix in the AO-like basis.
        fock : array
            Fock-like matrix used by the backend, in the AO-like basis.

        Returns
        -------
        array
            Gradient contribution with shape ``(natom, 3)``.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_two_electron_gradient(self, dm_bra: Any, dm_ket: Any) -> Any:
        """Return the two-electron gradient contribution per atom.

        The implementation must use both supplied density representations and
        return the symmetric bilinear result expected by the caller. The
        backend may construct this result from one-sided derivative tensors,
        but must account for their index asymmetry rather than applying a
        factor of two indiscriminately.

        Parameters
        ----------
        dm_bra : array
            First density matrix, AO-like basis.
        dm_ket : array
            Second density matrix, AO-like basis.

        Returns
        -------
        array
            Gradient contribution with shape ``(natom, 3)``.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )


    @abc.abstractmethod
    def compute_xc_gradient_per_atom(self, dm_prop: Any, dm_ao: Any) -> Any:
        """Return contracted exchange-correlation gradient contributions per atom.

        Implementations without an exchange-correlation functional may return
        zeros. The meaning of ``dm_prop`` and ``dm_ao`` is backend-dependent
        and should be documented by the concrete implementation.

        Parameters
        ----------
        dm_prop : array
            First density representation, AO-like basis.
        dm_ao : array
            Second density representation, AO-like basis.

        Returns
        -------
        array
            Gradient contribution with shape ``(natom, 3)``.
        """
        raise NotImplementedError(
            "Subclasses of the base class must implement this method"
        )
