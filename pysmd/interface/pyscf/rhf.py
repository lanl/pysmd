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

"""PySMD adapter for PySCF restricted Hartree--Fock calculations.

Standard RHF quantities are obtained by direct calls to PySCF's RHF and
gradient APIs. The adapter adds PySMD array-backend conversion and exposes
Coulomb and exchange gradient terms separately because the shadow-MD driver
needs them independently; PySCF's ordinary RHF effective potential combines
those terms.

PySMD uses spatial-orbital occupations in the range ``[0, 1]``. PySCF's
restricted RHF routines use spin-summed occupations in the range ``[0, 2]``.
Occupation conversion belongs at this adapter boundary: occupations supplied
by PySMD are multiplied by two before being passed to PySCF, while occupations
read from PySCF are divided by two before being used or returned to PySMD.
The same boundary conversion is applied to the corresponding density matrices
when PySMD calls PySCF routines.
"""

from typing import Any, Callable

from pyscf import grad, gto, scf

from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.interface.pyscf.base import PyscfBase

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()


class RHF(PyscfBase):
    """Interface class for PySCF restricted Hartree-Fock (RHF) calculations.

    This class provides an interface to PySCF's RHF implementation for
    closed-shell systems. It inherits common functionality from PyscfBase
    and implements RHF-specific energy and gradient operations.

    The density-matrix construction and core-Hamiltonian/overlap helpers call
    the corresponding PySCF functions directly. The separate gradient methods
    are thin PySMD-facing wrappers around PySCF gradient routines, with array
    conversion and the decomposition needed by the shadow-MD equations.

    The class can be initialized with either a PySCF molecule object or a
    pre-configured RHF mean-field object.
    """

    def __init__(
        self,
        pyscf_mol: gto.MoleBase | None = None,
        pyscf_mf: scf.hf.RHF | None = None,
    ) -> None:
        """Initialize RHF interface with PySCF objects.

        Parameters
        ----------
        pyscf_mol : gto.MoleBase, optional
            PySCF molecule object containing atomic structure and basis set.
            Used to create an RHF mean-field object if pyscf_mf is not provided.
        pyscf_mf : scf.hf.RHF, optional
            PySCF RHF mean-field object (may be pre-converged or unconverged).
            If provided, takes precedence over pyscf_mol.

        Raises
        ------
        TypeError
            If neither pyscf_mol nor pyscf_mf is provided, or if provided
            objects are not of the correct type

        Notes
        -----
        - If pyscf_mf is provided, it is used directly
        - If only pyscf_mol is provided, an RHF mean-field object is created from it
        - The molecule object is automatically converted to atomic units if needed
        - A gradient object is created for force calculations
        """
        log.info(">> Instantiating PySCF-backend RHF interface...")

        ### Mean-field object provided
        if pyscf_mf is not None:
            # Validate that it's specifically an RHF mean-field object
            if (
                not isinstance(pyscf_mf, scf.hf.RHF)
                or isinstance(pyscf_mf, scf.rohf.ROHF)
            ):
                raise TypeError(f"pyscf_mf must be an instance of scf.hf.RHF, "
                                f"got {type(pyscf_mf).__name__}")
            mf = pyscf_mf
            mf.verbose = mf.mol.verbose = 0
            log.info(f" -- Detected PySCF RHF mean-field object: {mf.__class__}")

        ### Molecule object provided - create RHF mean-field
        elif pyscf_mol is not None:
            # Validate that it's a molecule object
            if not isinstance(pyscf_mol, gto.MoleBase):
                raise TypeError(f"pyscf_mol must be an instance of gto.MoleBase, "
                                f"got {type(pyscf_mol).__name__}")
            mol = pyscf_mol
            mol.verbose = 0
            log.info(f" -- Detected PySCF molecule object: {mol.__class__}")
            log.info(" -- Creating RHF mean-field object...")

            # Create RHF mean-field object
            mf = scf.hf.RHF(mol=mol)
            mf.verbose = 0

        ### No objects provided
        else:
            log.error("Must provide either pyscf_mf (scf.hf.RHF) "
                      "or pyscf_mol (gto.MoleBase).")
            raise TypeError("Must provide either pyscf_mf (scf.hf.RHF) "
                            "or pyscf_mol (gto.MoleBase).")

        ### Verify molecule object is defined in atomic units
        if hasattr(mf.mol, "unit") and not gto.is_au(mf.mol.unit):
            mf.mol.unit = "au"
            mf.mol.set_geom_(
                atoms_or_coords=mf.mol.atom_coords(unit=mf.mol.unit),
                unit=mf.mol.unit,
                inplace=True
            )

        ### Initialize base class with mean-field object only
        super().__init__(pyscf_mf=mf)

        ### Create gradient object
        self.grad = grad.rhf.Gradients(method=self.mf)

        return


    def compute_overlap_matrix(self) -> Any:
        r"""Compute overlap matrix in atomic orbital (AO) basis.

        Returns
        -------
        np.ndarray
            Overlap matrix :math:`\mathbf{S}` with shape ``(nao, nao)``.
        """
        return la.asarray(scf.hf.get_ovlp(mol=self.mol))


    def compute_density_matrix(
        self,
        mo_coeff: Any = None,
        mo_occ: Any = None,
    ) -> Any:
        r"""Compute the closed-shell density matrix in the AO basis.

                Constructs the density matrix from molecular orbital coefficients
                and occupations using

                .. math::

              \mathbf{D}
              = \mathbf{C}\,\operatorname{diag}(\mathbf{n})\,\mathbf{C}^{T}.

                Here ``D`` is the real density matrix formed with PySMD spatial-orbital
                occupations. PySCF's restricted convention is spin-summed: its
                occupations range from 0 to 2 and its corresponding density is
                twice the PySMD spatial-orbital density.

        Parameters
        ----------
        mo_coeff : np.ndarray, optional
            Molecular orbital coefficients with shape (nao, nmo). Defaults to
            the coefficients stored on the mean-field object.
        mo_occ : np.ndarray, optional
            Molecular orbital occupations with shape (nmo). Values supplied by
            PySMD use the spatial-orbital convention ``[0, 1]``. Occupations
            stored on a PySCF restricted mean-field object use ``[0, 2]``;
            these are divided by two before use in the PySMD density.

        Returns
        -------
        np.ndarray
            Density matrix in AO basis with shape (nao, nao)
        """
        if mo_coeff is None: mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_occ is None:
            mo_occ = getattr(self.mf, "mo_occ", None)
            if mo_occ is not None:
                mo_occ = 0.5 * la.asarray(mo_occ)
        if mo_coeff is None or mo_occ is None:
            raise ValueError(
                "Molecular orbital coefficients and occupations are unavailable."
            )

        result = scf.hf.make_rdm1(
            mo_coeff=la.to_numpy(mo_coeff),
            mo_occ=2.0 * la.to_numpy(mo_occ),
        )
        return self._from_pyscf_density(result)


    def compute_guess_density_matrix(
        self,
        mo_coeff: Any = None,
        mo_occ: Any = None,
    ) -> Any:
        """Compute initial guess density matrix in AO basis.

        If molecular orbital coefficients and occupations are provided (or can
        be retrieved from the mean-field object), constructs the density matrix
        from them. Otherwise, generates an initial guess using the method
        specified in the mean-field object (default: 'minao').

        Parameters
        ----------
        mo_coeff : np.ndarray, optional
            Molecular orbital coefficients. If None, attempts to retrieve
            from mean-field object.
        mo_occ : np.ndarray, optional
            PySMD spatial-orbital occupations in ``[0, 1]``. If None, the
            occupation data are retrieved from the mean-field object or an
            initial guess is generated. Occupations supplied to PySCF are
            converted to its spin-summed ``[0, 2]`` convention internally.

        Returns
        -------
        np.ndarray
            Initial guess density matrix in AO basis with shape (nao, nao)
        """
        # Use passed parameters if provided, otherwise try to get from self.mf
        if mo_coeff is None: mo_coeff = getattr(self.mf, "mo_coeff", None)
        # Return initial guess if no MO coefficients/occupations available
        if mo_coeff is None or mo_occ is None:
            init_guess: str = getattr(self.mf, "init_guess", "minao")
            return self._from_pyscf_density(
                self.mf.get_init_guess(mol=self.mol, key=init_guess)
            )

        # Return current density matrix
        else:
            return self.compute_density_matrix(mo_coeff=mo_coeff, mo_occ=mo_occ)


    def compute_core_hamiltonian_matrix(self) -> Any:
        """Compute core Hamiltonian matrix in AO basis.

        The core Hamiltonian includes kinetic energy and nuclear-electron
        attraction terms.

        Returns
        -------
        np.ndarray
            Core Hamiltonian matrix with shape (nao, nao)
        """
        return la.asarray(scf.hf.get_hcore(mol=self.mol))


    def compute_eff_potential_matrix(
        self,
        dm: Any,
    ) -> Any:
        r"""Compute the RHF effective potential matrix in the AO basis.

        The RHF effective potential is

        .. math::

           \mathbf{V}_{\mathrm{eff}} = 2\mathbf{J} - \mathbf{K},

        where :math:`\mathbf{J}` and :math:`\mathbf{K}` are the Coulomb and
        exact-exchange matrices generated from the supplied density matrix.

        Parameters
        ----------
        dm : np.ndarray
            Density matrix in AO basis with shape (nao, nao)

        Returns
        -------
        np.ndarray
            Effective potential matrix with shape (nao, nao)
        """
        dm_numpy = self._to_pyscf_density(dm)
        result = self.mf.get_veff(
            mol=self.mol,
            dm=dm_numpy,
        )
        return la.asarray(result)


    def compute_exchange_matrix(
        self,
        dm: Any,
    ) -> Any:
        """Compute the RHF exchange matrix in the AO basis."""
        result = self.mf.get_k(
            mol=self.mol,
            dm=self._to_pyscf_density(dm),
        )
        return la.asarray(result)


    def compute_xc_terms(
        self,
        dm: Any,
    ) -> tuple[float, Any]:
        """Return RHF exchange-correlation scalar and potential matrix (zeros)."""
        return 0.0, la.zeros_like(dm)


    def compute_nuclear_gradient(self) -> Any:
        """Compute nuclear repulsion gradient.

        Returns
        -------
        np.ndarray
            Nuclear gradient with shape (natom, 3)
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")
        return la.asarray(self.grad.grad_nuc(mol=self.mol))


    def compute_overlap_gradient(self) -> Any:
        """Compute overlap matrix gradient.

        Returns
        -------
        np.ndarray
            Overlap gradient with shape (3, nao, nao)
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")
        return la.asarray(self.grad.get_ovlp(mol=self.mol))


    def compute_1e_gradient(self) -> Callable[[int], Any]:
        """Compute one-electron (core Hamiltonian) gradient generator.

        Returns a function that generates the core Hamiltonian gradient
        for a specific atom.

        Returns
        -------
        callable
            Function that takes an atom index and returns the gradient
            with shape (3, nao, nao)
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")
        generator = self.grad.hcore_generator(mol=self.mol)

        def backend_generator(atom_index):
            return la.asarray(generator(atom_index))

        return backend_generator


    def compute_coulomb_gradient(self, dm: Any) -> Any:
        """Compute Coulomb part of the two-electron gradient.

        This exposes PySCF's ``grad.rhf.Gradients.get_j`` result directly.
        PySCF's ordinary RHF ``get_veff`` combines this term with exchange;
        the shadow-gradient code needs the two pieces separately.
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")
        result = self.grad.get_j(mol=self.mol, dm=self._to_pyscf_density(dm))
        return la.asarray(result)


    def compute_exchange_gradient(self, dm: Any) -> Any:
        """Compute exchange part of the two-electron gradient.

        This exposes PySCF's ``grad.rhf.Gradients.get_k`` result directly.
        The calling code combines it as ``J' - 0.5 * K'``, matching PySCF's
        RHF gradient effective-potential convention.
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")
        result = self.grad.get_k(mol=self.mol, dm=self._to_pyscf_density(dm))
        return la.asarray(result)


    def compute_xc_gradient(
        self,
        dm_prop: Any,
        dm_ao: Any,
    ) -> tuple[Any, Any]:
        """Return zero exchange-correlation gradient components for RHF.

        RHF has no DFT XC contribution. This method exists only to keep the
        interface compatible with RKS, where the XC gradient is separated from
        Coulomb and exchange.
        """
        nao = self.mol.nao_nr()
        return la.zeros((3, nao, nao)), la.zeros((self.get_num_atoms(), 3))
