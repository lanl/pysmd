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

"""PySMD adapter for PySCF restricted Kohn--Sham calculations.

Support is deliberately restricted to pure local and semi-local exchange-correlation
functionals, i.e. the first two rungs of the Jacob's ladder classification: local
density approximations (LDA) and generalized gradient approximations (GGA).

Functionals outside that set raise :exc:`NotImplementedError` when the interface is
constructed. In particular there is no support for exact or hybrid exchange, for
range-separated hybrids, for meta-GGAs, or for non-local correlation. Pure LDA and
GGA functionals carry no exact-exchange contribution, so the exchange matrix and
exchange gradient are identically zero here and are inherited from
:class:`~pysmd.interface.pyscf.base.PyscfBase`.

The standard overlap, density, core-Hamiltonian, and effective-potential
operations call PySCF directly. The XC helper is adapted from the quadrature
work performed inside PySCF's RKS effective-potential implementation: it calls
``NumInt.nr_rks`` directly so PySMD can retain the exchange-correlation energy
and potential separately for its linearized gradient terms.

PySMD uses spatial-orbital occupations in the range ``[0, 1]``. PySCF's
restricted RKS routines use spin-summed occupations in the range ``[0, 2]``.
Occupation conversion belongs at this adapter boundary: occupations supplied
by PySMD are multiplied by two before being passed to PySCF, while occupations
read from PySCF are divided by two before being used or returned to PySMD.
The same boundary conversion is applied to the corresponding density matrices
when PySMD calls PySCF routines.
"""

from typing import Any, Callable

from pyscf import dft, grad, gto, scf
from pyscf.dft import gen_grid

from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.interface.pyscf.base import PyscfBase

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

### Exchange-correlation functional types supported by this interface.
# These correspond to rungs 1 and 2 of the Jacob's ladder classification.
SUPPORTED_XC_TYPES: tuple[str, ...] = ("LDA", "GGA")

### Default exchange-correlation functional.
# Spelled with literal libxc keys rather than relying on PySCF's fuzzy alias
# resolution: 'VWN' has no literal entry in PySCF's XC_CODES table and resolves
# to LDA_C_VWN (libxc id 7), which is the VWN5 parameterization. 'LDA,VWN5' names
# the same pair of functionals explicitly.
DEFAULT_XC: str = "LDA,VWN5"


def verify_supported_xc(xc: str, mf: Any) -> str:
    """Verify that an exchange-correlation functional is supported.

    Support is restricted to pure local and semi-local functionals, i.e. the
    first two rungs of the Jacob's ladder classification.

    This is a module-level function rather than a method because every
    Kohn-Sham interface, on any compute backend, must apply exactly the same
    restriction. Duplicating the checks per backend is what allows them to
    drift apart, which is the failure mode this gate exists to prevent.

    Parameters
    ----------
    xc : str
        Exchange-correlation functional string to classify.
    mf : pyscf.dft.rks.RKS or gpu4pyscf.dft.rks.RKS
        Mean-field object carrying the functional. Required because the
        non-local correlation test is a method on the mean-field object.

    Returns
    -------
    str
        The functional type, either ``"LDA"`` or ``"GGA"``.

    Raises
    ------
    ValueError
        If the exchange-correlation functional string cannot be parsed.
    NotImplementedError
        If the functional is not a pure LDA or GGA functional.
    """
    ### Classify the functional
    # PySCF raises a variety of exception types for unrecognized names, so they
    # are funnelled into a single ValueError.
    try:
        xc_type = dft.libxc.xc_type(xc)
    except (KeyError, ValueError, NotImplementedError) as error:
        log.error(f"Unrecognized XC functional string: {xc!r}")
        raise ValueError(
            f"Unrecognized XC functional string: {xc!r}"
        ) from error

    # Pure Hartree-Fock is not a density functional; RHF handles that case.
    if xc_type == "HF":
        raise NotImplementedError(
            f"XC functional {xc!r} is pure Hartree-Fock exchange. "
            f"Use the RHF interface for Hartree-Fock calculations."
        )

    # Reject meta-GGAs (rung 3) and anything PySCF cannot classify.
    if xc_type not in SUPPORTED_XC_TYPES:
        raise NotImplementedError(
            f"XC functional {xc!r} has type {xc_type!r}, which is not "
            f"supported. Supported types are {SUPPORTED_XC_TYPES}, i.e. "
            f"pure LDA and GGA functionals."
        )

    # A single check covers both global hybrids (nonzero hybrid coefficient)
    # and range-separated hybrids (nonzero range-separation coefficients).
    if dft.libxc.is_hybrid_xc(xc):
        raise NotImplementedError(
            f"XC functional {xc!r} includes exact exchange. Hybrid and "
            f"range-separated functionals are not supported; use a pure "
            f"LDA or GGA functional."
        )

    # Non-local correlation (e.g. VV10) is not supported.
    if mf.do_nlc():
        raise NotImplementedError(
            f"XC functional {xc!r} includes non-local correlation, "
            f"which is not supported."
        )

    log.info(f" -- Verified XC functional {xc!r} of type {xc_type!r}.")
    return xc_type


class RKS(PyscfBase):
    """Interface class for PySCF restricted Kohn-Sham (RKS) calculations.

    This class provides an interface to PySCF's RKS implementation for
    closed-shell systems using density functional theory. It inherits common
    functionality from PyscfBase and implements RKS-specific methods for
    energy and gradient calculations.

    The class can be initialized with either a PySCF molecule object or a
    pre-configured RKS mean-field object.

    Standard quantities are direct PySCF calls with conversion to the active
    PySMD array backend. The additional XC and gradient methods expose the
    separated terms required by PySMD; they are not replacements for PySCF's
    ordinary RKS energy and gradient APIs. Only pure LDA and GGA functionals
    are supported; see the module docstring.
    Requesting any other class of functional raises :exc:`NotImplementedError`.
    """

    def __init__(
        self,
        pyscf_mol: gto.MoleBase | None = None,
        pyscf_mf: dft.rks.RKS | None = None,
        xc: str = DEFAULT_XC,
    ) -> None:
        """Initialize RKS interface with PySCF objects.

        Parameters
        ----------
        pyscf_mol : gto.MoleBase, optional
            PySCF molecule object containing atomic structure and basis set.
            Used to create an RKS mean-field object if pyscf_mf is not provided.
        pyscf_mf : dft.rks.RKS, optional
            PySCF RKS mean-field object (may be pre-converged or unconverged).
            If provided, takes precedence over pyscf_mol.
        xc : str, default="LDA,VWN5"
            Exchange-correlation functional to use if creating a new RKS object.
            Ignored if pyscf_mf is provided.

        Raises
        ------
        TypeError
            If neither pyscf_mol nor pyscf_mf is provided, or if provided
            objects are not of the correct type
        ValueError
            If the exchange-correlation functional string cannot be parsed
        NotImplementedError
            If the exchange-correlation functional is not a pure LDA or GGA
            functional

        Notes
        -----
        - If pyscf_mf is provided, it is used directly
        - If only pyscf_mol is provided, an RKS mean-field object is created from it
        - The molecule object is automatically converted to atomic units if needed
        - A gradient object is created for force calculations
        - The functional is validated once, here. It is assumed to remain fixed
          for the lifetime of the interface object and therefore for the whole
          calculation.
        """
        log.info(">> Instantiating PySCF-backend RKS interface...")

        ### Mean-field object provided
        if pyscf_mf is not None:
            # Validate that it's specifically an RKS mean-field object
            if not isinstance(pyscf_mf, dft.rks.RKS):
                raise TypeError(f"pyscf_mf must be an instance of dft.rks.RKS, "
                                f"got {type(pyscf_mf).__name__}")
            mf = pyscf_mf
            mf.verbose = mf.mol.verbose = 0
            log.info(f" -- Detected PySCF RKS mean-field object: {mf.__class__}")

        ### Molecule object provided - create RKS mean-field
        elif pyscf_mol is not None:
            # Validate that it's a molecule object
            if not isinstance(pyscf_mol, gto.MoleBase):
                raise TypeError(f"pyscf_mol must be an instance of gto.MoleBase, "
                                f"got {type(pyscf_mol).__name__}")
            mol = pyscf_mol
            mol.verbose = 0
            log.info(f" -- Detected PySCF molecule object: {mol.__class__}")
            log.info(f" -- Creating RKS mean-field object with XC functional: {xc}")

            # Create RKS mean-field object
            mf = dft.rks.RKS(mol=mol, xc=xc)
            mf.verbose = 0

        ### No objects provided
        else:
            log.error("Must provide either pyscf_mf (dft.rks.RKS) "
                      "or pyscf_mol (gto.MoleBase).")
            raise TypeError("Must provide either pyscf_mf (dft.rks.RKS) "
                            "or pyscf_mol (gto.MoleBase).")

        ### Verify the requested functional is supported
        verify_supported_xc(mf.xc, mf)
        self.xc: str = mf.xc

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
        # Note: grad.rks.Gradients uses 'mf' parameter (unlike grad.rhf.Gradients which uses 'method')
        self.grad = grad.rks.Gradients(mf=self.mf)

        ### Grid response is always included
        # A gradient evaluated on a fixed quadrature grid is not the exact
        # derivative of the quadrature-approximated energy, so such forces are
        # not conservative and are unsuitable for molecular dynamics.
        self.grad.grid_response = True

        return


    def compute_overlap_matrix(self) -> Any:
        """Compute overlap matrix in atomic orbital (AO) basis.
        
        Note: Uses scf.hf.get_ovlp as there is no DFT-specific overlap matrix.

        Returns
        -------
        np.ndarray
            Overlap matrix S with shape (nao, nao)
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

                This is the real density matrix formed with PySMD spatial-orbital
                occupations. PySCF's restricted convention is spin-summed: its
                occupations range from 0 to 2 and its corresponding density is
                twice the PySMD spatial-orbital density.
        
        Note: Uses scf.hf.make_rdm1 as the density matrix construction is
        identical for RHF and RKS (DFT differences are in the potential).

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
        if mo_coeff is None:
            mo_coeff = getattr(self.mf, "mo_coeff", None)
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
        
        Note: Uses scf.hf.get_hcore as the core Hamiltonian is identical
        for RHF and RKS (DFT differences are in the potential, not the core).

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
        """Compute effective potential matrix in AO basis.

        Constructs the effective potential (Veff) including Coulomb (J),
        exchange (K), and exchange-correlation (XC) contributions for RKS.
        
        Note: Uses self.mf.get_veff() which calls the DFT-specific version
        that includes exchange-correlation contributions from the functional.

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


    def compute_xc_terms(
        self,
        dm: Any,
    ) -> tuple[Any, Any]:
        """Return XC quadrature outputs without Coulomb or exact exchange.

        The scalar ``exc`` and matrix ``vxc`` are returned from PySCF's
        quadrature evaluation. The caller uses ``vxc`` in the linearized XC
        matrix contraction with the difference between the target and
        propagated density matrices.
        """
        ni = self.mf._numint
        grids = self.mf.grids

        if not hasattr(grids, "coords") or grids.coords is None:
            grids.build()

        _, exc, vxc = ni.nr_rks(
            self.mol,
            grids,
            self.mf.xc,
            self._to_pyscf_density(dm),
            max_memory=self.mf.max_memory,
            verbose=self.mf.verbose
        )

        return exc, la.asarray(vxc)


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

        This is the ``ks_grad.get_j(mol, dm)`` Coulomb term used inside
        PySCF's :func:`pyscf.grad.rks.get_veff`, exposed separately so
        ``ShadowMD.update_gradients`` can contract Coulomb terms with the
        linearized density instead of with the density used for XC terms.
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")
        result = self.grad.get_j(mol=self.mol, dm=self._to_pyscf_density(dm))
        return la.asarray(result)


    def compute_xc_gradient(
        self,
        dm_prop: Any,
        dm_ao: Any,
    ) -> tuple[Any, Any]:
        r"""Return the linearized XC matrix derivative and grid-response gradient.

        Adapted from PySCF 2.11's
        :func:`pyscf.grad.rks.get_vxc_full_response`. The loop structure, AO
        derivative order, calls to ``eval_xc_eff``, and low-level AO
        contraction helpers are intentionally kept close to PySCF.

        The modification is the linearized shadow functional,

        .. math::

            E_{xc}^{lin} = E_{xc}[\rho_{prop}]
                + \int v_{xc}[\rho_{prop}]\,(\rho_{ao} - \rho_{prop}),

        so ``dm_prop`` builds ``rho``, ``vxc`` and ``fxc`` while ``dm_ao`` is
        the target density. Differentiating gives two contributions,

        .. math::

            \int v_{xc}[\rho_{prop}]\,\partial_{R}\rho_{ao}
            \;+\;
            \int f_{xc}[\rho_{prop}]\,(\rho_{ao} - \rho_{prop})\,
                 \partial_{R}\rho_{prop},

        plus the response of the quadrature weights, which uses the
        corresponding first-order energy density. The first term is returned
        as a matrix derivative for the caller to contract with ``dm_ao``; the
        second is already contracted with ``dm_prop`` and is folded into the
        second return value alongside the weight response.

        Because the exchange-correlation potential is a local multiplicative
        function on the quadrature grid, only the bra/ket AO-pair symmetry is
        involved and the usual factor of two is valid here. This is in
        contrast to the two-electron Coulomb term, which requires an explicit
        symmetrization when the propagated and linearized densities differ.

        Grid response is always included, so the returned gradient is the
        exact derivative of the quadrature-approximated energy and the
        resulting forces are conservative.

        Parameters
        ----------
        dm_prop : np.ndarray
            Propagated density matrix in AO basis with shape (nao, nao).
        dm_ao : np.ndarray
            Density matrix in AO basis with shape (nao, nao).

        Returns
        -------
        tuple of np.ndarray
            The XC matrix derivative with shape (3, nao, nao), to be
            contracted with ``dm_ao``, and the per-atom gradient
            contribution with shape (natom, 3) collecting the XC-kernel and
            grid-weight response terms.
        """
        if not self.grad:
            log.error("Gradient object is not defined.")
            raise RuntimeError("Gradient object is not initialized.")

        ni = self.mf._numint
        xctype = ni._xc_type(self.mf.xc)

        # The constructor restricts the functional to LDA or GGA; this guards
        # against the invariant being broken downstream.
        if xctype not in SUPPORTED_XC_TYPES:
            raise NotImplementedError(
                f"Linearized RKS gradients are not implemented for XC type "
                f"{xctype!r}."
            )

        grids = self.grad.grids if self.grad.grids is not None else self.mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)

        dm_prop_numpy = self._to_pyscf_density(dm_prop)
        dm_ao_numpy = self._to_pyscf_density(dm_ao)
        make_rho_prop, nset_prop, nao = ni._gen_rho_evaluator(
            self.mol, dm_prop_numpy, 1, False, grids
        )
        make_rho_ao, nset_ao, _ = ni._gen_rho_evaluator(
            self.mol, dm_ao_numpy, 1, False, grids
        )
        if nset_prop != 1 or nset_ao != 1:
            raise ValueError("Linearized RKS gradients require one density matrix.")

        ao_loc = self.mol.ao_loc_nr()
        ao_deriv = 1 if xctype == "LDA" else 2

        exc_grid = la.zeros((self.mol.natm, 3))
        vmat = la.zeros((3, nao, nao))
        kernel_vmat = la.zeros((3, nao, nao))

        for atm_id, (coords, weight, weight1) in enumerate(
            grad.rks.grids_response_cc(grids)
        ):
            mask = gen_grid.make_mask(self.mol, coords)
            ao = ni.eval_ao(
                self.mol,
                coords,
                deriv=ao_deriv,
                non0tab=mask,
                cutoff=grids.cutoff,
            )
            ao_values = ao[0] if xctype == "LDA" else ao[:4]
            rho_prop = make_rho_prop(0, ao_values, mask, xctype)
            rho_ao = make_rho_ao(0, ao_values, mask, xctype)
            exc, vxc, fxc = ni.eval_xc_eff(
                self.mf.xc, rho_prop, 2, xctype=xctype
            )[:3]
            delta_rho = rho_ao - rho_prop

            vtmp = la.to_numpy(la.zeros((3, nao, nao)))
            kernel_vtmp = la.to_numpy(la.zeros((3, nao, nao)))

            if xctype == "LDA":
                aow = grad.rks.numint._scale_ao(ao[0], weight * vxc[0])
                grad.rks._d1_dot_(
                    vtmp, self.mol, ao[1:4], aow, mask, ao_loc, True
                )

                kernel_vxc = fxc[0, 0] * delta_rho
                kernel_aow = grad.rks.numint._scale_ao(
                    ao[0], weight * kernel_vxc
                )
                grad.rks._d1_dot_(
                    kernel_vtmp,
                    self.mol,
                    ao[1:4],
                    kernel_aow,
                    mask,
                    ao_loc,
                    True,
                )

                exc_lin = exc * rho_prop + vxc[0] * delta_rho

            else:
                wv = weight * vxc
                wv[0] *= 0.5
                grad.rks._gga_grad_sum_(vtmp, self.mol, ao, wv, mask, ao_loc)

                kernel_vxc = la.to_numpy(
                    la.einsum("abr,br->ar", fxc, delta_rho, optimize=True)
                )
                kernel_wv = weight * kernel_vxc
                kernel_wv[0] *= 0.5
                grad.rks._gga_grad_sum_(
                    kernel_vtmp, self.mol, ao, kernel_wv, mask, ao_loc
                )
                exc_lin = (
                    la.asarray(exc * rho_prop[0])
                    + la.einsum("nr,nr->r", vxc, delta_rho, optimize=True)
                )

            vmat += la.asarray(vtmp)
            kernel_vmat += la.asarray(kernel_vtmp)

            ### Response of the quadrature weights and grid coordinates
            exc_grid += la.einsum("r,nxr->nx", exc_lin, weight1, optimize=True)
            exc_grid[atm_id] += (
                4.0 * la.einsum("xij,ji->x", vtmp, dm_ao, optimize=True)
                + 4.0 * la.einsum(
                    "xij,ji->x", kernel_vtmp, dm_prop, optimize=True
                )
            )

        ### XC-kernel contribution, contracted with the propagated density
        aoslices = self.compute_ao_slices_by_atom()
        for atom_index in range(self.get_num_atoms()):
            p0, p1 = aoslices[atom_index, 2:]
            exc_grid[atom_index] -= 4.0 * la.einsum(
                "xij,ij->x",
                kernel_vmat[:, p0:p1],
                dm_prop[p0:p1],
                optimize=True,
            )

        return -vmat, exc_grid
