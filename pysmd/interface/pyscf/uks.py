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

"""PySMD adapter for PySCF collinear unrestricted Kohn--Sham calculations.

Supported functionals are local (LDA), semi-local (GGA), and meta-GGA
exchange-correlation functionals, optionally combined with global-hybrid or
range-separated exact exchange. Non-local correlation (e.g. VV10) and
empirical dispersion corrections are rejected with
:exc:`NotImplementedError`. This is broader than the restricted
:class:`~pysmd.interface.pyscf.rks.RKS` adapter, which supports pure LDA and
GGA functionals only.

Exact exchange is bilinear in the density, like the Coulomb term, so its
shadow gradient uses the same symmetrized contraction as UHF. The semilocal
part uses a linearized, spin-polarized exchange-correlation gradient adapted
from PySCF's :func:`pyscf.grad.uks.get_vxc_full_response`; see
:meth:`UKS.compute_xc_gradient`. Grid response is always included.
"""

from typing import Any, Callable

import numpy as np
from pyscf import dft, grad, gto
from pyscf.dft import gen_grid

from pysmd.common import logger
log = logger.getLogger(__name__)

from pysmd.interface.pyscf.rks import DEFAULT_XC
from pysmd.interface.pyscf.uhf import UHF

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

### Exchange-correlation functional types supported by this interface.
SUPPORTED_UNRESTRICTED_XC_TYPES: tuple[str, ...] = ("LDA", "GGA", "MGGA")


def verify_supported_unrestricted_xc(xc: str, mf: Any) -> str:
    """Verify that a functional is supported by the unrestricted interface.

    Parameters
    ----------
    xc : str
        Exchange-correlation functional string to classify.
    mf : pyscf.dft.uks.UKS
        Mean-field object carrying the functional.

    Returns
    -------
    str
        The functional type: ``"LDA"``, ``"GGA"``, or ``"MGGA"``.

    Raises
    ------
    ValueError
        If the exchange-correlation functional string cannot be parsed.
    NotImplementedError
        For pure Hartree-Fock, non-local correlation, dispersion corrections,
        or an unsupported functional type.
    """
    try:
        xc_type = dft.libxc.xc_type(xc)
    except (KeyError, ValueError, NotImplementedError) as error:
        raise ValueError(f"Unrecognized XC functional string: {xc!r}") from error

    if xc_type == "HF":
        raise NotImplementedError(
            f"XC functional {xc!r} is pure Hartree-Fock exchange. "
            f"Use the UHF interface for Hartree-Fock calculations."
        )
    if xc_type not in SUPPORTED_UNRESTRICTED_XC_TYPES:
        raise NotImplementedError(
            f"XC functional {xc!r} has type {xc_type!r}; supported types are "
            f"{SUPPORTED_UNRESTRICTED_XC_TYPES}."
        )
    if mf.do_nlc() or getattr(mf, "disp", None):
        raise NotImplementedError(
            "UKS shadow MD does not include non-local correlation or "
            "dispersion corrections."
        )

    log.info(f" -- Verified XC functional {xc!r} of type {xc_type!r}.")
    return xc_type


class UKS(UHF):
    """Interface class for PySCF unrestricted Kohn-Sham (UKS) calculations.

    Collinear UKS with LDA, GGA, or meta-GGA functionals, including
    global-hybrid and range-separated variants. Inherits the unrestricted
    density conventions and copying behaviour of :class:`UHF`.
    """

    def __init__(
        self,
        pyscf_mol: gto.MoleBase | None = None,
        pyscf_mf: dft.uks.UKS | None = None,
        xc: str = DEFAULT_XC,
    ) -> None:
        """Initialize UKS interface with PySCF objects.

        Parameters
        ----------
        pyscf_mol : gto.MoleBase, optional
            PySCF molecule object. Used to create a UKS mean-field object if
            pyscf_mf is not provided.
        pyscf_mf : dft.uks.UKS, optional
            PySCF UKS mean-field object. If provided, takes precedence over
            pyscf_mol, and xc is ignored.
        xc : str, default="LDA,VWN5"
            Exchange-correlation functional used when creating a new object.

        Raises
        ------
        TypeError
            If neither object is provided or either has the wrong type.
        ValueError
            If the functional string cannot be parsed.
        NotImplementedError
            If the functional is not supported; see the module docstring.
        """
        log.info(">> Instantiating PySCF-backend UKS interface...")

        if pyscf_mf is not None:
            if not isinstance(pyscf_mf, dft.uks.UKS):
                raise TypeError(f"pyscf_mf must be an instance of dft.uks.UKS, "
                                f"got {type(pyscf_mf).__name__}")
            mf = pyscf_mf
        elif pyscf_mol is not None:
            if not isinstance(pyscf_mol, gto.MoleBase):
                raise TypeError(f"pyscf_mol must be an instance of gto.MoleBase, "
                                f"got {type(pyscf_mol).__name__}")
            mf = dft.uks.UKS(pyscf_mol, xc=xc)
        else:
            raise TypeError("Must provide either pyscf_mf (dft.uks.UKS) "
                            "or pyscf_mol (gto.MoleBase).")

        self.xc_type: str = verify_supported_unrestricted_xc(mf.xc, mf)
        self._initialize_unrestricted(mf)
        self.xc: str = self.mf.xc

        ### Grid response is always included, so forces are conservative.
        self.grad.grid_response = True

        return


    def _build_grids(self) -> Any:
        """Return the energy grids, building them if necessary."""
        grids = self.mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)
        return grids


    def _exchange_coefficients(self) -> tuple[float, float, float] | None:
        """Return (omega, alpha, hyb) for a hybrid functional, otherwise None."""
        ni = self.mf._numint
        if not ni.libxc.is_hybrid_xc(self.mf.xc):
            return None
        return ni.rsh_and_hybrid_coeff(self.mf.xc, spin=self.mol.spin)


    def compute_exchange_matrix(self, dm: Any) -> Any:
        """Return per-spin scaled exact-exchange matrices.

        Follows the global-hybrid and short-/long-range cases of PySCF's
        :func:`pyscf.dft.uks.get_veff`; zero for pure density functionals.
        """
        dm_numpy = self._to_pyscf_density(dm)
        coefficients = self._exchange_coefficients()
        if coefficients is None:
            return la.zeros_like(la.asarray(dm))

        omega, alpha, hyb = coefficients
        if omega == 0:
            vk = self.mf.get_k(mol=self.mol, dm=dm_numpy) * hyb
        elif alpha == 0: # LR=0, only SR exchange
            vk = self.mf.get_k(mol=self.mol, dm=dm_numpy, omega=-omega) * hyb
        elif hyb == 0: # SR=0, only LR exchange
            vk = self.mf.get_k(mol=self.mol, dm=dm_numpy, omega=omega) * alpha
        else: # SR and LR exchange with different ratios
            vk = self.mf.get_k(mol=self.mol, dm=dm_numpy) * hyb
            vk += (alpha - hyb) * self.mf.get_k(mol=self.mol, dm=dm_numpy, omega=omega)

        return la.asarray(vk)


    def compute_exchange_gradient(self, dm: Any) -> Any:
        """Return per-spin scaled exact-exchange derivative builds.

        Uses the same combination as PySCF's :func:`pyscf.grad.uks.get_veff`;
        zero for pure density functionals.
        """
        coefficients = self._exchange_coefficients()
        if coefficients is None:
            nao = self.mol.nao_nr()
            return la.zeros((2, 3, nao, nao))

        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        omega, alpha, hyb = coefficients
        dm_numpy = self._to_pyscf_density(dm)
        vk = self.grad.get_k(mol=self.mol, dm=dm_numpy) * hyb
        if omega != 0:
            vk += (alpha - hyb) * self.grad.get_k(mol=self.mol, dm=dm_numpy, omega=omega)

        return la.asarray(vk)


    def _jk_derivatives(self, dms: Any) -> tuple[Any, Any]:
        """Return Coulomb and scaled exact-exchange derivative builds for stacked densities."""
        if not self.grad:
            raise RuntimeError("Gradient object is not initialized.")
        coefficients = self._exchange_coefficients()
        if coefficients is None:
            vj = self.grad.get_j(mol=self.mol, dm=dms)
            return vj, np.zeros_like(vj)

        omega, alpha, hyb = coefficients
        vj, vk = self.grad.get_jk(mol=self.mol, dm=dms)
        vk *= hyb
        if omega != 0:
            vk += (alpha - hyb) * self.grad.get_k(mol=self.mol, dm=dms, omega=omega)
        return vj, vk


    def compute_xc_terms(self, dm: Any) -> tuple[float, Any]:
        """Return the semilocal XC energy and per-spin potentials at ``dm``.

        Coulomb and exact exchange are excluded. The caller contracts the
        potentials with the difference between target and propagated densities.
        """
        grids = self._build_grids()
        _, exc, vxc = self.mf._numint.nr_uks(
            self.mol, grids, self.mf.xc, self._to_pyscf_density(dm),
            max_memory=self.mf.max_memory, verbose=self.mf.verbose,
        )
        return exc, la.asarray(vxc)


    def get_potential_response(self, dm: Any) -> Callable[[Any], Any]:
        """Return the coupled alpha/beta XC kernel plus Coulomb and scaled exchange.

        Evaluating Vxc on a perturbation density is not a derivative, so fxc
        is cached at ``dm`` and applied to each perturbation.
        """
        dm = self._to_pyscf_density(dm)
        dm = (dm + dm.swapaxes(-1, -2)) * 0.5
        if self.mf.grids.coords is None:
            self.mf.initialize_grids(self.mol, dm)
        ni = self.mf._numint
        rho, vxc, fxc = ni.cache_xc_kernel1(
            self.mol, self.mf.grids, self.mf.xc, dm, spin=1,
            max_memory=self.mf.max_memory,
        )

        def response(delta_dm):
            delta_dm = self._to_pyscf_density(delta_dm)
            delta_dm = (delta_dm + delta_dm.swapaxes(-1, -2)) * 0.5
            potential = ni.nr_uks_fxc(
                self.mol, self.mf.grids, self.mf.xc, dm, delta_dm,
                hermi=1, rho0=rho, vxc=vxc, fxc=fxc,
                max_memory=self.mf.max_memory,
            )
            return la.asarray(potential + la.to_numpy(self.compute_coulomb_matrix(delta_dm))
                              - la.to_numpy(self.compute_exchange_matrix(delta_dm)))
        return response


    def _xc_matrix_derivative(self, vtmp, ao, wv, mask, ao_loc, xctype) -> None:
        """Accumulate an AO-derivative matrix for grid weights ``wv`` into ``vtmp``.

        ``wv`` has the layout of one spin channel of ``eval_xc_eff`` output,
        already multiplied by the quadrature weights.
        """
        if xctype == "LDA":
            aow = grad.rks.numint._scale_ao(ao[0], wv[0])
            grad.rks._d1_dot_(vtmp, self.mol, ao[1:4], aow, mask, ao_loc, True)
            return

        wv = wv.copy()
        wv[0] *= 0.5
        if xctype == "MGGA":
            wv[4] *= 0.5
        grad.rks._gga_grad_sum_(vtmp, self.mol, ao, wv, mask, ao_loc)
        if xctype == "MGGA":
            grad.rks._tau_grad_dot_(vtmp, self.mol, ao, wv[4], mask, ao_loc, True)

        return


    def compute_xc_gradient(self, dm_prop: Any, dm_ao: Any) -> tuple[Any, Any]:
        r"""Return the linearized spin-polarized XC matrix derivative and grid terms.

        Adapted from PySCF's :func:`pyscf.grad.uks.get_vxc_full_response`,
        whose loop structure, AO derivative order, and contraction helpers are
        kept. The modification, as in the restricted adapter, is the
        linearized shadow functional

        .. math::

            E_{xc}^{lin} = E_{xc}[\rho^{P}]
                + \sum_s \int v_{xc,s}[\rho^{P}]\,(\rho^{D}_s - \rho^{P}_s),

        whose derivative has an AO-derivative term contracted with the target
        density, a kernel term
        :math:`\sum_{st}\int f_{xc,st}[\rho^{P}]\,(\rho^{D}_t - \rho^{P}_t)
        \,\partial_R\rho^{P}_s`, and the response of the quadrature grid.

        Parameters
        ----------
        dm_prop : array
            Propagated per-spin density, shape ``(2, nao, nao)``.
        dm_ao : array
            Target per-spin density, shape ``(2, nao, nao)``.

        Returns
        -------
        tuple
            The XC matrix derivative with shape ``(2, 3, nao, nao)``, to be
            contracted with ``dm_ao``, and the per-atom gradient with shape
            ``(natom, 3)`` collecting the kernel and grid-response terms.
        """
        ni = self.mf._numint
        xctype = ni._xc_type(self.mf.xc)
        if xctype not in SUPPORTED_UNRESTRICTED_XC_TYPES:
            raise NotImplementedError(
                f"Linearized UKS gradients are not implemented for XC type {xctype!r}."
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
        if nset_prop != 2 or nset_ao != 2:
            raise ValueError("Linearized UKS gradients require alpha and beta densities.")

        ao_loc = self.mol.ao_loc_nr()
        ao_deriv = 1 if xctype == "LDA" else 2
        n_ao_values = {"LDA": 1, "GGA": 4, "MGGA": 10}[xctype]

        exc_grid = np.zeros((self.mol.natm, 3))
        vmat = np.zeros((2, 3, nao, nao))
        kernel_vmat = np.zeros((2, 3, nao, nao))

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
            ao_values = ao[0] if xctype == "LDA" else ao[:n_ao_values]
            rho_prop = np.asarray([make_rho_prop(s, ao_values, mask, xctype) for s in range(2)])
            rho_ao = np.asarray([make_rho_ao(s, ao_values, mask, xctype) for s in range(2)])
            exc, vxc, fxc = ni.eval_xc_eff(self.mf.xc, rho_prop, 2, xctype=xctype)[:3]

            # Uniform (spin, variable, grid) layout, including LDA.
            ngrid = weight.size
            rho_prop = rho_prop.reshape(2, -1, ngrid)
            delta_rho = rho_ao.reshape(2, -1, ngrid) - rho_prop
            vxc = vxc.reshape(2, -1, ngrid)
            nvar = vxc.shape[1]
            fxc = fxc.reshape(2, nvar, 2, nvar, ngrid)
            kernel_vxc = np.einsum("satbr,tbr->sar", fxc, delta_rho, optimize=True)
            exc_lin = (exc * (rho_prop[0, 0] + rho_prop[1, 0])
                       + np.einsum("sar,sar->r", vxc, delta_rho, optimize=True))

            for s in range(2):
                vtmp = np.zeros((3, nao, nao))
                kernel_vtmp = np.zeros((3, nao, nao))
                self._xc_matrix_derivative(vtmp, ao, weight * vxc[s], mask, ao_loc, xctype)
                self._xc_matrix_derivative(
                    kernel_vtmp, ao, weight * kernel_vxc[s], mask, ao_loc, xctype
                )
                vmat[s] += vtmp
                kernel_vmat[s] += kernel_vtmp

                ### Motion of the grid points belonging to this atom
                exc_grid[atm_id] += (
                    2.0 * np.einsum("xij,ji->x", vtmp, dm_ao_numpy[s])
                    + 2.0 * np.einsum("xij,ji->x", kernel_vtmp, dm_prop_numpy[s])
                )

            ### Response of the quadrature weights
            exc_grid += np.einsum("r,nxr->nx", exc_lin, weight1, optimize=True)

        ### XC-kernel contribution, contracted with the propagated density
        aoslices = self.compute_ao_slices_by_atom()
        for atom_index in range(self.get_num_atoms()):
            p0, p1 = aoslices[atom_index, 2:]
            exc_grid[atom_index] -= 2.0 * np.einsum(
                "sxij,sij->x",
                kernel_vmat[:, :, p0:p1],
                dm_prop_numpy[:, p0:p1],
                optimize=True,
            )

        # Negative sign because nabla_X = -nabla_x.
        return la.asarray(-vmat), la.asarray(exc_grid)


    def compute_xc_gradient_per_atom(self, dm_prop: Any, dm_ao: Any) -> Any:
        """Return all exchange-correlation gradient contributions per atom.

        Contracts the XC matrix derivative of :meth:`compute_xc_gradient`
        with each spin's target density and adds its kernel and grid-response
        terms.
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
                "sxij, sij-> x",
                xc_grad[:, :, p0:p1],
                dm_ao[:, p0:p1],
                optimize=True,
            )

        return total_grad
