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

"""Provides base class for Newton-Raphson SCF procedure.

See the documentation of NewtonRaphson for details.

PySMD uses spatial-orbital occupations in the range ``[0, 1]``. Conversion to
a backend's spin-summed convention is handled by the electronic-structure
interface.
"""

from __future__ import annotations

from pysmd.closed_shell import fermi_dirac
from pysmd.closed_shell import jacobian
from pysmd.common import constants
from pysmd.common import logger
log: logger.logging.Logger = logger.getLogger(__name__)

from pysmd.interface import qm_software

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

### Default attributes
# Finite temperature (keep synced with MD module)
DEFAULT_TEMP = 300.0
FRAC_OCC_ERROR_TOL = 1e-10
# Max iterations
MAX_CYCLES = 40
# Convergence criteria, energy and norm of residual matrix
ENERGY_DIFF_ERROR_TOL = 1e-8
RESIDUAL_NORM_ERROR_TOL = 1e-8
# Low-rank kernel
RELATIVE_RESIDUAL_ERROR_TOL = 1e-4
PT_RECURSION_DEPTH = 8


class NewtonRaphson:
    """Base class for ground-state Newton-Raphson SCF calculation."""


    def __init__(
        self,
        qm_interface: (qm_software.QMSoftware),
        scf_params,
    ) -> None:
        """Initializes instance of NewtonRaphson object.

        A QM software interface and dictionary of SCF parameters, usually
        defined by the MD pre-kernel, are required.

        Args:
            qm_interface: Instance of a QMSoftware subclass specific to the QM
                          software being used as a backend for computation.
            scf_params: Dictionary of SCF parameters.
        """
        log.info("\n>> Initializing NewtonRaphson object...")

        ### Verify interface object
        self.interface: (qm_software.QMSoftware | None) = None
        self.interface = qm_interface
        log.info(" -- Inherited QM software interface object:\n"
                f"   {self.interface.__class__}")

        ### SCF parameters
        # Maximum number of cycles
        self.max_cycle: int =\
            scf_params.get("max_cycle", MAX_CYCLES)
        # Convergence criteria;
        # energy difference
        self.energy_diff_error_tol: float =\
            scf_params.get("energy_diff_error_tol", ENERGY_DIFF_ERROR_TOL)
        # Convergence criteria;
        # norm of residual density matrix difference
        self.res_norm_error_tol: float =\
            scf_params.get("res_norm_error_tol", RESIDUAL_NORM_ERROR_TOL)

        ### System attributes
        # Number of atoms
        self.n_atom: int =\
            scf_params.get("n_atom", self.interface.get_num_atoms())
        # Number of electrons
        self.n_elec: int =\
            scf_params.get("n_elec", self.interface.get_num_electrons())
        # Number of molecular orbitals
        self.n_mo: int =\
            scf_params.get("n_mo", self.interface.get_num_basis_functions())
        # Number of doubly-occupied orbitals
        self.n_occ: int =\
            scf_params.get("n_occ", self.n_elec // 2)

        ### Finite temperature
        # Temperature
        self.temp: float =\
            scf_params.get("temp", DEFAULT_TEMP)
        # Convergence criteria;
        # fractional orbital occupation delta
        self.frac_occ_tol: float =\
            scf_params.get("frac_occ_tol", FRAC_OCC_ERROR_TOL)
        # Chemical potential
        self.mu: (float | None) = None
        # Inverse of kB*T
        self.beta: (float | None) = None

        ### Low-rank kernel parameters
        # Convergence criteria;
        # relative residual error
        self.rel_res_error_tol: float =\
            scf_params.get("rel_res_error_tol", RELATIVE_RESIDUAL_ERROR_TOL)
        # Max rank
        self.max_rank: int =\
            scf_params.get("max_rank", self.n_mo)
        # Recursive perturbation depth
        self.pt_recursion_depth: int =\
            scf_params.get("pt_recursion_depth", PT_RECURSION_DEPTH)
        self.initial_dm_ao =\
            scf_params.get("initial_dm_ao", None)

        ### Computed quantities
        # Overlap matrices
        self.S = None
        self.Sm1 = None
        self.Sp12 = None
        self.Sm12 = None
        # One-/two-body terms
        self.h1e = None
        self.vhf = None
        # Fock matrix
        self.fock = None
        self.fock_orth = None
        # Orbital info
        self.mo_energy = None
        self.mo_coeff = None
        self.mo_occ = None
        # Density matrices
        self.dm_ao = None
        self.dm_orth = None
        # Dynamical variable, X
        self.dynvar_X = None
        # Energies
        # Note: there is no separate e_xc here. This class obtains its energy
        # from the backend's total electronic energy, which folds any
        # exchange-correlation contribution into e2. ShadowMD builds the
        # decomposition explicitly and reports e_xc separately.
        self.e_tot: float = 0.0
        self.e_nuc: float = 0.0
        self.e1: float = 0.0
        self.e2: float = 0.0
        self.e_elec: float = 0.0
        self.e_ent: float = 0.0
        self.e_free: float = 0.0

        return


    def kernel(self) -> None:
        """Newton-Raphson-like SCF energy minimization kernel.

        TODO: More details
        """
        cput0: tuple[float, float] = (log.cpu_time(), log.wall_time())

        ### Initialize overlap and density matrix
        self.S, self.Sm1, self.Sp12, self.Sm12 = self.interface.compute_overlap_matrices()
        if self.initial_dm_ao is None:
            self.dm_ao = self.interface.compute_guess_density_matrix()
        else:
            self.dm_ao = la.copy(self.initial_dm_ao)
        log.info(" -- Calculated overlap matrix, S and density matrix, D.")

        ### Dynamical variable; X=D*S
        self.dynvar_X = la.matmul(self.dm_ao, self.S)
        log.info(" -- Calculated dynamical variable, X.")

        ### Components of Fock matrix
        self.h1e = self.interface.compute_core_hamiltonian_matrix()
        self.vhf = self.interface.compute_eff_potential_matrix(dm=self.dm_ao)
        log.info(" -- Calculated core Hamiltonian and initial potential matrix.")

        ### Initial energy
        self.update_energies(dm_ao=self.dm_ao, compute_e_nuc=True)
        log.info(" -- Calculated initial electronic energy.")


        ##### NEWTON RAPHSON-LIKE SELF-CONSISTENT FIELD #####
        cput1: tuple[float, float] = (log.cpu_time(), log.wall_time())

        log.info("\n>>>> Starting Newton-Raphson SCF cycles <<<<")
        for itr in range(1, self.max_cycle + 1):
            log.info(f"\n----- ITERATION #{itr:d} -----")
            e_prev: float = self.e_tot

            ### Fock matrix
            log.info(">> Diagonalizing orthogonal Fock matrix...\n")
            self.fock = self.h1e + self.vhf

            ### Orthogonalize and diagonalize
            self.fock_orth = la.matmul(
                la.matmul(self.Sm12.T.conj(), self.fock), self.Sm12
            )
            self.mo_energy, self.mo_coeff = la.linalg.eigh(self.fock_orth)

            ### Fermi-Dirac operator expansion (finite temperature)
            log.info(">> Computing fractional occupation...")
            self.mo_occ, self.mu = fermi_dirac.update_fractional_occ(self)

            log.info(
                " -- Updated: Orbital occupations\n"
                f"   = [{' '.join(format(x, '.2f') for x in self.mo_occ.tolist())}]"
            )
            log.info(f" -- Updated: Chemical potential\n   = {self.mu:8f}")

            ### Update density matrices
            log.info("\n>> Updating density matrices...")
            self.dm_orth = la.matmul(
                la.matmul(self.mo_coeff, la.diag(self.mo_occ)),
                self.mo_coeff.T.conj(),
            )
            self.dm_ao = la.matmul(
                la.matmul(self.Sm12, self.dm_orth),
                self.Sm12.T.conj(),
            )

            ### Residual of dynamical variable
            dynvar_X_new = la.matmul(self.dm_ao, self.S)
            residual_dDS = dynvar_X_new - self.dynvar_X

            ### Update dynamical variable using approximate inverse Jacobian
            log.info("\n>> Updating dynamical variable, X...")
            kernel_res = jacobian.pseudo_inverse_action(self, residual_dDS)

            ### Update density matrix after dynamical variable step
            self.dynvar_X -= kernel_res
            self.dm_ao = la.matmul(self.dynvar_X, self.Sm1)

            ### Update potential matrix
            self.vhf = self.interface.compute_eff_potential_matrix(dm=self.dm_ao)

            ### Update energies
            log.info("\n>> Updating energy components (Hartree)...")
            self.update_energies(dm_ao=self.dm_ao, compute_e_free=True)
            log.info(f"   {'-' * 46}\n"
                     f"   Nuclear Rep.   (E_nuc)  = {self.e_nuc:>20.15f}\n"
                     f"   Electronic     (E_elec) = {self.e_elec:>20.15f}\n"
                     f"   Entropic       (E_ent)  = {self.e_ent:>20.15f}\n"
                     f"   {'-' * 46}\n"
                     f"   Total Energy   (E_tot)  = {self.e_tot:>20.15f}"
            )

            ### Compute convergence metrics
            residual_norm: float = la.linalg.norm(residual_dDS)
            e_diff: float = la.abs(self.e_tot - e_prev)

            log.info(f"\n>>  | deltaE |  = {e_diff:10.6e}" +
                     f"\n>> || deltaX || = {residual_norm:10.6e}")
            cput1 = log.timer(f"Newton-Raphson SCF cycle #{(itr):d}", *cput1)

            ### Check convergence
            if (e_diff <= self.energy_diff_error_tol
            and residual_norm <= self.res_norm_error_tol):
                self.interface.conv_scf = True
                break

        ### End of Newton-Raphson SCF kernel
        if self.interface.get_scf_convergence():
            log.info(f"\n>>>>  CONVERGED: {itr:d} ITERATIONS  <<<<")
        else:
            log.error("Maximum SCF cycles reached before energy convergence.")

        log.info(">>>>    END NEWTON-RAPHSON SCF PROCEDURE    <<<<")
        cput0 = log.timer("Total Newton-Raphson SCF time", *cput0)

        return


    def update_energies(
        self,
        dm_ao,
        compute_e_nuc: bool = False,
        compute_e_free: bool = False,
    ) -> None:
        """Compute and update energy attributes.

        Parameters
        ----------
        dm_ao : numpy.ndarray
            Density matrix for trace, atomic orbital basis.
        compute_e_nuc : bool
            Optional. If True, update nuclear repulsion energy.
        compute_e_free : bool
            Optional. If True, update entropy and free energy.
        """

        # Electronic energy
        self.e1, self.e2, self.e_elec = self.interface.get_electronic_energy(
            dm = dm_ao,
            h1e = self.h1e,
            vhf = self.vhf
        )

        # Nuclear repulsion energy
        if compute_e_nuc:
            self.e_nuc = self.interface.get_nuclear_energy()

        # Entropic and free energy
        if compute_e_free and self.temp > 0:
            f = self.mo_occ[(self.mo_occ > 0) & (self.mo_occ < 1)] # prevent log(0)
            self.e_ent = (2.0 * self.temp * constants.KB
                          * la.sum(f * la.log(f) + (1.0-f) * la.log(1.0-f)))
        self.e_free = self.e_elec + self.e_ent

        # Total energy
        self.e_tot = self.e_nuc + self.e_free

        return
