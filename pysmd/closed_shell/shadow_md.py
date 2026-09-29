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

"""Provides base class for ground-state Shadow MD simulations.

See the documentation of ShadowMD for details.

PySMD uses spatial-orbital occupations in the range ``[0, 1]``. The closed-
shell occupation count is therefore ``n_elec / 2``; conversion to a backend's
spin-summed convention is handled by the electronic-structure interface.
"""

from __future__ import annotations

from pysmd.closed_shell import eom_integrators
from pysmd.closed_shell import eom_params
from pysmd.closed_shell import fermi_dirac
from pysmd.closed_shell import jacobian
from pysmd.closed_shell import newton_raphson
from pysmd.closed_shell import simulation_data
from pysmd.common import constants
from pysmd.common import logger
log: logger.logging.Logger = logger.getLogger(__name__)

from pysmd.interface import qm_software

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()

### Default attributes
# Finite temperature
DEFAULT_TEMP = 300.0
FRAC_OCC_ERROR_TOL = 1e-10
# EOM integration scheme and dissipation
DEFAULT_EOM_SCHEME = "verlet"
DEFAULT_EOM_L_MAX = 2
DEFAULT_DISS_K_MAX = 6
# Simulation time (atomic units, converted from femtoseconds)
DEFAULT_TSTEP_SIZE = (1.0 / constants.AU2FS) # 1 fs -> au
DEFAULT_NUM_TSTEPS = 100
DEFAULT_TOTAL_TIME = DEFAULT_TSTEP_SIZE * DEFAULT_NUM_TSTEPS
# Low-rank kernel
RELATIVE_RESIDUAL_ERROR_TOL = 1e-4
PT_RECURSION_DEPTH = 8


class ShadowMD:
    """Base class for ground-state MD simulation.

    Attributes:
        _scf_keys: strings of keys corresponding to values required in MD kernel.
    """
    _scf_keys: list[str] = [
        "temp", "mu", "beta", "frac_occ_tol", # finite temperature
        "S", "Sm1", "Sp12", "Sm12", # overlap matrices
        "h1e", "vhf", "fock", "fock_orth", # components of Fock matrix
        "mo_energy", "mo_coeff", "mo_occ", # orbital info
        "dm_ao", "dm_orth", "dynvar_X", # density matrices
        "e_nuc", "e1", "e2", # energy components
        "e_elec", "e_ent", "e_free", "e_tot" # energies
    ]


    def __init__(
        self,
        qm_interface: qm_software.QMSoftware,
    ) -> None:
        """Initializes instance of ShadowMD object.

        A QM software interface is required.

        Args:
            qm_interface: Instance of a QMSoftware subclass specific to the QM
                          software being used as a backend for computation.
        """
        log.info("\n>> Instantiating ShadowMD object...")

        ### Verify interface object
        if not isinstance(qm_interface, qm_software.QMSoftware):
            raise TypeError(
                f"qm_interface must be an instance of QMSoftware, "
                f"got {type(qm_interface).__name__}"
            )
        self.interface: qm_software.QMSoftware = qm_interface
        log.info(" -- Inherited QM software interface object:\n"
                f"   {self.interface.__class__}")

        ### General attributes
        # Number of atoms
        self.n_atom: (int | None) = None
        # Number of electrons
        self.n_elec: (int | None) = None
        # Number of molecular orbitals (contracted GTOs)
        self.n_mo: (int | None) = None
        # Number of doubly-occupied orbitals
        self.n_occ: (int | None) = None

        ### Finite temperature
        # Temperature
        self.temp: float = DEFAULT_TEMP
        # Convergence criteria;
        # fractional orbital occupation delta
        self.frac_occ_tol: float = FRAC_OCC_ERROR_TOL
        # Chemical potential
        self.mu: (float | None) = None
        # Inverse of kB*T
        self.beta: (float | None) = None

        ### Compute system attributes
        self.initialize_general_attributes()

        ### Simulation data
        self.sim_data: (simulation_data.SimulationData | None) = None

        ### Timestep attributes
        # Units of time
        self._md_unit: str = "au"
        # Size of timestep
        self._md_timestep: float = DEFAULT_TSTEP_SIZE
        # Total simulation time
        self._md_total_time: float = DEFAULT_TOTAL_TIME
        # Number of timesteps
        self._md_num_tsteps: int = DEFAULT_NUM_TSTEPS

        ### Equation of motion
        # EOM parameter object
        self.eom_params: (eom_params.EOMParams | None) = None
        # Integration scheme
        self.eom_int_scheme: str = DEFAULT_EOM_SCHEME
        # Number of sub-integration steps per timestep
        self.eom_l_max: int = DEFAULT_EOM_L_MAX
        # Number of previous X matrices in dissipation term
        self.diss_k_max: int = DEFAULT_DISS_K_MAX
        # Coefficient-split Verlet integration
        self.coeff_split_verlet: bool = False

        ### Low-rank kernel parameters
        # Convergence criteria;
        # relative residual error
        self.rel_res_error_tol: float = RELATIVE_RESIDUAL_ERROR_TOL
        # Max rank
        assert self.n_mo is not None
        self.max_rank: int = self.n_mo
        # Recursive perturbation depth
        self.pt_recursion_depth: int = PT_RECURSION_DEPTH

        ### System attributes
        # Atomic masses
        self.atom_mass = None
        # Atomic coordinates
        self.atom_coord = None
        # Atomic velocities
        self.atom_veloc = None

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
        self.dm_prop = None
        self.dm_lin = None
        # Dynamical variables, X
        self.dynvar_X = None
        self.X_veloc = None
        self.X_accel = None
        self.X_history = None
        # Energies
        self.e_kin: float = 0.0
        self.e_tot: float = 0.0
        self.e_nuc: float = 0.0
        self.e1: float = 0.0
        self.e2: float = 0.0
        self.e_xc: float = 0.0
        self.e_elec: float = 0.0
        self.e_ent: float = 0.0
        self.e_free: float = 0.0

        ### Flags
        # First-level update of density matrices
        self.first_level_dm_update: bool = False
        # Perform SCF procedure for each MD timestep
        self.end_of_timestep_scf: bool = False

        return


    @property
    def md_unit(self) -> str:
        r"""Units of time."""
        return self._md_unit

    @md_unit.setter
    def md_unit(self, unit: str) -> None:
        r"""Set unit of time attribute."""
        if unit not in ("au", "fs"):
            raise ValueError(
                "Only atomic units (au) and femtoseconds (fs) "
                "currently supported."
            )

        if unit == self._md_unit:
            log.warning(f"Units of time already set to '{unit}'.")
            return

        self._md_unit = unit
        return

    def _time_to_au(self, time: float) -> float:
        r"""Convert a time value from current MD units to atomic units."""
        if self._md_unit == "fs":
            return time / constants.AU2FS

        return time

    def _time_from_au(self, time: float) -> float:
        r"""Convert a time value from atomic units to current MD units."""
        if self._md_unit == "fs":
            return time * constants.AU2FS

        return time


    @property
    def md_timestep(self) -> float:
        r"""Simulation timestep size in user-specified units."""
        return self._time_from_au(self._md_timestep)

    @md_timestep.setter
    def md_timestep(self, dt: float) -> None:
        r"""Set timestep (in current md_unit), stores in au."""
        if not la.isfinite(dt) or dt <= 0:
            raise ValueError(
                f"Timestep size must be finite and positive; "
                f"{dt = :.2f} {self.md_unit}."
            )

        timestep_au = self._time_to_au(dt)
        if timestep_au > self._md_total_time:
            raise ValueError(
                "Timestep size must not exceed total simulation time."
            )

        # Set timestep size in au
        self._md_timestep = timestep_au

        # Preserve total simulation time and update derived timestep count
        self._md_num_tsteps = int(round(self._md_total_time / self._md_timestep))

        return


    @property
    def md_num_tsteps(self) -> int:
        r"""Total number of timesteps."""
        return self._md_num_tsteps

    @md_num_tsteps.setter
    def md_num_tsteps(self, num_tsteps: int) -> None:
        r"""Set number of simulation timesteps."""
        if isinstance(num_tsteps, bool) or not isinstance(num_tsteps, int):
            raise ValueError("Number of timesteps must be an integer; "
                             f"{num_tsteps = }.")

        if num_tsteps < 1:
            raise ValueError("Number of timesteps must be at least one; "
                             f"{num_tsteps = }.")

        # Set number of timesteps
        self._md_num_tsteps = num_tsteps

        # Update total simulation time in au
        self._md_total_time = self._md_timestep * self._md_num_tsteps

        return


    @property
    def md_total_time(self) -> float:
        r"""Total simulation time in user-specifed units."""
        return self._time_from_au(self._md_total_time)

    @md_total_time.setter
    def md_total_time(self, total_time: float):
        r"""Set total simulation time (in current md_unit), stores in au."""
        if not la.isfinite(total_time) or total_time <= 0:
            raise ValueError("Total simulation time must be finite and positive.")

        # Convert provided value to au
        total_time_au = self._time_to_au(total_time)

        if total_time_au < self._md_timestep:
            raise ValueError(
                "Total simulation time must be greater than or equal to "
                "timestep size."
            )

        # Set total simulation time
        self._md_total_time = total_time_au

        # Update derived number of timesteps
        self._md_num_tsteps = int(round(self._md_total_time / self._md_timestep))

        return


    def kernel(self) -> None:
        """Perform ShadowMD simulation.

        Takes a configured electronic-structure interface as input. If the
        initial SCF state is not converged, a Newton-Raphson-like SCF
        optimization procedure is performed.
        """
        cput0: tuple[float, float] = (log.cpu_time(), log.wall_time())

        ### SCF convergence
        log.info("\n>> Verifying SCF convergence...")
        self.verify_scf_convergence()

        ### Nuclear degrees of freedom
        self.atom_mass = self.interface.get_atomic_masses()
        self.atom_coord = self.interface.get_atomic_coords()
        self.atom_veloc = la.zeros_like(self.atom_coord)

        ### Electronic degrees of freedom
        self.X_veloc = la.zeros_like(self.dynvar_X)
        self.X_accel = la.zeros_like(self.dynvar_X)

        ### Update energies
        log.info("\n>> Initial energy components (Hartree)...")
        log.info(f"   {'-' * 46}\n"
                 f"   Kinetic        (E_kin)  = {self.e_kin:>20.15f}\n"
                 f"   Nuclear Rep.   (E_nuc)  = {self.e_nuc:>20.15f}\n"
                 f"   Electronic     (E_elec) = {self.e_elec:>20.15f}\n"
                 f"   Entropic       (E_ent)  = {self.e_ent:>20.15f}\n"
                 f"   {'-' * 46}\n"
                 f"   Total Energy   (E_tot)  = {self.e_tot:>20.15f}"
        )

        ### Simulation parameters
        log.info("\n>> Instantiating simulation results object...\n "
                 "-- Total # of timesteps:\n   "
                f"{self.md_num_tsteps}\n "
                 "-- Timestep size:\n   "
                f"{self.md_timestep:.1f} ({self.md_unit})\n "
                 "-- Total time:\n   "
                f"{self.md_total_time:.1f} ({self.md_unit})"
        )

        ### Simulation data object
        self.sim_data = simulation_data.SimulationData(
            unit=self.md_unit,
            timestep=self.md_timestep,
            num_tsteps=self.md_num_tsteps,
            num_atoms=self.n_atom
        )
        self.sim_data.store_initial_system_data(
            atomic_coord=self.atom_coord,
            energy_tot=self.e_tot,
            energy_kin=self.e_kin
        )

        ### EOM parameters
        log.info("\n>> Storing EOM parameters...")
        self.eom_params = eom_params.EOMParams(
            scheme=self.eom_int_scheme,
            l_max=self.eom_l_max,
            k_max=self.diss_k_max
        )

        ### Dynamical variable, X; "history" matrix
        # Horizontal stacking: K_max + 1
        self.X_history = la.hstack([self.dynvar_X] * (self.diss_k_max + 1))

        # Vertical stacking: L_max
        if (
            self.eom_int_scheme == "optimal"
            or (
                self.eom_int_scheme == "verlet"
                and self.coeff_split_verlet
            )
        ):
            self.X_history = la.vstack([self.X_history] * (self.eom_l_max))

        ### Initialize residual matrix and gradient
        residual_dDS = None

        log.info("\n>> Computing initial gradients...")
        total_grad = self.update_gradients(
            fock=self.fock,
            dm_ao=self.dm_ao,
            dm_prop=self.dm_ao,
            dm_lin=self.dm_ao
        )

        ##### SHADOW BORN-OPPENHEIMER MOLECULAR DYNAMICS #####
        cput1: tuple[float, float] = (log.cpu_time(), log.wall_time())

        log.info("\n>>>> Starting SMD simulation <<<<")
        for ts in range(self.md_num_tsteps):
            log.info(f"\n----- TIMESTEP #{(ts+1):d} -----")
            e_prev: float = self.e_tot

            ### Equations of motion
            log.info(">> Integrating equations of motion...")

            # Verlet integration scheme
            if (
                self.eom_int_scheme == "verlet"
                and not self.coeff_split_verlet
            ):
                residuals, total_grad = eom_integrators.integrate_verlet_timestep(
                    self,
                    tstep=ts,
                    total_grad=total_grad,
                    residual_dDS=residual_dDS
                )

                ### Update history matrix
                self.X_history = la.hstack((self.dynvar_X,
                                            self.X_history[:, :-self.n_mo]))

            # Optimal symplectic integration schemes
            elif (
                self.eom_int_scheme == "optimal"
                or (
                    self.eom_int_scheme == "verlet"
                    and self.coeff_split_verlet
                )
            ):
                (
                    residuals,
                    total_grad,
                    tstep_X_history,
                ) = eom_integrators.integrate_coeff_split_timestep(
                    self,
                    tstep=ts,
                    total_grad=total_grad,
                    residual_dDS=residual_dDS
                )

                ### Update history matrix
                self.X_history = la.hstack((tstep_X_history,
                                            self.X_history[:, :-self.n_mo]))

            ### Dynamical variable residual
            log.info("\n>> Computing residual matrix norms...")
            residual_dDS = self.sim_data.update_residual_norm_data(
                tstep=ts,
                residual_tuple=residuals
            )

            log.info(" -- ||deltaX^(0)|| =" +
                    f" {self.sim_data.residual_norm[ts]:10.6e}")
            if self.first_level_dm_update:
                log.info(" -- ||deltaX^(1)|| =" +
                        f" {self.sim_data.first_level_residual_norm[ts]:10.6e}")

            ### Update energies
            log.info("\n>> Updating energies...")
            self.update_energies(
                dm_ao = self.dm_ao,
                dm_prop = self.dm_prop,
                dm_lin = self.dm_lin,
                compute_e_nuc = True,
                compute_e_free = True,
                compute_e_kin = True,
            )

            e_diff: float = la.abs(self.e_tot - e_prev)
            log.info(f"   {'-' * 46}\n"
                 f"   Kinetic        (E_kin)  = {self.e_kin:>20.15f}\n"
                 f"   Nuclear Rep.   (E_nuc)  = {self.e_nuc:>20.15f}\n"
                 f"   Electronic     (E_elec) = {self.e_elec:>20.15f}\n"
                 f"   Entropic       (E_ent)  = {self.e_ent:>20.15f}\n"
                 f"   {'-' * 46}\n"
                 f"   Total Energy   (E_tot)  = {self.e_tot:>20.15f}\n"
                 f"   {'-' * 46}\n"
                 f"   Energy Diff.   (E_diff) = {e_diff:>20.15f}\n"
            )

            ### Exact gradient
            if self.end_of_timestep_scf:
                log.info(f"\n>> Computing exact SCF gradients...")
                exact_grad, exact_scf_energy = self.end_of_timestep_scf_kernel()

                self.sim_data.update_gradient_error_data(
                    tstep=ts,
                    exact_grad=exact_grad,
                    timestep_grad=total_grad,
                    exact_scf_energy=exact_scf_energy,
                    timestep_energy=(self.e_tot - self.e_kin)
                )
                log.info(" -- Updated: Gradient error norm\n" +
                        f"   = {self.sim_data.grad_error_norm[ts]:10.6e}")
                log.info(" -- Updated: Exact SCF energy and energy error\n"
                        f"   E_SCF       = {self.sim_data.exact_scf_energy[ts]:20.15f}\n"
                        f"   dE(SCF-lin) = {self.sim_data.scf_energy_diff[ts]:20.15f}")

            ### Store system info
            self.sim_data.update_system_data(
                tstep=ts,
                atomic_coord=self.atom_coord,
                energy_tot=self.e_tot,
                energy_kin=self.e_kin
            )

            cput1 = log.timer(f"MD timestep #{(ts+1):d}", *cput1)

        ### End of MD kernel
        log.info(">>>>   END MD SIMULATION   <<<<")
        cput0 = log.timer("Total MD time", *cput0)

        return


    def initialize_general_attributes(self) -> None:
        """Update general attributes of the system based on interface methods.

        This method queries the underlying computational interface to retrieve
        and update values describing the system. Specifically, it obtains the
        total number of atoms, electrons, molecular orbitals, and occupied
        orbitals.

        Attributes updated:
            - `self.n_atom`: Number of atoms in the molecular system.
            - `self.n_elec`: Total number of electrons in the system.
            - `self.n_mo`: Total number of molecular orbitals (contracted GTOs).
            - `self.n_occ`: Number of occupied spatial orbitals, computed as
                            half the number of electrons. Each such orbital
                            has PySMD occupation one at zero temperature.
        """
        ### System properties
        self.n_atom = self.interface.get_num_atoms()
        self.n_elec = self.interface.get_num_electrons()
        self.n_mo = self.interface.get_num_basis_functions()
        self.n_occ = self.n_elec // 2

        ### Finite temperature
        if self.interface.temp is not None:
            self.temp = self.interface.temp

        return


    def update_atomic_coords(
        self,
        new_coords
    ):
        """Return updated atomic coordinates."""
        return self.interface.get_updated_atomic_coords(new_coords)


    def update_overlap_matrices(self) -> None:
        """Compute and update the object's overlap matrices."""
        self.S, self.Sm1, self.Sp12, self.Sm12 = self.interface.compute_overlap_matrices()
        return


    def update_energies(
        self,
        dm_ao,
        dm_prop=None,
        dm_lin=None,
        compute_e_nuc: bool = False,
        compute_e_free: bool = False,
        compute_e_kin: bool = False,
    ) -> None:
        """Compute and update energy attributes.

        Parameters
        ----------
        dm_ao : numpy.ndarray
            Density matrix, AO basis.
        dm_prop : numpy.ndarray
            Optional. Propagated density matrix, AO basis.
        dm_lin : numpy.ndarray
            Optional. Linearized density matrix, AO basis.
        compute_e_nuc : bool
            Optional. If True, update nuclear repulsion energy.
        compute_e_free : bool
            Optional. If True, update entropy and free energy.
        """
        # Propagated and linearized density matrices
        if dm_prop is None: dm_prop = dm_ao
        if dm_lin is None: dm_lin = dm_ao

        # One-electron energy
        self.e1 = 2.0 * la.einsum("ij, ji->", self.h1e, dm_ao, optimize=True)

        # Linearized two-electron energy
        vj = self.interface.compute_coulomb_matrix(dm=dm_prop)
        vk = self.interface.compute_exchange_matrix(dm=dm_prop)

        # Exchange-correlation potential energy (RKS)
        self.e_xc, vxc = self.interface.compute_xc_terms(dm=dm_prop)

        e_coul = la.einsum("ij, ji->", vj, dm_lin, optimize=True)
        e_coul += 2.0 * la.einsum(
            "ij, ji->", vxc, (dm_ao - dm_prop), optimize=True
        )
        e_exchange = -0.5 * la.einsum("ij, ji->", vk, dm_lin, optimize=True)

        self.e2 = e_coul + e_exchange
        self.e_elec = self.e1 + self.e2 + self.e_xc

        # Nuclear repulsion energy
        if compute_e_nuc:
            self.e_nuc = self.interface.get_nuclear_energy()

        # Entropy and free energy
        if compute_e_free and self.temp > 0:
            f = self.mo_occ[(self.mo_occ > 0) & (self.mo_occ < 1)] # prevent log(0)
            self.e_ent = (2.0 * self.temp * constants.KB
                          * la.sum(f * la.log(f) + (1.0-f) * la.log(1.0-f)))
        self.e_free = self.e_elec + self.e_ent

        # Kinetic energy
        if compute_e_kin:
            veloc_squared = la.sum(self.atom_veloc ** 2, axis=1)
            self.e_kin = 0.5 * la.sum(self.atom_mass * veloc_squared)

        # Total energy
        self.e_tot = self.e_nuc + self.e_free + self.e_kin

        return


    def update_gradients(
        self,
        fock,
        dm_ao,
        dm_prop=None,
        dm_lin=None,
    ):
        r"""Compute nuclear and electronic components of the energy gradient.

        This is the exact nuclear derivative of the shadow potential energy
        surface :math:`\mathcal{U}(\mathbf{R}, \mathbf{X})`, i.e. Eq. 40 of
        Niklasson, *J. Chem. Theory Comput.* **2020**, *16*, 3628,

        .. math::

            \frac{\partial\mathcal{U}}{\partial R_I} =
                2\,\mathrm{Tr}[\mathbf{h}_{R_I}\mathbf{D}]
              + \mathrm{Tr}[(2\mathbf{D} - \mathbf{P})\,\mathbf{G}_{R_I}]
              + \frac{\partial V_{nn}}{\partial R_I}
              - 2\,\mathrm{Tr}[\mathbf{Z}\mathbf{Z}^{T}
                               \mathbf{F}\mathbf{D}\mathbf{S}_{R_I}],

        where the density matrices used here are the doubled (closed-shell)
        forms ``dm_ao`` = :math:`2\mathbf{D}`, ``dm_prop`` = :math:`2\mathbf{P}`
        and ``dm_lin`` = :math:`2(2\mathbf{D} - \mathbf{P})`.

        Notes
        -----
        The two-electron term requires care. Equation 43 of the reference
        defines

        .. math::

            \mathbf{G}_{R_I} = \left.
                \frac{\partial \mathbf{G}(\mathbf{R}, \mathbf{P})}
                     {\partial R_I}\right|_{\mathbf{P}},

        i.e. the derivative of the two-electron *integrals* with respect to
        **all four** AO indices at fixed :math:`\mathbf{P}`. Writing

        .. math::

            T[X, Y]_A = \sum_{\mu \in A}
                (\nabla_{\!\mu}\,\mu\nu|\lambda\sigma)\,
                X_{\mu\nu} Y_{\lambda\sigma},

        the :math:`\partial_\mu, \partial_\nu` derivatives contribute
        :math:`T[D_{lin}, P]` while the :math:`\partial_\lambda,
        \partial_\sigma` derivatives contribute :math:`T[P, D_{lin}]`, so the
        exact two-electron force is the *symmetrized* pair

        .. math::

            \mathrm{Tr}[(2\mathbf{D} - \mathbf{P})\,\mathbf{G}_{R_I}]
                = T[D_{lin}, P] + T[P, D_{lin}].

        The backend derivative builders expose only the
        :math:`\partial_\mu` piece, which carries only the
        :math:`\lambda \leftrightarrow \sigma` permutational symmetry --
        the gradient operator breaks both :math:`\mu \leftrightarrow \nu` and
        the :math:`(\mu\nu) \leftrightarrow (\lambda\sigma)` pair exchange.
        Consequently :math:`T[X, Y] \neq T[Y, X]`, and the factor of two used
        by conventional analytic-gradient codes (which fold in the remaining
        three derivatives) is valid *only* when the density inside the
        two-electron build equals the density contracted outside it. That
        holds at self-consistency, but not for a shadow functional where the
        propagated and linearized densities differ. Both builds are therefore
        performed explicitly here.

        The one-electron, exchange-correlation and Pulay terms need no such
        treatment: ``hcore_generator`` already returns a symmetrized
        derivative, and the exchange-correlation potential is a local
        multiplicative function on the quadrature grid, so only the
        bra/ket AO-pair symmetry is involved.

        Parameters
        ----------
        fock : numpy.ndarray
            Fock matrix built from ``dm_prop``, AO basis.
        dm_ao : numpy.ndarray
            Density matrix, AO basis.
        dm_prop : numpy.ndarray, optional
            Propagated density matrix, AO basis. Defaults to ``dm_ao``.
        dm_lin : numpy.ndarray, optional
            Linearized density matrix, AO basis. Defaults to ``dm_ao``.

        Returns
        -------
        numpy.ndarray
            Total energy gradient with shape ``(n_atom, 3)``.
        """
        # Propagated and linearized density matrices
        if dm_prop is None: dm_prop = dm_ao
        if dm_lin is None: dm_lin = dm_ao

        ### Sum the per-atom contributions
        # Each term is returned by the interface already contracted with the
        # appropriate density, because some backends never expose the
        # derivative matrices themselves.
        total_grad = self.interface.compute_one_electron_gradient(
            dm_ao=dm_ao,
            fock=fock,
        )
        total_grad += self.interface.compute_two_electron_gradient(
            dm_bra=dm_prop,
            dm_ket=dm_lin,
        )
        total_grad += self.interface.compute_xc_gradient_per_atom(
            dm_prop=dm_prop,
            dm_ao=dm_ao,
        )
        total_grad += self.interface.compute_nuclear_gradient()

        return total_grad


    def update_first_level_dm(
        self,
        res_mat
    ):
        """"Compute first-level update of density matrices.

        The current residual matrix is acted upon and utilizied to
        compute an approximate Newton step and construct a more accurate
        "first-level" approximation of the dynamical variable matrix.

        Attributes updated (in order of modification):
            - `self.dm_prop`: the propagated density matrix,
            - `self.vhf`: the effective potential matrix, two-electron
                          part of the Fock matrix.
            - `self.fock`: Fock matrix, atomic orbital basis.
            - `self.fock_orth`: Fock matrix, orthogonal basis.
            - `self.mo_energy`: Molecular orbital energies.
            - `self.mo_coeff`: Molecular orbital coefficients.
            - `self.mo_occ`: Molecular orbital occupations.
            - `self.mu`: Chemical potential of system.
            - `self.dm_orth`: Density matrix, orthogonal basis.
            - `self.dm_ao`: Density matrix, atomic orbital basis.
            - `self.dm_lin`: Linearized density matrix, atomic orbital basis.

        Args:
            res_mat: current residual matrix.
        """
        ### Temporarily suppress logger output
        log.disabled = True

        ### Update dynamical variable and propagated density matrix
        kernel_action = jacobian.pseudo_inverse_action(self, res_mat)
        X_update = self.dynvar_X - kernel_action
        self.dm_prop = la.matmul(X_update, self.Sm1)

        ### Update potential and Fock matrix
        self.vhf = self.interface.compute_eff_potential_matrix(dm=self.dm_prop)
        self.fock = self.h1e + self.vhf
        self.fock_orth = la.matmul(
            la.matmul(self.Sm12.T.conj(), self.fock), self.Sm12
        )
        self.mo_energy, self.mo_coeff = la.linalg.eigh(self.fock_orth)

        ### Fermi-Dirac operator expansion (finite temperature)
        self.mo_occ, self.mu = fermi_dirac.update_fractional_occ(self)

        ### Update density matrices
        self.dm_orth = la.matmul(
            la.matmul(self.mo_coeff, la.diag(self.mo_occ)),
            self.mo_coeff.T.conj(),
        )
        self.dm_ao = la.matmul(
            la.matmul(self.Sm12, self.dm_orth),
            self.Sm12.T.conj(),
        )
        self.dm_lin = (2.0 * self.dm_ao) - self.dm_prop

        ### First-order residual matrix
        updated_dDS = la.matmul(self.dm_ao, self.S) - X_update

        log.disabled = False
        return updated_dDS


    def verify_scf_convergence(self) -> None:
        """Prepare SCF results before running MD kernel."""
        ### Determine if SCF results can be reused at the requested temperature
        if not self.interface.can_reuse_scf(temp=self.temp):
            log.info(
                " -- SCF results cannot be reused at the requested temperature."
            )

            ### Dictionary of system attributes
            scf_params: dict[str, (float | int)] = {
                "n_atom": self.n_atom,
                "n_elec": self.n_elec,
                "n_mo": self.n_mo,
                "n_occ": self.n_occ,
                "temp": self.temp,
                "frac_occ_tol": self.frac_occ_tol,
            }

            ### Include user-defined SCF parameters
            scf_params.update(
                {
                attr[4:]: getattr(self, attr)  # Remove 'scf_' prefix for key
                for attr in dir(self)
                if attr.startswith("scf_") and getattr(self, attr) is not None
                }
            )
            log.info(" -- Defined parameters for Newton-Raphson SCF.")

            ### Newton-Raphson SCF kernel
            self.interface.conv_scf = False
            nrscf = newton_raphson.NewtonRaphson(self.interface, scf_params)
            nrscf.kernel()

            ### Converged Newton-Raphson SCF
            if self.interface.get_scf_convergence():
                self.interface.temp = self.temp
                log.info("\n>> Copying attributes from Newton-Raphson SCF calculation...")

                ### Copy converged SCF quantities from NewtonRaphson object
                for key in self._scf_keys:
                    setattr(self, key, getattr(nrscf, key, None))

                    if getattr(self, key) is not None:
                        log.debug(f" -- Copied attribute: '{key}'.")
                    else:
                        log.error(f" -- Missing attribute: '{key}'.")

                ### Rebuild the energy decomposition
                # The Newton-Raphson SCF takes its energy from the backend's
                # total electronic energy, which folds any exchange-correlation
                # contribution into e2 and leaves e_xc undefined. Recomputing
                # here gives the same decomposition as the reused-SCF branch
                # below, so both entry points report energies consistently.
                self.update_energies(
                    dm_ao=self.dm_ao,
                    compute_e_nuc=True,
                    compute_e_free=True,
                )

                ### Resume MD
                log.info(" -- Resuming MD calculation.")

            ### Newton-Raphson SCF not converged (shouldn't happen?)
            else:
                log.error("Newton-Raphson SCF procedure did not converge.")

        ### Converged SCF procedure
        else:
            log.info(" -- Reusing converged SCF procedure.")

            self.S, self.Sm1, self.Sp12, self.Sm12 =\
                self.interface.compute_overlap_matrices()
            self.dm_ao = self.interface.compute_density_matrix()
            self.dm_orth = la.matmul(
                la.matmul(self.Sp12, self.dm_ao),
                self.Sp12.T.conj(),
            )
            self.dynvar_X = la.matmul(self.dm_ao, self.S)

            self.h1e = self.interface.compute_core_hamiltonian_matrix()
            self.vhf = self.interface.compute_eff_potential_matrix(dm=self.dm_ao)
            self.fock = self.h1e + self.vhf
            self.fock_orth = la.matmul(
                la.matmul(self.Sm12.T.conj(), self.fock),
                self.Sm12,
            )
            self.mo_energy, self.mo_coeff = la.linalg.eigh(self.fock_orth)
            self.mo_occ, self.mu = fermi_dirac.update_fractional_occ(self)

            self.update_energies(
                dm_ao=self.dm_ao,
                compute_e_nuc=True,
                compute_e_free=True,
            )
            log.info(" -- Initialized ShadowMD from converged SCF results.")

        return


    def end_of_timestep_scf_kernel(self):
        """"Compute converged SCF energy and gradients of system.

        A small SCF calculation is performed before moving onto the next
        simulation timestep, in order to obtain a converged SCF energy,
        as well as gradients from this minimized energy. These values
        are used to quantify the accuracy of the approximate linearized
        gradients computed during the MD simulation.
        """
        ### Dictionary of system attributes
        scf_params = {
            "n_atom": self.n_atom,
            "n_elec": self.n_elec,
            "n_mo": self.n_mo,
            "n_occ": self.n_occ,
            "temp": self.temp,
            "frac_occ_tol": self.frac_occ_tol,
            "initial_dm_ao": la.copy(self.dm_ao),
        }

        ### SCF parameters
        scf_params.update(
            {
            "max_cycle": 50,
            "energy_diff_error_tol": 1e-10,
            "res_norm_error_tol": 1e-10,
            "max_rank": self.max_rank,
            "rel_res_error_tol": 1e-8,
            }
        )

        ### Newton-Raphson SCF kernel
        self.interface.conv_scf = False
        log_disabled = log.disabled
        log.disabled = True
        try:
            nrscf = newton_raphson.NewtonRaphson(self.interface, scf_params)
            nrscf.kernel()
        finally:
            log.disabled = log_disabled

        ### Copy converged SCF quantities from NewtonRaphson object
        exact_grad = la.zeros((self.n_atom, 3))
        exact_scf_energy = la.nan
        if self.interface.get_scf_convergence():
            exact_grad = self.update_gradients(
                fock=nrscf.fock,
                dm_ao=nrscf.dm_ao,
                dm_prop=nrscf.dm_ao,
                dm_lin=nrscf.dm_ao
            )
            exact_scf_energy = nrscf.e_tot

        ### Newton-Raphson SCF not converged (shouldn't happen?)
        else:
            log.error("End of timestep SCF procedure did not converge.")

        del nrscf
        return exact_grad, exact_scf_energy
