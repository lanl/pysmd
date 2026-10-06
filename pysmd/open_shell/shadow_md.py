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

"""Unrestricted shadow molecular dynamics with fixed alpha/beta populations.

Extension of the restricted PySMD driver. Densities and dynamical
variables carry a leading spin axis; overlap matrices remain spin independent.
The open-shell drivers currently require the NumPy linear-algebra backend.
"""

import numpy as np

from pysmd.closed_shell.shadow_md import ShadowMD as RestrictedShadowMD
from pysmd.common import constants
from pysmd.common import logger
log: logger.logging.Logger = logger.getLogger(__name__)

from pysmd.lib import linalg_helper
from pysmd.open_shell import fermi_dirac, jacobian
from pysmd.open_shell.eom_params import EOMParams
from pysmd.open_shell.occupations import InitialMaximumOverlap
from pysmd.open_shell.simulation_data import SimulationData

### Default attributes
# Low-rank kernel
DEFAULT_MAX_RANK = 20


class ShadowMD(RestrictedShadowMD):
    """UHF/UKS analogue of closed_shell.ShadowMD; X_s = P_s S.

    mo_coeff is stored in the orthonormal basis; dm_ao and dm_prop are AO
    densities. Occupations are per spin, between zero and one. Nuclear
    coordinates, velocities, energies, timestep settings, and SimulationData
    follow the restricted driver.

    occupation_method='mom' tracks the previous accepted determinant at each
    integration stage; 'imom' tracks the initial guess at its original geometry.
    Both use cross-geometry AO overlaps and require zero electronic temperature.
    Pass the initial non-Aufbau determinant to kernel(dm0=...).
    """

    # The unrestricted energy keeps an explicit XC contribution.
    _scf_keys: list[str] = RestrictedShadowMD._scf_keys + ["e_xc"]

    def __init__(self, qm_interface, occupation_method="aufbau"):
        super().__init__(qm_interface)
        if occupation_method not in ("aufbau", "imom", "mom"):
            raise ValueError("MD occupation_method must be 'aufbau', 'imom', or 'mom'.")
        self.occupation_method = occupation_method
        self.occupation_reference = None
        if occupation_method != "aufbau":
            self.temp = 0.
        self.max_rank = min(DEFAULT_MAX_RANK, 2 * self.n_mo ** 2)
        self.converged_rank = []

    def initialize_general_attributes(self):
        if linalg_helper.get_linalg_backend_name() != "numpy":
            raise NotImplementedError(
                "pysmd.open_shell currently requires the NumPy linear-algebra backend."
            )
        if getattr(self.interface, "spin_channels", 1) != 2:
            raise TypeError("open_shell requires an unrestricted UHF or UKS interface.")
        self.n_atom = self.interface.get_num_atoms()
        self.n_elec = self.interface.get_num_electrons()
        self.n_mo = self.interface.get_num_basis_functions()
        self.n_occ = tuple(self.interface.get_spin_electron_counts())
        if sum(self.n_occ) != self.n_elec or any(n < 0 or n > self.n_mo for n in self.n_occ):
            raise ValueError("Invalid unrestricted spin populations.")
        if self.interface.temp is not None:
            self.temp = self.interface.temp

    def evaluate_density(self, propagated):
        """Diagonalize both Fock matrices and build fixed-population densities."""
        propagated = np.asarray(propagated, dtype=float)
        if propagated.shape != (2, self.n_mo, self.n_mo):
            raise ValueError("Unrestricted densities must have shape (2, nmo, nmo).")
        self.dm_prop = propagated.copy()
        self.h1e = self.interface.compute_core_hamiltonian_matrix()
        self.vhf = self.interface.compute_eff_potential_matrix(self.dm_prop)
        self.fock = self.h1e + self.vhf
        self.fock_orth = self.Sm12 @ self.fock @ self.Sm12
        self.mo_energy, self.mo_coeff = np.linalg.eigh(self.fock_orth)
        self.mo_occ, self.mu = self.update_occupations()
        self.dm_orth = (self.mo_coeff * self.mo_occ[:, None, :]) @ self.mo_coeff.swapaxes(-1, -2)
        self.dm_ao = self.Sm12 @ self.dm_orth @ self.Sm12
        self.dm_lin = 2 * self.dm_ao - self.dm_prop

    def update_occupations(self):
        """Use the same frozen reference in all evaluations at this geometry."""
        if self.occupation_method == "aufbau":
            return fermi_dirac.update_fractional_occ(self)
        if self.temp != 0:
            raise ValueError("Maximum-overlap DeltaSCF/MD requires temp=0 and integer occupations.")
        if self.occupation_reference is None:
            raise RuntimeError("Initialize the occupation reference by calling kernel(dm0=...).")
        self.beta = np.inf
        return self.occupation_reference.occupations(self.mo_coeff), None

    def capture_occupation_reference(self):
        """Save the accepted determinant before moving nuclei (rolling MOM)."""
        self._occupation_basis = self.interface.capture_overlap_basis()
        self._occupation_inverse_sqrt = self.Sm12.copy()
        self._occupation_anchor = InitialMaximumOverlap(self.dm_ao, self.Sp12, self.n_occ)

    def transport_occupation_reference(self):
        """Update overlap weights after a move, without changing the anchor."""
        if self.occupation_method != "aufbau":
            self.occupation_reference = self._occupation_anchor.at_geometry(
                self.interface.compute_cross_overlap(self._occupation_basis),
                self._occupation_inverse_sqrt, self.Sm12)

    def spin_electron_counts(self):
        """Return the alpha and beta electron counts Tr[D_s S]."""
        return np.einsum("sij,ji->s", self.dm_ao, self.S)

    def update_energies(self, dm_ao, dm_prop=None, dm_lin=None,
                        compute_e_nuc=False, compute_e_free=False, compute_e_kin=False):
        """Spin-resolved first-order energy E(P) + sum_s Tr[F_s(P)(D_s-P_s)]."""
        if dm_prop is None:
            dm_prop = dm_ao
        if dm_lin is None:
            dm_lin = 2 * dm_ao - dm_prop
        vj = self.interface.compute_coulomb_matrix(dm_prop)
        vk = self.interface.compute_exchange_matrix(dm_prop)
        self.e_xc, vxc = self.interface.compute_xc_terms(dm_prop)
        self.e1 = np.einsum("ij,sji->", self.h1e, dm_ao)
        self.e2 = (0.5 * np.einsum("ij,sji->", vj, dm_lin)
                   - 0.5 * np.einsum("sij,sji->", vk, dm_lin)
                   + np.einsum("sij,sji->", vxc, dm_ao - dm_prop))
        self.e_elec = self.e1 + self.e2 + self.e_xc
        if compute_e_nuc:
            self.e_nuc = self.interface.get_nuclear_energy()
        if compute_e_free:
            self.e_ent = 0.
            if self.temp > 0:
                f = self.mo_occ[(self.mo_occ > 0) & (self.mo_occ < 1)]
                self.e_ent = constants.KB * self.temp * np.sum(f * np.log(f) + (1-f) * np.log1p(-f))
        self.e_free = self.e_elec + self.e_ent
        if compute_e_kin:
            self.e_kin = 0.5 * np.sum(self.atom_mass[:, None] * self.atom_veloc ** 2)
        self.e_tot = self.e_nuc + self.e_free + self.e_kin

    def update_gradients(self, fock, dm_ao, dm_prop=None, dm_lin=None):
        """Assemble the unrestricted shadow force from the interface's contractions."""
        if dm_prop is None:
            dm_prop = dm_ao
        if dm_lin is None:
            dm_lin = 2 * dm_ao - dm_prop
        total = super().update_gradients(fock, dm_ao, dm_prop, dm_lin)
        return total

    def update_first_level_dm(self, res_mat):
        corrected_x = self.dynvar_X - jacobian.pseudo_inverse_action(self, res_mat)
        self.evaluate_density(corrected_x @ self.Sm1)
        return self.dm_ao @ self.S - corrected_x

    def _scf_parameters(self):
        params = {"temp": self.temp, "frac_occ_tol": self.frac_occ_tol}
        params.update({name[4:]: getattr(self, name) for name in vars(self)
                       if name.startswith("scf_") and getattr(self, name) is not None})
        expected = "aufbau" if self.occupation_method == "aufbau" else "imom"
        if params.get("occupation_method", expected) != expected:
            raise ValueError("Set occupation_method on ShadowMD itself so SCF and MD track the same state.")
        params["occupation_method"] = expected
        if expected == "imom" and (self.temp != 0 or params["temp"] != 0):
            raise ValueError("Maximum-overlap MD requires temp=0, including scf_temp.")
        return params

    def verify_scf_convergence(self, dm0=None):
        """Converge the unrestricted SCF from dm0 and copy the converged state."""
        from pysmd.open_shell.newton_raphson import NewtonRaphson
        params = self._scf_parameters()
        solver = NewtonRaphson(self.interface, params)
        solver.kernel(dm0=dm0)
        for key in self._scf_keys:
            value = getattr(solver, key)
            setattr(self, key, value.copy() if isinstance(value, np.ndarray) else value)
        self.dm_prop = self.dm_ao.copy()
        self.dm_lin = self.dm_ao.copy()
        self.occupation_reference = solver.occupation_reference
        if self.occupation_method != "aufbau":
            self._occupation_basis = self.interface.capture_overlap_basis()
            self._occupation_inverse_sqrt = self.Sm12.copy()
            # IMOM retains the user's initial occupied space, before SCF relaxation.
            self._occupation_anchor = self.occupation_reference

    def end_of_timestep_scf_kernel(self):
        """Return the converged SCF gradient and energy for the tracked state.

        The SCF starts from the current density and, for MOM/IMOM, keeps the
        current occupation reference fixed.
        """
        from pysmd.open_shell.newton_raphson import NewtonRaphson
        solver = NewtonRaphson(self.interface, self._scf_parameters())
        solver.kernel(dm0=self.dm_ao, occupation_reference=self.occupation_reference)
        return solver.update_gradients(solver.fock, solver.dm_ao), solver.e_tot

    def initialize_electronic_dynamics(self, dm0=None):
        """Converge the initial state and start X at rest with a flat history."""
        self.eom_params = EOMParams(self.eom_int_scheme, self.eom_l_max, self.diss_k_max)
        self.verify_scf_convergence(dm0=dm0)
        self.atom_coord = self.interface.get_atomic_coords()
        self.X_veloc = np.zeros_like(self.dynvar_X)
        self.X_accel = np.zeros_like(self.dynvar_X)
        stages = self.eom_l_max if self._uses_coeff_split_integrator() else 1
        self.X_history = np.broadcast_to(
            self.dynvar_X, (stages, self.diss_k_max + 1, *self.dynvar_X.shape),
        ).copy()

    def _uses_coeff_split_integrator(self):
        return (self.eom_int_scheme == "optimal"
                or (self.eom_int_scheme == "verlet" and self.coeff_split_verlet))

    def kernel(self, dm0=None):
        """Run MD from dm0; MOM follows accepted states, IMOM the initial guess."""
        from pysmd.open_shell import eom_integrators
        cput0: tuple[float, float] = (log.cpu_time(), log.wall_time())

        ### SCF convergence and initial state
        log.info("\n>> Converging initial unrestricted SCF...")
        self.initialize_electronic_dynamics(dm0)
        self.atom_mass = self.interface.get_atomic_masses()
        self.atom_veloc = np.zeros_like(self.atom_coord)
        self.update_energies(self.dm_ao, self.dm_prop, self.dm_lin, True, True, True)
        residual = self.dm_ao @ self.S - self.dynvar_X

        ### Simulation data object
        self.sim_data = SimulationData(
            unit=self.md_unit,
            timestep=self.md_timestep,
            num_tsteps=self.md_num_tsteps,
            num_atoms=self.n_atom,
            num_mo=self.n_mo,
        )
        self.sim_data.store_initial_system_data(
            atomic_coord=self.atom_coord,
            energy_tot=self.e_tot,
            energy_kin=self.e_kin,
        )
        self.sim_data.store_initial_electronic_data(
            self.spin_electron_counts(), self.mo_occ, np.linalg.norm(residual),
        )

        gradient = self.update_gradients(self.fock, self.dm_ao)
        integrate = (eom_integrators.integrate_coeff_split_timestep
                     if self._uses_coeff_split_integrator()
                     else eom_integrators.integrate_verlet_timestep)

        ##### SHADOW BORN-OPPENHEIMER MOLECULAR DYNAMICS #####
        cput1: tuple[float, float] = (log.cpu_time(), log.wall_time())
        log.info("\n>>>> Starting unrestricted SMD simulation <<<<")
        for ts in range(self.md_num_tsteps):
            e_prev: float = self.e_tot
            (residual, updated), gradient = integrate(self, ts, gradient, residual)
            self.update_energies(self.dm_ao, self.dm_prop, self.dm_lin, True, True, True)
            self.sim_data.update_residual_norm_data(ts, (residual, updated))
            self.sim_data.update_electronic_data(ts, self.spin_electron_counts(), self.mo_occ)
            if self.end_of_timestep_scf:
                exact_grad, exact_energy = self.end_of_timestep_scf_kernel()
                self.sim_data.update_gradient_error_data(
                    tstep=ts,
                    exact_grad=exact_grad,
                    timestep_grad=gradient,
                    exact_scf_energy=exact_energy,
                    timestep_energy=(self.e_tot - self.e_kin),
                )
            self.sim_data.update_system_data(
                tstep=ts,
                atomic_coord=self.atom_coord,
                energy_tot=self.e_tot,
                energy_kin=self.e_kin,
            )
            log.info(f"Unrestricted MD step {ts + 1}/{self.md_num_tsteps}: "
                     f"E = {self.e_tot:.12f} Ha; dE = {self.e_tot - e_prev:.3e} Ha; "
                     f"||deltaX|| = {self.sim_data.residual_norm[ts]:.3e}")
            cput1 = log.timer(f"MD timestep #{(ts+1):d}", *cput1)

        log.info(">>>>   END UNRESTRICTED MD SIMULATION   <<<<")
        cput0 = log.timer("Total MD time", *cput0)
