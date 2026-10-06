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

"""Ziegler-type two-determinant singlet energy estimates, gradients, and MD.

Uses separately relaxed mixed-spin and high-spin states:
E_singlet ≈ 2 E_mixed - E_triplet. This is not orbital-optimized ROKS and does
not produce a spin-pure wavefunction. ZieglerDeltaSCF evaluates single points;
ZieglerShadowMD propagates both determinants on the same surface.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from pyscf.scf.uhf import spin_square

from pysmd.common import logger
log: logger.logging.Logger = logger.getLogger(__name__)

from pysmd.open_shell.simulation_data import ZieglerSimulationData


def _assumptions_hold(s2_mixed, s2_triplet, spin_tolerance):
    return bool(abs(s2_mixed - 1) <= spin_tolerance and abs(s2_triplet - 2) <= spin_tolerance)


def _spin_square(state):
    """<S²> of a state's current determinant; orbitals are stored orthonormal."""
    coeff = state.Sm12 @ state.mo_coeff
    return spin_square([coeff[s][:, state.mo_occ[s] > .5] for s in (0, 1)], state.S)[0]


def _check_singlet_reference(qm_interface, name):
    from pysmd.interface.pyscf.uhf import UHF
    if not isinstance(qm_interface, UHF) or qm_interface.get_spin_electron_counts()[0] != qm_interface.get_spin_electron_counts()[1]:
        raise ValueError(f"{name} requires an M_s=0 UHF/UKS interface.")


@dataclass(frozen=True)
class ZieglerEstimate:
    e_mixed: float
    e_triplet: float
    s2_mixed: float
    s2_triplet: float
    assumptions_satisfied: bool

    @property
    def e_singlet(self):
        return 2 * self.e_mixed - self.e_triplet


def ziegler_estimate(e_mixed, e_triplet, s2_mixed, s2_triplet, spin_tolerance=.1):
    """Combine matched states and report departures from individual S^2 quantum numbers.
    """
    values = (e_mixed, e_triplet, s2_mixed, s2_triplet, spin_tolerance)
    if not np.isfinite(values).all() or spin_tolerance < 0:
        raise ValueError("Require finite energies/S² and a nonnegative spin tolerance.")
    valid = _assumptions_hold(s2_mixed, s2_triplet, spin_tolerance)
    if not valid:
        warnings.warn(f"Ziegler assumptions are not met within {spin_tolerance:g}: "
                      f"<S²>mixed={s2_mixed:.6f} (expected ≈1), "
                      f"<S²>triplet={s2_triplet:.6f} (expected ≈2). "
                      "The returned singlet energy is only a formal estimate.", RuntimeWarning, stacklevel=2)
    return ZieglerEstimate(float(e_mixed), float(e_triplet), float(s2_mixed), float(s2_triplet), valid)


def singlet_triplet_guesses(mo_coeff, mo_occ, overlap, occupied, virtual, spin="alpha",
                            closed_shell_tol=1e-4):
    """Matched M_s=0 and M_s=1 seeds for one promotion from a closed shell.

    Input coefficients are AO coefficients. Both seeds use the promoted
    channel's common spatial orbitals. The triplet occupies both active
    orbitals with alpha spin. This avoids trying to infer a triplet from a
    relaxed mixed-spin density, which can represent a different excitation.

    The reference counts as closed shell when its alpha and beta AO densities
    agree to closed_shell_tol. PySCF UHF breaks spin symmetry in its initial
    guess, so a converged closed shell typically retains ~1e-6 differences;
    a broken-symmetry reference differs by ~1e-1. The seeds use one channel's
    orbitals, so such residual differences do not enter them.
    """
    coeff, occ, overlap = map(np.asarray, (mo_coeff, mo_occ, overlap))
    if (spin not in ("alpha", "beta") or coeff.ndim != 3 or coeff.shape[0] != 2
            or occ.shape != (2, coeff.shape[2]) or overlap.shape != (coeff.shape[1], coeff.shape[1])
            or np.iscomplexobj(coeff) or not np.isfinite(coeff).all()
            or not np.isfinite(overlap).all() or not np.isin(occ, (0, 1)).all()):
        raise ValueError("Require real unrestricted AO orbitals and integer occupations.")
    if not np.allclose(coeff.swapaxes(-1, -2) @ overlap @ coeff, np.eye(coeff.shape[2]), atol=1e-7, rtol=0):
        raise ValueError("Orbitals must be orthonormal in the supplied AO overlap.")
    if not np.isfinite(closed_shell_tol) or closed_shell_tol < 0:
        raise ValueError("closed_shell_tol must be finite and nonnegative.")
    density = (coeff * occ[:, None, :]) @ coeff.swapaxes(-1, -2)
    if not np.array_equal(occ[0], occ[1]):
        raise ValueError("Automatic paired guesses require a closed-shell reference with equal "
                         "alpha/beta occupations; supply explicit matched densities otherwise.")
    deviation = np.abs(density[0] - density[1]).max()
    if deviation > closed_shell_tol:
        raise ValueError(f"Automatic paired guesses require a closed-shell reference, but the alpha and beta "
                         f"densities differ by {deviation:.2e} (closed_shell_tol={closed_shell_tol:g}). "
                         "The reference may be spin-symmetry broken; supply explicit matched densities, "
                         "or raise closed_shell_tol if it is only loosely converged.")
    channel = 0 if spin == "alpha" else 1
    if (not isinstance(occupied, (int, np.integer)) or not isinstance(virtual, (int, np.integer))
            or not 0 <= occupied < occ.shape[1] or not 0 <= virtual < occ.shape[1]
            or occ[channel, occupied] != 1 or occ[channel, virtual] != 0):
        raise ValueError("Promote from an occupied orbital into an empty orbital (zero-based indices).")
    mixed_occ = occ.copy()
    mixed_occ[channel, occupied], mixed_occ[channel, virtual] = 0, 1
    triplet_occ = occ.copy()
    triplet_occ[0, virtual], triplet_occ[1, occupied] = 1, 0
    common = coeff[channel]
    return tuple((common * occupations[:, None, :]) @ common.T for occupations in (mixed_occ, triplet_occ))


class ZieglerDeltaSCF:
    """Converge a matched M_s=0/M_s=1 pair and estimate its singlet surface.

    Supply both AO density guesses to kernel. Independent interfaces keep the
    spin populations, orbitals, and XC grids separate. Individual NewtonRaphson.e_tot
    values remain determinant energies; this object's e_tot is the estimate.
    """

    def __init__(self, qm_interface, scf_params=None, spin_tolerance=.1):
        from pysmd.open_shell.newton_raphson import NewtonRaphson
        _check_singlet_reference(qm_interface, "ZieglerDeltaSCF")
        if not np.isfinite(spin_tolerance) or spin_tolerance < 0:
            raise ValueError("spin_tolerance must be finite and nonnegative.")
        params = dict(scf_params or {})
        if params.get("temp", 0) != 0 or params.get("occupation_method", "imom") != "imom":
            raise ValueError("ZieglerDeltaSCF requires temp=0 and occupation_method='imom'.")
        params.update(temp=0., occupation_method="imom")
        self.mixed = NewtonRaphson(qm_interface.with_spin(0), params)
        self.triplet = NewtonRaphson(qm_interface.with_spin(2), params)
        self.spin_tolerance = spin_tolerance
        self.converged = False
        self.estimate = None
        self.e_tot = None

    def kernel(self, dm_mixed, dm_triplet):
        """Converge both determinants at the same geometry; return E_singlet."""
        self.converged = False
        self.estimate = self.e_tot = None
        if not np.allclose(self.mixed.interface.mol.atom_coords(), self.triplet.interface.mol.atom_coords(), atol=1e-12, rtol=0):
            raise ValueError("Mixed and triplet geometries must match.")
        self.mixed.kernel(dm0=dm_mixed)
        self.triplet.kernel(dm0=dm_triplet)
        spins = [_spin_square(state) for state in (self.mixed, self.triplet)]
        self.estimate = ziegler_estimate(self.mixed.e_tot, self.triplet.e_tot, *spins, self.spin_tolerance)
        self.e_tot = self.estimate.e_singlet
        self.converged = True
        return self.e_tot

    def gradient(self):
        """Derivative of the separately relaxed estimate, in Hartree/Bohr."""
        if not self.converged or not self.mixed.converged or not self.triplet.converged:
            raise RuntimeError("Converge both determinants before requesting a gradient.")
        return (2 * self.mixed.update_gradients(self.mixed.fock, self.mixed.dm_ao)
                - self.triplet.update_gradients(self.triplet.fock, self.triplet.dm_ao))


def _shared_setting(name):
    def get(self):
        return getattr(self.mixed, name)

    def set(self, value):
        for state in self.states:
            setattr(state, name, value)
    return property(get, set, doc=f"``{name}`` of both electronic states.")


class ZieglerShadowMD:
    """Shadow MD on the Ziegler singlet surface, U = 2 U_mixed - U_triplet.

    Two open_shell.ShadowMD states, the M_s=0 mixed determinant and the
    M_s=1 triplet, share one set of nuclei. Each propagates its own X with
    the unweighted single-state EOM and tracks its own occupations with MOM
    or IMOM; the weights enter only the force, 2 F_mixed - F_triplet. One
    combined density cannot replace the pair: 2 P_mixed - P_triplet is not a
    determinant density, and the energy is nonlinear in the density.

    The conserved shadow energy, sim_data.energy_tot, is E_kin plus
    2 U_mixed - U_triplet. Per-state shadow potentials, <S²>, residual norms
    (columns mixed, triplet), and occupations are stored in sim_data too,
    with the state before the first step in its initial_* attributes.
    Timestep, EOM, and kernel settings apply to both states; set scf_*
    parameters on md.mixed and md.triplet.
    """

    weights = (2., -1.)

    md_unit = _shared_setting("md_unit")
    md_timestep = _shared_setting("md_timestep")
    md_num_tsteps = _shared_setting("md_num_tsteps")
    md_total_time = _shared_setting("md_total_time")
    eom_int_scheme = _shared_setting("eom_int_scheme")
    eom_l_max = _shared_setting("eom_l_max")
    diss_k_max = _shared_setting("diss_k_max")
    coeff_split_verlet = _shared_setting("coeff_split_verlet")
    first_level_dm_update = _shared_setting("first_level_dm_update")
    max_rank = _shared_setting("max_rank")
    rel_res_error_tol = _shared_setting("rel_res_error_tol")

    def __init__(self, qm_interface, occupation_method="mom", spin_tolerance=.1):
        from pysmd.open_shell.shadow_md import ShadowMD
        _check_singlet_reference(qm_interface, "ZieglerShadowMD")
        if occupation_method not in ("mom", "imom"):
            raise ValueError("ZieglerShadowMD tracks two non-Aufbau determinants: use 'mom' or 'imom'.")
        if not np.isfinite(spin_tolerance) or spin_tolerance < 0:
            raise ValueError("spin_tolerance must be finite and nonnegative.")
        self.mixed = ShadowMD(qm_interface.with_spin(0), occupation_method)
        self.triplet = ShadowMD(qm_interface.with_spin(2), occupation_method)
        self.states = (self.mixed, self.triplet)
        self.occupation_method = occupation_method
        self.spin_tolerance = spin_tolerance
        self.end_of_timestep_scf = False
        self.sim_data = None
        self.e_kin = self.e_pot = self.e_tot = None

    @property
    def _md_timestep(self):
        return self.mixed._md_timestep

    @property
    def eom_params(self):
        return self.mixed.eom_params

    def _combine(self, values):
        return sum(weight * value for weight, value in zip(self.weights, values))

    def kernel(self, dm_mixed, dm_triplet):
        """Run MD from matched mixed/triplet guesses, e.g. singlet_triplet_guesses."""
        from pysmd.open_shell import eom_integrators
        if not np.allclose(self.mixed.interface.mol.atom_coords(), self.triplet.interface.mol.atom_coords(),
                           atol=1e-12, rtol=0):
            raise ValueError("Mixed and triplet geometries must match.")
        for state, dm0 in zip(self.states, (dm_mixed, dm_triplet)):
            state.initialize_electronic_dynamics(dm0)
        self.atom_mass = self.mixed.interface.get_atomic_masses()
        self.atom_coord = self.mixed.atom_coord.copy()
        self.atom_veloc = np.zeros_like(self.atom_coord)
        self.sim_data = ZieglerSimulationData(self.md_unit, self.md_timestep, self.md_num_tsteps,
                                              self.mixed.n_atom, self.mixed.n_mo)
        self._warned = False
        gradient = self._combine(state.update_gradients(state.fock, state.dm_ao) for state in self.states)
        residuals = [(state.dm_ao @ state.S - state.dynvar_X, None) for state in self.states]
        integrate = (eom_integrators.weighted_coeff_split_timestep
                     if self.mixed._uses_coeff_split_integrator()
                     else eom_integrators.weighted_verlet_timestep)
        self._record(None, residuals, gradient)
        for step in range(self.md_num_tsteps):
            residuals, gradient = integrate(self, self.states, self.weights, gradient,
                                            [residual for residual, _ in residuals])
            self._record(step, residuals, gradient)
        return self.e_tot

    def _record(self, step, residuals, gradient):
        """Store the state after ``step``; ``None`` stores the initial state."""
        data = self.sim_data
        potentials = []
        for state in self.states:
            state.update_energies(state.dm_ao, state.dm_prop, state.dm_lin, compute_e_nuc=True, compute_e_free=True)
            potentials.append(state.e_nuc + state.e_free)
        self.e_kin = 0.5 * np.sum(self.atom_mass[:, None] * self.atom_veloc ** 2)
        self.e_pot = self._combine(potentials)
        self.e_tot = self.e_kin + self.e_pot
        s2_values = [_spin_square(state) for state in self.states]
        valid = _assumptions_hold(*s2_values, self.spin_tolerance)
        label = "initial state" if step is None else f"Step {step + 1}"
        if not valid and not self._warned:
            warnings.warn(f"{label}: Ziegler assumptions are not met within {self.spin_tolerance:g} "
                          f"(<S²>mixed={s2_values[0]:.6f}, <S²>triplet={s2_values[1]:.6f}); "
                          "see sim_data.ziegler_assumptions_satisfied.", RuntimeWarning, stacklevel=3)
            self._warned = True
        residual_norms = [np.linalg.norm(residual) for residual, _ in residuals]
        counts = [state.spin_electron_counts() for state in self.states]
        occupations = [state.mo_occ for state in self.states]

        if step is None:
            data.store_initial_system_data(self.atom_coord, self.e_tot, self.e_kin)
            data.store_initial_electronic_data(counts, occupations, residual_norms)
            data.store_initial_state_data(potentials, s2_values, valid)
            log.info(f"Ziegler MD initial state: E = {self.e_tot:.12f} Ha")
            return

        data.residual_norm[step] = residual_norms
        data.first_level_residual_norm[step] = [0. if updated is None else np.linalg.norm(updated)
                                                for _, updated in residuals]
        data.update_electronic_data(step, counts, occupations)
        data.update_state_data(step, potentials, s2_values, valid)
        if self.end_of_timestep_scf:
            exact = [state.end_of_timestep_scf_kernel() for state in self.states]
            data.update_gradient_error_data(
                tstep=step,
                exact_grad=self._combine(grad for grad, _ in exact),
                timestep_grad=gradient,
                exact_scf_energy=self._combine(energy for _, energy in exact),
                timestep_energy=self.e_pot,
            )
        data.update_system_data(step, self.atom_coord, self.e_tot, self.e_kin)
        log.info(f"Ziegler MD step {step + 1}/{self.md_num_tsteps}: E = {self.e_tot:.12f} Ha")
