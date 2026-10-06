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

"""Simulation data for unrestricted and spin-corrected shadow MD.

As in :class:`pysmd.closed_shell.simulation_data.SimulationData`, per-step
arrays have one row for each of the ``num_tsteps`` propagated states and the
state before the first step is kept in ``initial_*`` attributes.
:meth:`SimulationData.full_trajectory` prepends that initial state, giving
``num_tsteps + 1`` rows.
"""

import numpy as np

from pysmd.closed_shell import simulation_data


class SimulationData(simulation_data.SimulationData):
    """Adds alpha/beta electron counts and occupations at every step.

    ``state_shape`` adds leading state axes after the step axis, e.g. ``(2,)``
    for the (mixed, triplet) pair of a Ziegler trajectory.
    """


    def __init__(
        self,
        unit: str,
        timestep: float,
        num_tsteps: int,
        num_atoms: int,
        num_mo: int,
        state_shape: tuple[int, ...] = (),
    ) -> None:
        super().__init__(unit, timestep, num_tsteps, num_atoms)
        self.state_shape = tuple(state_shape)

        # Residual norms, one column per electronic state
        self.residual_norm = np.zeros((num_tsteps, *self.state_shape))
        self.first_level_residual_norm = np.zeros((num_tsteps, *self.state_shape))

        # Spin populations and occupations
        self.spin_electron_counts = np.zeros((num_tsteps, *self.state_shape, 2))
        self.mo_occ = np.zeros((num_tsteps, *self.state_shape, 2, num_mo))

        return


    def store_initial_electronic_data(
        self,
        spin_electron_counts,
        mo_occ,
        residual_norm,
    ) -> None:
        """Store pre-MD electron counts, occupations, and residual norms."""
        self.initial_spin_electron_counts = np.array(spin_electron_counts, dtype=float)
        self.initial_mo_occ = np.array(mo_occ, dtype=float)
        self.initial_residual_norm = np.array(residual_norm, dtype=float)
        self.initial_first_level_residual_norm = np.zeros_like(self.initial_residual_norm)

        return


    def update_electronic_data(
        self,
        tstep: int,
        spin_electron_counts,
        mo_occ,
    ) -> None:
        """Store electron counts and occupations of the propagated state."""
        self.spin_electron_counts[tstep] = spin_electron_counts
        self.mo_occ[tstep] = mo_occ

        return


    def full_trajectory(self, name: str):
        """Return ``initial_<name>`` followed by the per-step values of ``name``."""
        initial = getattr(self, f"initial_{name}", None)
        if initial is None:
            raise KeyError(f"No initial value is stored for {name!r}.")
        values = np.asarray(getattr(self, name))
        return np.concatenate((np.asarray(initial, dtype=values.dtype)[None], values))


class ZieglerSimulationData(SimulationData):
    """Simulation data for a Ziegler (mixed, triplet) trajectory.

    Residual norms, electron counts, and occupations carry a state axis
    ordered (mixed, triplet). Per-state shadow potentials, <S²> values, and
    whether the Ziegler assumptions hold are stored at every step.
    """

    _state_names: tuple[str, ...] = (
        "energy_mixed", "energy_triplet", "s2_mixed", "s2_triplet",
    )


    def __init__(
        self,
        unit: str,
        timestep: float,
        num_tsteps: int,
        num_atoms: int,
        num_mo: int,
    ) -> None:
        super().__init__(unit, timestep, num_tsteps, num_atoms, num_mo, state_shape=(2,))
        for name in self._state_names:
            setattr(self, name, np.zeros(num_tsteps))
        self.ziegler_assumptions_satisfied = np.zeros(num_tsteps, dtype=bool)

        return


    def store_initial_state_data(self, energies, s2_values, valid: bool) -> None:
        """Store pre-MD per-state shadow potentials, <S²>, and diagnostics."""
        for name, value in zip(self._state_names, (*energies, *s2_values)):
            setattr(self, f"initial_{name}", float(value))
        self.initial_ziegler_assumptions_satisfied = bool(valid)

        return


    def update_state_data(self, tstep: int, energies, s2_values, valid: bool) -> None:
        """Store per-state shadow potentials, <S²>, and diagnostics."""
        for name, value in zip(self._state_names, (*energies, *s2_values)):
            getattr(self, name)[tstep] = value
        self.ziegler_assumptions_satisfied[tstep] = valid

        return
