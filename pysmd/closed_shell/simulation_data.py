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

"""WIP: Class definition for storing molecular dynamics simulation data."""

import h5py

from pysmd.lib import linalg_helper
la = linalg_helper.get_linalg_backend()


class SimulationData:
    """Base class for storing PySMD simulation data."""


    def __init__(
        self,
        unit : str,
        timestep : float,
        num_tsteps : int,
        num_atoms : int
    ) -> None:
        """Initialize arrays to store simulation results.

        Args:
            unit: Units of time used in simulation.
            timestep: Simulation timestep size.
            num_tsteps: Total number of timesteps.
            num_atoms: Total number of atoms.
        """
        self.unit : str = unit
        self.timestep : float = timestep
        self.num_tsteps : int = num_tsteps
        self.num_atoms : int = num_atoms

        # Coordinates
        self.atomic_coords = la.zeros((self.num_tsteps,
                                       self.num_atoms,
                                       3))

        # Energies
        self.energy_tot = la.zeros(self.num_tsteps)
        self.energy_kin = la.zeros(self.num_tsteps)
        self.energy_free = la.zeros(self.num_tsteps)
        self.exact_scf_energy = la.zeros(self.num_tsteps)
        self.scf_energy_diff = la.zeros(self.num_tsteps)

        # Residual norms
        self.residual_norm = la.zeros(self.num_tsteps)
        self.first_level_residual_norm = la.zeros(self.num_tsteps)

        return


    def store_initial_system_data(
        self,
        atomic_coord,
        energy_tot : float,
        energy_kin : float
    ) -> None:
        """Store pre-MD system attributes data."""

        self.initial_atomic_coords = la.copy(atomic_coord)
        self.initial_energy_tot = float(energy_tot)
        self.initial_energy_kin = float(energy_kin)
        self.initial_energy_free = float(energy_tot - energy_kin)

        return


    def update_system_data(
        self,
        tstep : int,
        atomic_coord,
        energy_tot : float,
        energy_kin : float
    ) -> None:
        """Update system attributes data."""

        # Atomic coordinates
        self.atomic_coords[tstep] = atomic_coord

        # Total, kinetic and free energy
        self.energy_tot[tstep] = energy_tot
        self.energy_kin[tstep] = energy_kin
        self.energy_free[tstep] = energy_tot - energy_kin

        return


    def update_residual_norm_data(
        self,
        tstep : int,
        residual_tuple
    ):
        """Compute norm of provided residual matrices."""
        ### Zeroth-order residual
        residual_dDS = la.copy(residual_tuple[0])

        if not hasattr(self, "residual_norm"):
            self.residual_norm = la.zeros(self.num_tsteps)

        self.residual_norm[tstep] = la.linalg.norm(residual_dDS)

        ### First-level residual
        updated_dDS = None
        if residual_tuple[1] is not None:
            updated_dDS = la.copy(residual_tuple[1])

        if updated_dDS is not None:
            if not hasattr(self, "first_level_residual_norm"):
                self.first_level_residual_norm = la.zeros(self.num_tsteps)

            self.first_level_residual_norm[tstep] = la.linalg.norm(updated_dDS)

        return residual_dDS


    def update_gradient_error_data(
        self,
        tstep : int,
        exact_grad,
        timestep_grad,
        exact_scf_energy : float | None = None,
        timestep_energy : float | None = None
    ) -> None:
        """Compute gradient and energy errors and store data."""

        if not hasattr(self, "grad_error_norm"):
            self.grad_error_norm = la.zeros(self.num_tsteps)
        self.grad_error_norm[tstep] = la.linalg.norm(exact_grad - timestep_grad)

        if exact_scf_energy is not None:
            if not hasattr(self, "exact_scf_energy"):
                self.exact_scf_energy = la.zeros(self.num_tsteps)
            self.exact_scf_energy[tstep] = exact_scf_energy

        if exact_scf_energy is not None and timestep_energy is not None:
            if not hasattr(self, "scf_energy_diff"):
                self.scf_energy_diff = la.zeros(self.num_tsteps)
            self.scf_energy_diff[tstep] = exact_scf_energy - timestep_energy

        return


    def save_data_to_hdf5(self, filename : str) -> None:
        """Save simulation data to .hdf5 file."""
        string_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(filename, "w") as f:
            for attr_name, attr_value in self.__dict__.items():
                if isinstance(attr_value, str):
                    f.create_dataset(attr_name, data=attr_value, dtype=string_dtype)
                else:
                    f.create_dataset(attr_name, data=attr_value)

        return
