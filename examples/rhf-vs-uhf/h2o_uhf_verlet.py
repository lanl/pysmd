import h5py
import numpy as np
np.set_printoptions(
    linewidth = 150,
    edgeitems = 10,
    suppress = True
)
from pathlib import Path

from pyscf import gto
from pyscf import scf

from pysmd import interface
from pysmd.open_shell import shadow_md

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def run_xlbomd(
    basis_set = "cc-pvdz",
    temp = 500,
    md_time_unit = "fs",
    md_num_tsteps = 300,
    md_dt = 0.2,
    verb = 4,
    first_level_update = False,
    endstep_scf = False,
):
    """Run XLBOMD calculation for molecular hydrogen."""


    # PySCF molecule object
    mol = gto.Mole()
    mol.atom = r"""
                O        0.000000    0.000000    0.117790
                H        0.000000    0.755453   -0.471161
                H        0.000000   -0.755453   -0.471161
                """
    mol.basis = basis_set
    mol.verbose = verb
    mol.build()

    # RHF object
    mf = scf.UHF(mol)
    mf.max_cycle = 0

    mf.kernel()

    ###############

    # PySCF interface and MD object
    qm_interface = interface.pyscf.UHF(pyscf_mf=mf)
    xlbomd = shadow_md.ShadowMD(qm_interface=qm_interface)

    # Temperature
    xlbomd.temp = temp

    # MD parameters
    xlbomd.md_unit = md_time_unit
    # Set the timestep first: changing it keeps the total time fixed and
    # recomputes the number of steps.
    xlbomd.md_timestep = md_dt
    xlbomd.md_num_tsteps = md_num_tsteps
    xlbomd.first_level_dm_update = first_level_update
    xlbomd.end_of_timestep_scf = endstep_scf

    # NRSCF kernel parameters
    xlbomd.scf_max_rank = mol.nao_nr()
    xlbomd.scf_energy_diff_error_tol = 1e-8
    xlbomd.scf_rel_res_error_tol = 1e-4

    # XLBOMD kernel parameters
    xlbomd.max_rank = mol.nao_nr()
    xlbomd.rel_res_error_tol = 1e-4

    # Run kernel
    xlbomd.kernel()


    ###############


    # Store values to return
    # Per-step arrays hold the propagated states; the state before the first
    # step is stored separately, so prepend it.
    sim_data = xlbomd.sim_data
    total_time = np.arange(sim_data.num_tsteps + 1) * xlbomd.md_timestep
    energy_kin = np.concatenate(([sim_data.initial_energy_kin], sim_data.energy_kin))
    energy_free = np.concatenate(([sim_data.initial_energy_free], sim_data.energy_free))
    # MD starts from a converged SCF with X = DS. The closed-shell driver does
    # not store that (vanishing) initial residual.
    initial_res_norm = getattr(sim_data, "initial_residual_norm", 0.0)
    res_norm = np.concatenate(([initial_res_norm], sim_data.residual_norm))

    stored_time_params = {
        "__time_unit": xlbomd.sim_data.unit,
        "__dt": xlbomd._md_timestep,
        "__num_tsteps": xlbomd.sim_data.num_tsteps,
        "__simulation_time": xlbomd._md_total_time,
    }

    del xlbomd
    return total_time, energy_free, energy_kin, res_norm, stored_time_params


def plot_results(simulation_time, e_free, e_kin, residual_norm, fig_name):
    """Plot the results of the XLBOMD simulation."""
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9

    fig = plt.figure(figsize=(8, 12))

    # Free energy
    plt.subplot(5, 1, 1)
    plt.plot(simulation_time, e_free, ".-", color="red", markersize=4, linewidth=1)
    plt.xlabel(r"\textbf{Time (fs)}")
    plt.ylabel(r"\textbf{$E_{\mathrm{free}}$ (Ha)}")
    plt.grid(True, alpha=0.3, linestyle='--')

    # Kinetic energy
    plt.subplot(5, 1, 2)
    plt.plot(simulation_time, e_kin, ".-", color="orange", markersize=4, linewidth=1)
    plt.xlabel(r"\textbf{Time (fs)}")
    plt.ylabel(r"\textbf{$E_{\mathrm{kin}}$ (Ha)}")
    plt.grid(True, alpha=0.3, linestyle='--')

    # Total energy
    e_tot = e_free + e_kin
    plt.subplot(5, 1, 3)
    plt.plot(simulation_time, e_tot, ".-", color="green", markersize=4, linewidth=1)
    plt.xlabel(r"\textbf{Time (fs)}")
    plt.ylabel(r"\textbf{$E_{\mathrm{total}}$ (Ha)}")
    plt.grid(True, alpha=0.3, linestyle='--')

    # Total energy fluctuations
    plt.subplot(5, 1, 4)
    plt.plot(simulation_time, e_tot - e_tot[0], ".-", color="blue", markersize=4, linewidth=1)
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3e'))
    plt.xlabel(r"\textbf{Time (fs)}")
    plt.ylabel(r"\textbf{$\Delta E_{\mathrm{total}}$ (Ha)}")
    plt.grid(True, alpha=0.3, linestyle='--')

    # Norm of dp2dt2 residual (0)
    plt.subplot(5, 1, 5)
    plt.plot(simulation_time, residual_norm, ".-", color="purple", markersize=4, linewidth=1)
    plt.xlabel(r"\textbf{Time (fs)}")
    plt.ylabel(r"\textbf{$\|\mathbf{r}\|$ (residual)}")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    #plt.show()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return


if __name__ == "__main__":

    # Basis set
    #basis = "6-31g"
    basis = "cc-pvdz"

    # Temperature, Kelvin
    #temperature = 1500
    temperature = 500

    # Total time
    sim_time = 30.0 # in fs

    # Timestep sizes
    dt1 = 0.4
    dt2 = dt1 / 2.0
    dt3 = dt2 / 2.0

    # Number of steps
    nsteps1 = int(sim_time / dt1)
    nsteps2 = int(sim_time / dt2)
    nsteps3 = int(sim_time / dt3)

    # First-level residual update
    first_level = False
    #first_level = True

    # Convergence SCF
    timestep_scf = False
    #timestep_scf = True


    # Timestep 1
    total_time1, energy_free1, energy_kin1, res_norm1, params1 = run_xlbomd(
        basis_set = basis,
        temp = temperature,
        md_time_unit = "fs",
        md_num_tsteps = nsteps1,
        md_dt = dt1,
        verb=4,
        first_level_update = first_level,
        endstep_scf = timestep_scf,
    )

    # Timestep 2
    total_time2, energy_free2, energy_kin2, res_norm2, params2 = run_xlbomd(
        basis_set = basis,
        temp = temperature,
        md_time_unit = "fs",
        md_num_tsteps = nsteps2,
        md_dt = dt2,
        verb = 4,
        first_level_update = first_level,
        endstep_scf = timestep_scf,
    )

    # Timestep 3
    total_time3, energy_free3, energy_kin3, res_norm3, params3 = run_xlbomd(
        basis_set = basis,
        temp = temperature,
        md_time_unit = "fs",
        md_num_tsteps = nsteps3,
        md_dt = dt3,
        verb = 4,
        first_level_update = first_level,
        endstep_scf = timestep_scf,
    )

    # Create output directory
    base_dir_name = f"h2o_verlet_uhf_firstlevel{first_level}_{basis}"
    dir_name = base_dir_name
    counter = 1
    while Path(dir_name).exists():
        counter += 1
        dir_name = f"{base_dir_name}{counter}"
    Path(dir_name).mkdir()

    # Plot figures
    plot_results(total_time1, energy_free1, energy_kin1, res_norm1,
                 fig_name=f"{dir_name}/h2o_dt1_{dt1}.pdf")
    plot_results(total_time2, energy_free2, energy_kin2, res_norm2,
                 fig_name=f"{dir_name}/h2o_dt2_{dt2}.pdf")
    plot_results(total_time3, energy_free3, energy_kin3, res_norm3,
                 fig_name=f"{dir_name}/h2o_dt3_{dt3}.pdf")

    # Dataset lists
    group_names = [
        f"dt1_{dt1}_tsteps_{nsteps1}",
        f"dt2_{dt2}_tsteps_{nsteps2}",
        f"dt3_{dt3}_tsteps_{nsteps3}",
    ]
    times = [
        total_time1,
        total_time2,
        total_time3,
    ]
    free_energies = [
        energy_free1,
        energy_free2,
        energy_free3,
    ]
    kinetic_energies = [
        energy_kin1,
        energy_kin2,
        energy_kin3,
    ]
    norms = [
        res_norm1,
        res_norm2,
        res_norm3,
    ]
    parameters = [
        params1,
        params2,
        params3,
    ]

    # Open (or create) the HDF5 file
    with h5py.File(f"{dir_name}/h2o_data.h5", "a") as f:

        # Create groups for each timestep
        for i in range(3):
            grp = f.require_group(group_names[i])

            # Helper function to create or overwrite dataset
            def create_or_overwrite(group, name, data):
                if name in group:
                    del group[name]
                group.create_dataset(name, data=data)

            # Save arrays as datasets in the group
            create_or_overwrite(grp, "total_time", data=times[i])
            create_or_overwrite(grp, "energy_free", data=free_energies[i])
            create_or_overwrite(grp, "energy_kin", data=kinetic_energies[i])
            create_or_overwrite(grp, "res_norm", data=norms[i])

            # Store parameters as attributes
            for key, value in parameters[i].items():
                create_or_overwrite(grp, f"{key}", value)
