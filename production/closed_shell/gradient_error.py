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

"""Run closed-shell ShadowMD trajectories for gradient error studies."""

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyscf import dft, gto, scf

from pysmd import interface
from pysmd.closed_shell import shadow_md


plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{bm}")
plt.rc("font", family="serif", weight="bold")
plt.rc("axes", labelweight="bold")


def ensure_plot_dir(plot_dir):
    """Ensure the plot output directory exists."""
    os.makedirs(plot_dir, exist_ok=True)
    return


def timestep_tag(timestep_fs):
    """Return a filesystem-safe timestep label."""
    return f"{timestep_fs:.3f}".replace(".", "p") + "fs"


def non_overwriting_filename(filename):
    """Return filename or a numbered variant that does not already exist."""
    root, ext = os.path.splitext(filename)

    if not os.path.exists(filename):
        return filename

    run_idx = 1
    while True:
        candidate = f"{root}_run{run_idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        run_idx += 1


def simulation_data_filename(
    output_dir,
    mol_name,
    wf_type,
    eom_scheme,
    timestep_fs,
    first_level_dm_update,
):
    """Return the canonical SimulationData filename for one run."""
    return os.path.join(
        output_dir,
        f"{mol_name}_{wf_type}_{eom_scheme}_"
        f"dt_{timestep_tag(timestep_fs)}_"
        f"first_level_{int(first_level_dm_update)}.h5",
    )


def combine_simulation_hdf5_files(input_records, output_file):
    """Combine individual SimulationData HDF5 files into one grouped file."""
    with h5py.File(output_file, "w") as fout:
        for filename, timestep_fs, first_level_dm_update in input_records:
            first_level_tag = f"first_level_{int(first_level_dm_update)}"
            dt_tag = f"dt_{timestep_tag(timestep_fs)}"
            group = fout.require_group(first_level_tag).create_group(dt_tag)

            group.attrs["source_file"] = os.fspath(filename)
            group.attrs["timestep_fs"] = timestep_fs
            group.attrs["first_level_dm_update"] = first_level_dm_update

            with h5py.File(filename, "r") as fin:
                for dataset_name in fin:
                    fin.copy(dataset_name, group)

    return output_file


def plot_gradient_error(
    hdf5_file,
    plot_dir,
    scheme_tag,
    average_last_n=200,
):
    """Plot averaged gradient error norms as a function of timestep."""
    ensure_plot_dir(plot_dir)

    first_level_data = {
        0: {"timesteps": [], "avg_grad_error_norm": []},
        1: {"timesteps": [], "avg_grad_error_norm": []},
    }

    with h5py.File(hdf5_file, "r") as h5_file:
        for first_level in first_level_data:
            group_name = f"first_level_{first_level}"
            if group_name not in h5_file:
                continue

            for dt_group_name in h5_file[group_name]:
                dt_group = h5_file[group_name][dt_group_name]
                timestep_fs = float(dt_group.attrs["timestep_fs"])
                grad_error_norm = dt_group["grad_error_norm"][:]

                if grad_error_norm.size < average_last_n:
                    raise ValueError(
                        "Gradient error trajectory has fewer samples than "
                        "average_last_n."
                    )

                avg_grad_error_norm = np.average(
                    grad_error_norm[-average_last_n:]
                )
                if avg_grad_error_norm <= 0.0:
                    raise ValueError(
                        "grad_error_norm average is nonpositive for "
                        f"{group_name}/{dt_group_name}."
                    )

                first_level_data[first_level]["timesteps"].append(timestep_fs)
                first_level_data[first_level]["avg_grad_error_norm"].append(
                    avg_grad_error_norm
                )

    fig, ax = plt.subplots(figsize=(8, 5))

    plot_specs = {
        0: {
            "linestyle": "-",
            "marker": "o",
            "color": "black",
            "label": r"\textbf{Verlet}",
        },
        1: {
            "linestyle": "--",
            "marker": "o",
            "color": "tab:blue",
            "label": r"\textbf{Verlet + first-level update}",
        },
    }

    for first_level, data in first_level_data.items():
        timesteps = np.array(data["timesteps"])
        avg_grad_error_norm = np.array(data["avg_grad_error_norm"])

        if timesteps.size == 0:
            continue

        sort_idx = np.argsort(timesteps)
        specs = plot_specs[first_level]
        ax.plot(
            np.log10(timesteps[sort_idx]),
            np.log10(avg_grad_error_norm[sort_idx]),
            linestyle=specs["linestyle"],
            marker=specs["marker"],
            color=specs["color"],
            linewidth=1.5,
            label=specs["label"],
        )

    ax.set_xlabel(r"$\bm{\log_{10}(\Delta t/\mathrm{fs})}$")
    ax.set_ylabel(
        r"$\bm{\log_{10}\!\left(\left<\left\|\nabla E_{\mathrm{SCF}}-\nabla E\right\|\right>\right)}$"
    )
    ax.legend(fontsize=12, fancybox=False, framealpha=0.6, edgecolor="black")

    fig.tight_layout()

    fig_name = os.path.join(plot_dir, f"{scheme_tag}_gradient_error.pdf")
    fig.savefig(fig_name)
    plt.close(fig)
    print(f"[PLOT] Saved gradient error plot to {fig_name}")

    return


def run_md_for_timestep(
    xyz_file,
    wf_type,
    timestep_fs,
    xc=None,
    num_tsteps=1000,
    average_last_n=200,
    eom_scheme="verlet",
    eom_l_max=2,
    diss_k_max=6,
    end_of_timestep_scf=True,
    first_level_dm_update=False,
):
    """Run one closed-shell ShadowMD trajectory and return the MD object."""

    # Local constants for this figure workflow.
    # basis = "cc-pvdz"
    basis = "6-31g"
    temperature = 500.0
    rel_res_error_tol = 1e-4
    scf_rel_res_error_tol = 1e-5
    scf_energy_diff_error_tol = 1e-8

    if not isinstance(wf_type, str):
        raise TypeError("wf_type must be 'rhf' or 'rks'.")

    wf_type = wf_type.lower()
    if wf_type not in ("rhf", "rks"):
        raise ValueError("wf_type must be 'rhf' or 'rks'.")

    if type(average_last_n) is not int or average_last_n <= 0:
        raise ValueError("average_last_n must be a positive integer.")

    if num_tsteps < average_last_n:
        raise ValueError("num_tsteps must be greater than or equal to average_last_n.")

    mol = gto.Mole()
    mol.atom = os.fspath(xyz_file)
    mol.basis = basis
    mol.build()

    # Only pure LDA and GGA functionals are supported, e.g. "LDA,VWN5",
    # "PBE", "BLYP". Hybrid, range-separated, meta-GGA and non-local
    # correlation functionals are rejected by the RKS interface.
    if xc is not None:
        mf = dft.RKS(mol, xc=xc)
    elif wf_type == "rks":
        mf = dft.RKS(mol)
    else:
        mf = scf.RHF(mol)

    qm_interface = interface.pyscf.PyscfDriver(pyscf_mf=mf)

    md = shadow_md.ShadowMD(qm_interface=qm_interface)

    md.temp = temperature
    md.eom_int_scheme = eom_scheme
    md.eom_l_max = eom_l_max
    md.diss_k_max = diss_k_max

    md.md_unit = "fs"
    md.md_total_time = timestep_fs * num_tsteps
    md.md_timestep = timestep_fs
    md.md_num_tsteps = num_tsteps

    md.end_of_timestep_scf = end_of_timestep_scf
    md.first_level_dm_update = first_level_dm_update

    md.rel_res_error_tol = rel_res_error_tol
    md.scf_rel_res_error_tol = scf_rel_res_error_tol
    md.scf_energy_diff_error_tol = scf_energy_diff_error_tol

    md.kernel()

    return md


if __name__ == "__main__":
    mol_name = "h2"
    mol_file = f"xyz/{mol_name}.xyz"
    #wf_type = "rks"
    wf_type = "rhf"
    xc = None

    timestep_list = [0.4, 0.2, 0.1]
    first_level_values = [False, True]

    num_tsteps = 1000
    average_last_n = 200
    eom_scheme = "verlet"
    eom_l_max = 2
    diss_k_max = 6
    end_of_timestep_scf = True

    output_dir = os.path.join("data", mol_name)
    os.makedirs(output_dir, exist_ok=True)

    input_records = []
    created_new_data = False
    for first_level in first_level_values:
        for timestep in timestep_list:
            filename = simulation_data_filename(
                output_dir=output_dir,
                mol_name=mol_name,
                wf_type=wf_type,
                eom_scheme=eom_scheme,
                timestep_fs=timestep,
                first_level_dm_update=first_level,
            )

            if os.path.exists(filename):
                print(f"Using existing {filename}")
                input_records.append((filename, timestep, first_level))
                continue

            md = run_md_for_timestep(
                xyz_file=mol_file,
                wf_type=wf_type,
                timestep_fs=timestep,
                xc=xc,
                num_tsteps=num_tsteps,
                average_last_n=average_last_n,
                eom_scheme=eom_scheme,
                eom_l_max=eom_l_max,
                diss_k_max=diss_k_max,
                end_of_timestep_scf=end_of_timestep_scf,
                first_level_dm_update=first_level,
            )

            if md.sim_data is None:
                raise RuntimeError("MD simulation finished without SimulationData.")

            md.sim_data.save_data_to_hdf5(filename)
            print(f"Saved {filename}")

            input_records.append((filename, timestep, first_level))
            created_new_data = True
            del md

    combined_file = os.path.join(
        output_dir,
        f"{mol_name}_{wf_type}_{eom_scheme}_simulation_data.h5",
    )
    if os.path.exists(combined_file) and not created_new_data:
        print(f"Using existing {combined_file}")
    else:
        if os.path.exists(combined_file):
            combined_file = non_overwriting_filename(combined_file)
        combined_file = combine_simulation_hdf5_files(input_records, combined_file)
        print(f"Saved {combined_file}")

    PLOT_DATA = True

    if PLOT_DATA:
        plotting_dir = os.path.join(output_dir, "plots")
        plot_gradient_error(
            hdf5_file=combined_file,
            plot_dir=plotting_dir,
            scheme_tag=f"{wf_type}_{eom_scheme}",
            average_last_n=average_last_n,
        )
