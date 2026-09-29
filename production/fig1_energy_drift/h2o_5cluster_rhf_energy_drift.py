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

"""Compare RHF BOMD and shadow-MD energy drift for a five-water cluster."""

import argparse
import csv
import io
import logging
from pathlib import Path

import matplotlib
import numpy as np
from pyscf import gto, md, scf

from pysmd import interface
from pysmd.closed_shell import shadow_md
from pysmd.common import constants


matplotlib.use("Agg")
import matplotlib.pyplot as plt


GEOMETRY_FILE = Path(__file__).with_name("h2o_5cluster.xyz")
TEMPERATURE_K = 300.0
REL_RES_ERROR_TOL = 1e-5
DEFAULT_PLOT_SMD_TOL = 1e-2

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)


def build_molecule(basis, verbose=4):
    """Build the five-water cluster used by every trajectory."""
    return gto.M(
        atom=str(GEOMETRY_FILE),
        basis=basis,
        unit="Angstrom",
        verbose=verbose,
    )


def converged_rhf(mol, conv_tol=1e-12):
    """Return an RHF object with a tightly converged reference density."""
    mf = scf.RHF(mol)
    scf.addons.smearing_(
        mf,
        sigma=constants.KB * TEMPERATURE_K,
        method="fermi",
    )
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = np.sqrt(conv_tol)
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("The initial RHF calculation did not converge.")
    return mf


def run_pyscf_bomd(basis, conv_tol, timestep_fs, steps):
    """Run conventional PySCF RHF Born-Oppenheimer MD."""
    mol = build_molecule(basis)
    mf = converged_rhf(mol)

    # The initial electronic state is converged identically for every run.
    # Only the convergence criteria used during MD are varied.
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = np.sqrt(conv_tol)

    records = []

    def collect_frame(envs):
        frame = envs["current_frame"]
        records.append(
            (
                frame.time * constants.AU2FS,
                frame.epot,
                frame.ekin,
                frame.etot,
                envs["scanner"].base.cycles,
            )
        )

    dynamics = md.NVE(
        mf,
        dt=timestep_fs / constants.AU2FS,
        steps=steps + 1,
        veloc=np.zeros((mol.natm, 3)),
        callback=collect_frame,
        verbose=0,
    )
    dynamics.stdout = io.StringIO()
    dynamics.kernel(dump_flags=False)

    data = np.asarray(records, dtype=float)
    return {
        "method": "PySCF BOMD",
        "conv_tol": conv_tol,
        "time_fs": data[:, 0],
        "potential": data[:, 1],
        "kinetic": data[:, 2],
        "total": data[:, 3],
        "scf_cycles": data[:, 4],
    }


def run_pysmd(basis, timestep_fs, steps, rel_res_error_tol=REL_RES_ERROR_TOL):
    """Run RHF shadow MD using the PySCF interface."""
    mol = build_molecule(basis)
    mf = converged_rhf(mol)
    qm_interface = interface.pyscf.PyscfDriver(pyscf_mf=mf)
    dynamics = shadow_md.ShadowMD(qm_interface=qm_interface)

    dynamics.temp = TEMPERATURE_K
    dynamics.md_unit = "fs"
    dynamics.md_timestep = timestep_fs
    dynamics.md_num_tsteps = steps
    dynamics.eom_int_scheme = "verlet"
    dynamics.first_level_dm_update = True
    dynamics.end_of_timestep_scf = False

    dynamics.max_rank = mol.nao_nr()
    dynamics.rel_res_error_tol = rel_res_error_tol
    dynamics.scf_max_rank = mol.nao_nr()
    dynamics.scf_energy_diff_error_tol = 1e-10
    dynamics.scf_rel_res_error_tol = 1e-7
    dynamics.kernel()

    sim_data = dynamics.sim_data
    if sim_data is None:
        raise RuntimeError("PySMD did not return simulation data.")

    time_fs = np.arange(steps + 1) * timestep_fs
    potential = np.concatenate(
        ([sim_data.initial_energy_free], np.asarray(sim_data.energy_free))
    )
    kinetic = np.concatenate(
        ([sim_data.initial_energy_kin], np.asarray(sim_data.energy_kin))
    )
    total = np.concatenate(
        ([sim_data.initial_energy_tot], np.asarray(sim_data.energy_tot))
    )

    return {
        "method": "PySMD shadow MD",
        "conv_tol": None,
        "smd_tol": rel_res_error_tol,
        "time_fs": time_fs,
        "potential": potential,
        "kinetic": kinetic,
        "total": total,
        "scf_cycles": np.full(steps + 1, np.nan),
    }


def trajectory_name(trajectory):
    """Return a short, filesystem-safe trajectory name."""
    if trajectory["method"] == "SMD":
        return f"pysmd_shadow_tol_{trajectory['smd_tol']:.0e}"
    if trajectory["conv_tol"] is None:
        return "pysmd_shadow"
    return f"pyscf_bomd_tol_{trajectory['conv_tol']:.0e}"


def write_trajectory(trajectory, output_dir):
    """Write one trajectory to CSV."""
    filename = output_dir / f"{trajectory_name(trajectory)}.csv"
    initial_energy = trajectory["total"][0]

    with filename.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_fs",
                "potential_hartree",
                "kinetic_hartree",
                "total_hartree",
                "delta_total_hartree",
                "scf_cycles",
            ]
        )
        for time, potential, kinetic, total, cycles in zip(
            trajectory["time_fs"],
            trajectory["potential"],
            trajectory["kinetic"],
            trajectory["total"],
            trajectory["scf_cycles"],
        ):
            writer.writerow(
                [
                    time,
                    potential,
                    kinetic,
                    total,
                    total - initial_energy,
                    "" if np.isnan(cycles) else int(cycles),
                ]
            )

    return filename


def read_trajectory(filename):
    """Load one trajectory from a CSV file written by this script."""
    data = np.atleast_1d(
        np.genfromtxt(filename, delimiter=",", names=True, dtype=float)
    )
    stem = filename.stem
    if stem == "pysmd_shadow":
        method = "SMD"
        conv_tol = None
        smd_tol = REL_RES_ERROR_TOL
    elif stem.startswith("pysmd_shadow_tol_"):
        method = "SMD"
        conv_tol = None
        smd_tol = float(stem.removeprefix("pysmd_shadow_tol_"))
    elif stem.startswith("pyscf_bomd_tol_"):
        method = "BOMD"
        conv_tol = float(stem.removeprefix("pyscf_bomd_tol_"))
        smd_tol = None
    else:
        raise ValueError(f"Unrecognized trajectory filename: {filename}")

    return {
        "method": method,
        "conv_tol": conv_tol,
        "smd_tol": smd_tol,
        "time_fs": data["time_fs"],
        "potential": data["potential_hartree"],
        "kinetic": data["kinetic_hartree"],
        "total": data["total_hartree"],
        "scf_cycles": data["scf_cycles"],
    }


def load_trajectories(output_dir, smd_file=None):
    """Load all BOMD and SMD trajectories in an output directory."""
    bomd_files = list(output_dir.glob("pyscf_bomd_tol_*.csv"))
    if not bomd_files:
        raise FileNotFoundError(f"No BOMD trajectory CSV files found in {output_dir}")

    trajectories = [read_trajectory(filename) for filename in bomd_files]
    trajectories.sort(key=lambda trajectory: trajectory["conv_tol"], reverse=True)

    if smd_file:
        shadow_file = output_dir / smd_file
    else:
        shadow_file = output_dir / (
            trajectory_name(
                {"method": "SMD", "smd_tol": DEFAULT_PLOT_SMD_TOL}
            )
            + ".csv"
        )
        if not shadow_file.is_file():
            shadow_file = output_dir / "pysmd_shadow.csv"
    if not shadow_file.is_file():
        raise FileNotFoundError(f"SMD trajectory CSV file not found: {shadow_file}")
    trajectories.append(read_trajectory(shadow_file))
    return trajectories


def format_tolerance(value):
    """Format a convergence tolerance for a math-text legend."""
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / 10**exponent
    if np.isclose(coefficient, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{coefficient:g}\mathbin{{\times}}10^{{{exponent}}}"


def plot_trajectories(output_dir, trajectories=None, smd_file=None):
    """Plot trajectories, loading them from CSV files when not supplied."""
    if trajectories is None:
        trajectories = load_trajectories(output_dir, smd_file=smd_file)

    bomd_trajectories = [
        trajectory for trajectory in trajectories if trajectory["method"] == "BOMD"
    ]
    shadow = next(
        trajectory for trajectory in trajectories if trajectory["method"] == "SMD"
    )
    num_pyscf = len(bomd_trajectories)
    colors = plt.cm.tab10(np.arange(num_pyscf))
    linestyles = ["-", "--", "-.", ":"]

    total_fig, total_axis = plt.subplots(figsize=(8, 4.8))
    difference_fig, difference_axis = plt.subplots(figsize=(8, 4.8))

    for index, (color, trajectory) in enumerate(
        zip(colors, bomd_trajectories)
    ):
        delta_millihartree = (
            trajectory["total"] - trajectory["total"][0]
        ) * 1e3
        tolerance = format_tolerance(trajectory["conv_tol"])
        label = rf"BOMD, $\epsilon_E={tolerance}$"
        plot_options = {
            "color": color,
            "linestyle": linestyles[index % len(linestyles)],
            "linewidth": 2,
            "zorder": 2,
        }
        total_axis.plot(
            trajectory["time_fs"],
            trajectory["total"],
            label=label,
            **plot_options,
        )
        difference_axis.plot(
            trajectory["time_fs"],
            delta_millihartree,
            label=label,
            **plot_options,
        )

    shadow_delta = (shadow["total"] - shadow["total"][0]) * 1e3
    shadow_label = rf"SMD, $\epsilon_K={format_tolerance(shadow['smd_tol'])}$"
    total_axis.plot(
        shadow["time_fs"],
        shadow["total"],
        color="black",
        linewidth=2.8,
        label=shadow_label,
        zorder=1,
    )
    difference_axis.plot(
        shadow["time_fs"],
        shadow_delta,
        color="black",
        linewidth=2.8,
        label=shadow_label,
        zorder=1,
    )

    total_axis.set_xlabel(r"time, $t$ [fs]")
    total_axis.set_ylabel(r"$E_{\mathrm{tot}}(t)$ [Ha]")
    total_axis.ticklabel_format(axis="y", style="plain", useOffset=False)

    difference_axis.axhline(
        0.0, color="0.75", linestyle="--", linewidth=0.8
    )
    difference_axis.set_xlabel(r"time, $t$ [fs]")
    difference_axis.set_ylabel(
        r"$E_{\mathrm{tot}}(t)-E_{\mathrm{tot}}(0)$ [mHa]"
    )

    for axis in (total_axis, difference_axis):
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1])

    total_fig.tight_layout()
    difference_fig.tight_layout()
    total_file = output_dir / "h2o_5cluster_rhf_total_energy.pdf"
    difference_file = output_dir / "h2o_5cluster_rhf_energy_difference.pdf"
    total_fig.savefig(total_file)
    difference_fig.savefig(difference_file)
    plt.close(total_fig)
    plt.close(difference_fig)
    return total_file, difference_file


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare RHF Born-Oppenheimer energy drift at several SCF "
            "thresholds with PySMD shadow MD for a five-water cluster."
        )
    )
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--timestep-fs", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument(
        "--conv-tols",
        type=float,
        nargs="+",
        default=[1e-5, 1e-6, 1e-7, 1e-8],
        metavar="TOL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("h2o_5cluster_rhf_energy_drift"),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate plots from trajectory CSV files without running MD",
    )
    parser.add_argument(
        "--smd-file",
        default=None,
        metavar="FILE",
        help=(
            "SMD CSV filename to plot; defaults to the 1e-2 SMD file "
            "(or the legacy pysmd_shadow.csv)"
        ),
    )
    parser.add_argument(
        "--smd-only",
        action="store_true",
        help="run only SMD, preserving all existing BOMD trajectory files",
    )
    parser.add_argument(
        "--smd-rel-res-error-tol",
        type=float,
        default=REL_RES_ERROR_TOL,
        metavar="TOL",
        help="PySMD relative residual threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2.")
    if args.timestep_fs <= 0:
        raise ValueError("--timestep-fs must be positive.")
    if not args.conv_tols or any(tol <= 0 for tol in args.conv_tols):
        raise ValueError("--conv-tols must contain positive values.")
    if args.smd_rel_res_error_tol <= 0:
        raise ValueError("--smd-rel-res-error-tol must be positive.")

    logging.getLogger("pysmd").setLevel(logging.WARNING)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        for plot_file in plot_trajectories(
            args.output_dir, smd_file=args.smd_file
        ):
            print(f"Wrote {plot_file}")
        return

    trajectories = []
    if not args.smd_only:
        for conv_tol in args.conv_tols:
            print(f"Running PySCF BOMD with conv_tol={conv_tol:.0e}")
            trajectories.append(
                run_pyscf_bomd(
                    basis=args.basis,
                    conv_tol=conv_tol,
                    timestep_fs=args.timestep_fs,
                    steps=args.steps,
                )
            )

    print("Running PySMD shadow MD")
    trajectories.append(
        run_pysmd(
            basis=args.basis,
            timestep_fs=args.timestep_fs,
            steps=args.steps,
            rel_res_error_tol=args.smd_rel_res_error_tol,
        )
    )

    for trajectory in trajectories:
        filename = write_trajectory(trajectory, args.output_dir)
        print(f"Wrote {filename}")

    if args.smd_only:
        trajectories = load_trajectories(
            args.output_dir,
            smd_file=trajectory_name(trajectories[-1]) + ".csv",
        )
    for plot_file in plot_trajectories(
        args.output_dir, trajectories
    ):
        print(f"Wrote {plot_file}")


if __name__ == "__main__":
    main()
