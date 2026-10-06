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

"""Fermi occupations for two independent spin channels."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from pysmd.common import constants


def update_fractional_occ(method, mo_energy=None):
    """Return occupations (2, nmo) and chemical potentials (2,).

    Each spin orbital holds at most one electron. Alpha and beta populations
    are fixed even at finite temperature. Empty/full channels are handled
    explicitly, and temp=0 uses Aufbau occupations without entropy.
    """
    energies = np.asarray(method.mo_energy if mo_energy is None else mo_energy)
    if energies.ndim != 2 or energies.shape[0] != 2 or not np.isfinite(energies).all():
        raise ValueError("Orbital energies must be a finite (2, nmo) array.")
    if not np.isfinite(method.temp) or method.temp < 0:
        raise ValueError("Electronic temperature must be finite and nonnegative.")
    method.beta = np.inf if method.temp == 0 else 1 / (constants.KB * method.temp)
    occ = np.zeros_like(energies)
    mu = np.zeros(2)
    for spin, count in enumerate(method.n_occ):
        levels = energies[spin]
        if count < 0 or count > len(levels) or int(count) != count:
            raise ValueError("Spin population must be an integer between zero and nmo.")
        count = int(count)
        if count == 0:
            mu[spin] = -np.inf
        elif count == len(levels):
            occ[spin] = 1
            mu[spin] = np.inf
        elif method.temp == 0:
            order = np.argsort(levels)
            occ[spin, order[:count]] = 1
            mu[spin] = (levels[order[count - 1]] + levels[order[count]]) * 0.5
        else:
            width = max(1., 50 / method.beta)
            mu[spin] = brentq(
                lambda value: expit(method.beta * (value - levels)).sum() - count,
                levels.min() - width, levels.max() + width, xtol=1e-14,
            )
            occ[spin] = expit(method.beta * (mu[spin] - levels))
            if abs(occ[spin].sum() - count) > method.frac_occ_tol:
                raise RuntimeError("Unable to converge the spin population.")
    return occ, mu
