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

"""Adapters between PySMD and quantum-chemistry software packages.

The :mod:`pysmd.interface.qm_software` module defines the backend contract.
Concrete adapters live in subpackages named after the supported software;
currently :mod:`pysmd.interface.pyscf` provides the PySCF implementation.

The PySCF submodule provides a factory function for automatic interface detection:
    >>> from pysmd.interface.pyscf import PyscfDriver
    >>> qm_interface = PyscfDriver(pyscf_mf=mf)  # Auto-detects RHF, RKS, UHF, UKS

Or import specific classes:
    >>> from pysmd.interface.pyscf import RHF, RKS, UHF, UKS

The unrestricted UHF and UKS adapters are used by
:mod:`pysmd.open_shell` and implement the optional spin-resolved hooks of
:class:`pysmd.interface.qm_software.QMSoftware`.
"""

from pysmd.interface import qm_software
from pysmd.interface import pyscf

__all__ = [
    "qm_software",
    "pyscf",
]
