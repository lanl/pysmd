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

"""Shared constants and logging infrastructure used throughout PySMD.

The :mod:`pysmd.common.constants` module contains physical constants and unit
conversion factors. The :mod:`pysmd.common.logger` module provides the
package-wide logging configuration and timing helpers.
"""

from pysmd.common import constants
from pysmd.common import logger


__all__ = [
    "constants",
    "logger",
]
