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

"""Top-level package for first-principles shadow molecular dynamics.

Importing :mod:`pysmd` initializes the process-wide array backend to NumPy and
configures the package logger. Optional backends can be selected later with
``pysmd.lib.linalg_helper.set_linalg_backend`` before calculation objects are
created.

Available modules:
    - `interface`: Functions for accessing package features.
    - `common`: Shared infrastructure for logging and constants.
    - `closed_shell`: Classes, kernels, and EOM parameters for molecular
                      dynamics (MD) and self-consistent field (SCF)
                      procedures for closed-shell systems.

The variable ``DEBUG_MODE`` controls the logging behavior of the package.
When it is ``True``, debug output is written to ``log/pysmd_debug.log`` in the
current working directory.
"""

DEBUG_MODE = False
#DEBUG_MODE = True # Enable to save DEBUG level output to .log file

from pysmd.lib import linalg_helper


linalg_helper.set_linalg_backend("numpy")

from pysmd.common import logger


logger.configure(debug=DEBUG_MODE, print_header=True)

from pysmd import (
    interface,
    common,
    closed_shell,
    lib,
)


__version__ = "1.0.0"
__all__ = [
    "interface",
    "common",
    "closed_shell",
    "lib",
]
