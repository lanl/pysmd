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

"""Logging utilities for the PySMD package.

We define two additional levels for the logger:
    - HEADER: Defined at level = 100, such that the header always prints
    - TIMING: Defined at level = 15, such that performance can be timed without
              use of the the debug mode.

For the two levels above, we define two functions, `pysmd_header` and `timer`,
which are added to the logging object before the package instantiates an
instance of the Logger class.

Note:
    The implementation of adding the `timer` functionality to the Logger object
    is largely-based on a similar procedure performed in the PySCF open-source
    quantum chemistry package.

The function `filter_output_single_level` serves as a "filter" function
for the various handlers in the Logger performing output to various streams.

Call :func:`configure` once from package initialization to configure the
top-level ``pysmd`` logger. Package modules should request hierarchical loggers
with ``getLogger(__name__)``.
"""

import os, time
import logging, logging.config
from collections.abc import Callable
from typing import Any


# Modify logging module before creating object
logging.Logger.cpu_time = time.process_time
logging.Logger.wall_time = time.perf_counter


# Additional levels
HEADER_LEVEL = 100
logging.addLevelName(HEADER_LEVEL, "HEADER")
TIMING_LEVEL = 15
logging.addLevelName(TIMING_LEVEL, "TIMING")


def pysmd_header(self, *args, **kwargs) -> None:
    """Print the PySMD header when the package is imported."""
    msg = r"""
###############################################################################
##                                                                           ##
##                       ____         _____  __  ___ ____                    ##
##                      / __ \ __  __/ ___/ /  |/  // __ \                   ##
##                     / /_/ // / / /\__ \ / /|_/ // / / /                   ##
##                    / ____// /_/ /___/ // /  / // /_/ /                    ##
##                   /_/     \__, //____//_/  /_//_____/                     ##
##                          /____/                                           ##
##                                version 1.0                                ##
##             -------------------------------------------------             ##
##              Authors:                                                     ##
##              - Ilia M. Mazin               <imazin@lanl.gov>              ##
##              - Yu Zhang                       <zhy@lanl.gov>              ##
##              - Anders M.N. Niklasson          <amn@lanl.gov>              ##
##                                                                           ##
###############################################################################\n"""
    if self.isEnabledFor(HEADER_LEVEL):
        self._log(HEADER_LEVEL, msg, args, **kwargs)
    return


def timer(self, msg, cpu0, wall0, *args, **kwargs) -> float | tuple[float, float]:
    """Compute current CPU and walltimes."""
    if cpu0 is None:
        cpu0 = self.t_cpu

    if wall0:
        timings = self.t_cpu, self.t_wall = self.cpu_time(), self.wall_time()
        info = f">> {msg:s}:\n" + \
                f" --  CPU time: {(self.t_cpu-cpu0):9.2f} sec\n" + \
                f" -- wall time: {(self.t_wall-wall0):9.2f} sec"

    else:
        timings = self.t_cpu = self.cpu_time()
        info = f">> {msg:s}:\n" + \
                f" --  CPU time: {(self.t_cpu-cpu0):9.2f} sec"

    if self.isEnabledFor(TIMING_LEVEL):
        self._log(TIMING_LEVEL, info, args, **kwargs)

    return timings


# Modify Logger class with header output and timing methods
logging.Logger.print_header = pysmd_header
logging.Logger.timer = timer


def filter_output_single_level(level: str) -> Callable[[logging.LogRecord], bool]:
    """Create logger filter to filter messages of provided `level`.

    Args:
        level: A string reprsenting the desired log level,
               (e.g. "INFO", "ERROR", etc).

    Returns:
        A callable filter function that returns True for log records
        matching the specified `level`.
    """
    level_no: int | None = logging._nameToLevel.get(level)
    def filter(record: logging.LogRecord) -> bool:
        return record.levelno == level_no
    return filter


def filter_logfile(level: str) -> Callable[[logging.LogRecord], bool]:
    """Create logger filter to exclude package headers from debug logs."""
    level_no: int | None = logging._nameToLevel.get(level)

    def filter(record: logging.LogRecord) -> bool:
        return record.levelno < level_no

    return filter


def _build_log_config(debug: bool = False) -> dict:
    """Build package logger configuration."""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "info_format": {
                "format": "{message:s}",
                "style": "{"
            },

            "warning_format": {
                "format": "WARNING >> {message:s}",
                "style": "{"
            },

            "error_format": {
                "format": "ERROR! >> {message:s}",
                "style": "{"
            },

            "critical_format": {
                "format": "CRITICAL!!! >> {message:s}",
                "style": "{"
            }
        },

        "filters": {
            "header_filter": {
                "()": filter_output_single_level,
                "level": "HEADER",
            },

            "timing_filter": {
                "()": filter_output_single_level,
                "level": "TIMING",
            },
            "info_filter": {
                "()": filter_output_single_level,
                "level": "INFO",
            },

            "warning_filter": {
                "()": filter_output_single_level,
                "level": "WARNING",
            },

            "error_filter": {
                "()": filter_output_single_level,
                "level": "ERROR",
            },

            "critical_filter": {
                "()": filter_output_single_level,
                "level": "CRITICAL",
            },
        },

        "handlers": {
            "stdout_header": {
                "class": "logging.StreamHandler",
                "level": "HEADER",
                "stream": "ext://sys.stdout",
                "filters": ["header_filter"]
            },

            "stdout_timing": {
                "class": "logging.StreamHandler",
                "level": "TIMING",
                "stream": "ext://sys.stdout",
                "filters": ["timing_filter"]
            },

            "stdout_info": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "info_format",
                "stream": "ext://sys.stdout",
                "filters": ["info_filter"]
            },

            "stdout_warning": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "warning_format",
                "stream": "ext://sys.stdout",
                "filters": ["warning_filter"]
            },

            "stderr_error": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "error_format",
                "stream": "ext://sys.stderr",
                "filters": ["error_filter"]
            },

            "stderr_critical": {
                "class": "logging.StreamHandler",
                "level": "CRITICAL",
                "formatter": "critical_format",
                "stream": "ext://sys.stderr",
                "filters": ["critical_filter"]
            }
        },

        "loggers": {
            "pysmd": {
                "level": "DEBUG" if debug else "INFO",
                "handlers": ["stdout_header",
                             "stdout_timing",
                             "stdout_info",
                             "stdout_warning",
                             "stderr_error",
                             "stderr_critical"
                            ],
                "propagate": False
            }
        }
    }

    if debug:
        log_file_path = "log/pysmd_debug.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        log_config["filters"]["debug_filter"] = {
            "()": filter_logfile,
            "level": "HEADER",
        }

        log_config["formatters"]["debug_format"] = {
            "format": "[ {asctime:^s} ] {levelname:>8s} ||" + \
                      " filename: {filename:<10.10s} |" + \
                      " line: {lineno:3d} || \"{message:s}\"",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "{"
        }

        log_config["handlers"]["log_file_debug"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "debug_format",
            "filename": log_file_path,
            "mode": "a",
            "maxBytes": 5000,
            "backupCount": 3,
            "filters": ["debug_filter"]
        }

        log_config["loggers"]["pysmd"]["handlers"].append("log_file_debug")

    return log_config


getLogger = logging.getLogger # alias


def configure(debug: bool = False, print_header: bool = True) -> None:
    """Configure the top-level PySMD logger."""
    log = getLogger("pysmd")
    already_configured = getattr(log, "_pysmd_configured", False)

    if not already_configured:
        logging.config.dictConfig(_build_log_config(debug=debug))
        log = getLogger("pysmd")
        log.t_cpu = log.cpu_time()
        log.t_wall = log.wall_time()
        log._pysmd_configured = True

    if print_header and not getattr(log, "_pysmd_header_printed", False):
        log.print_header()
        log._pysmd_header_printed = True

    return
