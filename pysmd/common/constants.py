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

"""Fundamental physical constants and atomic-unit conversion factors.

Values obtained from the 2022 CODATA adjustment:
https://physics.nist.gov/cuu/Constants/Table/allascii.txt

The constants are module-level floating-point values. Names ending in a
conversion suffix describe the destination unit, for example ``BOHR2ANG``
converts Bohr radii to Angstroms and ``AU2FS`` converts atomic time to
femtoseconds.
"""

from math import pi as PI


# atoms/mole
AVOGADRO = 6.02214076e23
# Joule*seconds
PLANCK = 6.62607015e-34
# Joule*seconds
HBAR = PLANCK/(2 * PI)

# meters
BOHR = 5.29177210544e-11
# kilograms
E_MASS = 9.1093837139e-31
# Coulombs
E_CHARGE = 1.602176634e-19
# seconds
ATOMIC_TIME = 2.4188843265864e-17
# kilograms
ATOMIC_MASS = 1.0e-3/AVOGADRO
# Bohr radius in Angstroms
BOHR2ANG = BOHR * 1.0e10
# Hartree energy in electronvolts
HARTREE2EV = 27.211386245981
# Mass in units of electron mass (dimensionless)
AMU2AU = ATOMIC_MASS/E_MASS
# atomic time in femtoseconds
AU2FS = ATOMIC_TIME * 1.0e15
# Hartree energy in Joules
HARTREE2J = HBAR**2/(E_MASS*BOHR**2)

# Joules/Kelvin
BOLTZMANN = 1.380649e-23
# Boltzmann constant in units of Hartree/Kelvin
KB = BOLTZMANN/HARTREE2J
