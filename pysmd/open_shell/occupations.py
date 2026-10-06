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

"""Maximum-overlap occupations for DeltaSCF and moving-basis dynamics."""

import numpy as np


class InitialMaximumOverlap:
    """Track the occupied subspaces of an initial alpha/beta determinant.

    The reference is stored in the orthonormal AO basis. Scores are the
    squared overlaps of current orbitals with the initially occupied space,
    summed over occupied reference orbitals. Orbital signs and rotations
    within that space do not affect the selection. The reference never
    changes during SCF iterations or rejected Newton line-search trials.
    """

    def __init__(self, dm0, overlap_sqrt, spin_counts):
        dm0 = np.asarray(dm0)
        if np.iscomplexobj(dm0):
            raise ValueError("IMOM currently requires real density matrices.")
        nmo = overlap_sqrt.shape[0]
        if dm0.shape != (2, nmo, nmo) or not np.isfinite(dm0).all():
            raise ValueError("IMOM requires a finite initial density with shape (2, nmo, nmo).")
        orthogonal = overlap_sqrt @ dm0 @ overlap_sqrt
        if not np.allclose(orthogonal, orthogonal.swapaxes(-1, -2), atol=1e-8, rtol=0):
            raise ValueError("IMOM initial density must be symmetric.")
        values, vectors = np.linalg.eigh(orthogonal)
        rounded = np.rint(values)
        if (not np.allclose(values, rounded, atol=1e-7, rtol=0)
                or not np.isin(rounded, (0, 1)).all()):
            raise ValueError("IMOM needs an idempotent determinant: build dm0 from orbitals "
                             "with 0/1 occupations, not a fractional or mixed density.")
        if not np.array_equal(rounded.sum(axis=1), spin_counts):
            raise ValueError("The initial determinant must preserve the reference's alpha/beta populations.")
        self.spin_counts = tuple(spin_counts)
        self.projector = (vectors * rounded[:, None, :]) @ vectors.swapaxes(-1, -2)
        self.projector.flags.writeable = False

    def at_geometry(self, cross_overlap, reference_inverse_sqrt, current_inverse_sqrt):
        """Represent the same physical reference in a new orthonormal AO basis.

        The resulting weights give actual cross-geometry orbital overlaps.
        They are not renormalized: projection into a different finite basis
        can reduce an orbital's norm. Always project from the saved reference,
        not from an already projected set of weights.
        """
        transform = current_inverse_sqrt @ cross_overlap @ reference_inverse_sqrt
        weights = transform @ self.projector @ transform.T
        if not np.isfinite(weights).all():
            raise ValueError("Nonfinite cross-geometry occupation overlaps.")
        for spin, count in enumerate(self.spin_counts):
            if count and np.linalg.eigvalsh(weights[spin])[-count] < 1e-10:
                raise RuntimeError("The occupied reference lost rank in the new AO basis; "
                                   "reduce the MD timestep or choose a closer reference.")
        reference = object.__new__(type(self))
        reference.spin_counts = self.spin_counts
        reference.projector = weights
        reference.projector.flags.writeable = False
        return reference

    def occupations(self, mo_coeff):
        """Select occupations for orbitals represented in the orthonormal basis."""
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.shape != self.projector.shape or not np.isfinite(mo_coeff).all():
            raise ValueError("IMOM orbital coefficients must have shape (2, nmo, nmo).")
        scores = np.diagonal(mo_coeff.swapaxes(-1, -2) @ self.projector @ mo_coeff,
                             axis1=-2, axis2=-1).copy()
        occ = np.zeros_like(scores)
        for spin, count in enumerate(self.spin_counts):
            order = np.argsort(-scores[spin], kind="stable")
            if 0 < count < len(order):
                gap = scores[spin, order[count - 1]] - scores[spin, order[count]]
                if gap < 1e-10:
                    raise RuntimeError("IMOM overlaps are tied at the occupied/virtual boundary; "
                                       "the initial determinant does not uniquely identify this state.")
            occ[spin, order[:count]] = 1.
        return occ
