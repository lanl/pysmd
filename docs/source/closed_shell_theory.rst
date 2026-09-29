Closed-shell theory and implementation
======================================

The :mod:`pysmd.closed_shell` package implements finite-temperature,
closed-shell shadow Born--Oppenheimer molecular dynamics (SMD) using a
single-particle density matrix and a quantum-chemistry backend.  The theory
summarized here follows the associated manuscript, especially
Sections ``Theory`` and ``Finite-Temperature Electronic Structure Ansatz``
and Appendices ``Low-rank kernel action`` and ``Explicit RHF and RKS energies
and nuclear gradients``.

The theory summarized here is independent of the quantum-chemistry backend.

.. contents::
   :local:
   :depth: 2

Density-matrix conventions
--------------------------

Let :math:`\mathbf{D}` denote the density matrix formed from occupied spatial
orbitals, as in the manuscript.  PySMD uses spatial-orbital occupations in
the range :math:`[0,1]`: a fully occupied spatial orbital has occupation one,
and the electron count satisfies :math:`N_{\mathrm{elec}} = 2\sum_i f_i`.
These are the backend-independent conventions used throughout the closed-shell
implementation.  The principal density-like quantities in
:class:`pysmd.closed_shell.shadow_md.ShadowMD` are:

``dm_ao``
   The relaxed density matrix in the AO basis, :math:`\mathbf{D}`.

``dynvar_X``
   The extended electronic dynamical variable :math:`\mathbf{X}`.

``dm_prop``
   The propagated auxiliary density used to construct the Fock matrix,
   :math:`\mathbf{P} = \mathbf{X}\mathbf{S}^{-1}`.

``dm_lin``
   The linearized density used by the zeroth-level shadow energy and gradient,
   as defined by the corresponding shadow functional.

``S``
   The AO overlap matrix.  ``Sm1``, ``Sp12``, and ``Sm12`` are its inverse,
   positive square root, and inverse square root, respectively, subject to
   the backend's matrix representation.

The electronic residual is

.. math::

      \Delta = \mathbf{D}\mathbf{S}-\mathbf{X}.

Shadow Born--Oppenheimer molecular dynamics
--------------------------------------------

Conventional Born--Oppenheimer molecular dynamics minimizes the electronic
energy or free-energy functional at every nuclear configuration.  SMD instead
introduces an auxiliary density generated from the extended variable
:math:`\mathbf{X}`:

.. math::

   \mathbf{P}_t = \mathbf{X}_t\mathbf{S}_t^{-1}.

The shadow functional is minimized with respect to the electronic density
while its nonlinear two-electron dependence is evaluated around
:math:`\mathbf{P}`.  The resulting relaxed density is used to evaluate a
variational shadow potential and its nuclear gradient.  This construction
allows the electronic state to be updated by direct Fock diagonalization
rather than by a nonlinear SCF optimization at every MD step.

The main execution path is:

.. code-block:: text

   ShadowMD.kernel()
       -> verify_scf_convergence()
       -> initialize nuclear and electronic variables
       -> update_gradients()
       -> integrate one nuclear/electronic timestep
       -> update density, energy, gradient, and residual
       -> store SimulationData

The entry point :meth:`pysmd.closed_shell.shadow_md.ShadowMD.kernel` can reuse
converged backend SCF data or invoke
:class:`pysmd.closed_shell.newton_raphson.NewtonRaphson` when a suitable
initial state is unavailable.

Zeroth- and first-level surfaces
--------------------------------

At zeroth level, the auxiliary density is formed directly from the current
extended variable.  The Fock matrix is constructed from ``dm_prop`` and is
diagonalized in an orthogonal basis.  The resulting orbital energies and
fractional occupations define ``dm_ao``.  The code then forms ``dm_lin`` and
uses the corresponding shadow gradient.

When ``first_level_dm_update`` is enabled, the residual is acted on by an
approximate inverse-Jacobian kernel:

.. math::

   \mathbf{X}^{(1)}
       = \mathbf{X}^{(0)} - \mathcal{K}\Delta^{(0)},
   \qquad
   \mathbf{P}^{(1)} = \mathbf{X}^{(1)}\mathbf{S}^{-1}.

This update is implemented by
:meth:`pysmd.closed_shell.shadow_md.ShadowMD.update_first_level_dm`, which
calls :func:`pysmd.closed_shell.jacobian.pseudo_inverse_action` and rebuilds
the propagated, relaxed, and linearized density matrices around the updated
auxiliary density.

Finite-temperature occupations
------------------------------

The Fock matrix is transformed to the orthogonal basis and diagonalized.  At
nonzero electronic temperature, the occupation of spatial orbital :math:`i` is

.. math::

   f_i = \left[\exp\left(\beta(\varepsilon_i-\mu)\right)+1\right]^{-1},
   \qquad
   \beta = (k_B T)^{-1}.

The chemical potential is updated so that

.. math::

   \sum_i f_i = N_{\mathrm{elec}}/2.

:func:`pysmd.closed_shell.fermi_dirac.update_fractional_occ` performs this
update with a scalar Newton iteration.  It starts from the midpoint between
the highest occupied and lowest unoccupied orbital energies and stops when the
occupation-number error is below ``frac_occ_tol``.  The routine also guards
against a vanishing derivative and stops after a maximum of 100 iterations.

The entropy contribution is included in the free energy when the temperature
is nonzero.  At zero temperature the occupations become integer occupations
and the entropy contribution vanishes.

Low-rank inverse-Jacobian action
--------------------------------

The electronic equation of motion and the first-level update require only the
action of the inverse Jacobian on a residual, not an explicitly constructed
four-index Jacobian:

.. math::

   \mathcal{K}\Delta = \mathcal{J}^{-1}\Delta.

:func:`pysmd.closed_shell.jacobian.pseudo_inverse_action` approximates this
action in a low-rank subspace.  For each rank, it:

#. orthonormalizes a residual-derived matrix to form a basis matrix
   :math:`\mathbf{V}_m`;
#. evaluates the canonical density response to that perturbation;
#. constructs the corresponding residual-response matrix
   :math:`\mathbf{W}_m`;
#. solves the small overlap system in the low-rank subspace; and
#. estimates the relative residual of the reconstructed kernel action.

The canonical response is evaluated by
:func:`pysmd.closed_shell.jacobian.recursive_canonical_dm_pt_response`, which
uses the recursive Fermi-operator expansion described in the manuscript
appendix.  The response includes a chemical-potential derivative so that the
electron number remains fixed.

The approximation stops when ``rel_res_error_tol`` is reached or when
``max_rank`` is exhausted.  If the maximum rank is reached first, the code
returns the available rank-limited approximation and emits a warning rather
than terminating the trajectory.

Equation-of-motion integration
------------------------------

The nuclear coordinates and velocities can be propagated with
:func:`pysmd.closed_shell.eom_integrators.integrate_verlet_timestep`.  Its
nuclear updates are the leapfrog velocity-Verlet updates.  The electronic
variable uses a modified Verlet recurrence containing the inverse-Jacobian
kernel action and a weak dissipative history term:

.. math::

   \mathbf{X}(t+\Delta t)
   = 2\mathbf{X}(t)-\mathbf{X}(t-\Delta t)
     + \Delta t^2\omega^2\mathcal{K}\Delta(t)
     + \alpha\sum_{k=0}^{K}c_k\mathbf{X}(t-k\Delta t).

The coefficient-split implementation,
:func:`pysmd.closed_shell.eom_integrators.integrate_coeff_split_timestep`,
uses :math:`L` intermediate position and velocity updates.  The coefficient
tables and dissipative parameters are selected by
:class:`pysmd.closed_shell.eom_params.EOMParams` from
``EOM_INTEGRATION_COEFFS``, ``EOM_DISS_COEFFS``, and
``EOM_DISS_KAPPA_ALPHA_PARAMS``.

``verlet`` supports ``L=2``.  ``optimal`` supports ``L`` equal to 2, 3, 4,
or 6.  Dissipative history orders from ``K=3`` through ``K=9`` are available.
The history matrices are stored in ``X_history``; higher-order or
coefficient-split propagation requires storage for each intermediate stage.

Energy and gradient decomposition
---------------------------------

:class:`pysmd.closed_shell.shadow_md.ShadowMD` stores separate kinetic,
nuclear, electronic, exchange-correlation, entropic, free, and total energy
components.  The electronic free energy is assembled from the one-electron,
linearized two-electron, exchange, and exchange-correlation contributions.
The nuclear gradient combines one-electron, two-electron, exchange-correlation,
Pulay, and nuclear-repulsion terms through the active QM interface.

The distinction between ``dm_prop`` and ``dm_lin`` is essential for a shadow
surface.  They are equal only in the self-consistent limit; the backend
therefore receives separate propagated and linearized density arguments when
constructing the two-electron gradient.

Simulation data
---------------

:class:`pysmd.closed_shell.simulation_data.SimulationData` stores coordinates,
energy components, residual norms, and optional gradient and SCF-reference
errors for each timestep.  The stored arrays use the simulation timestep and
number of atoms supplied at construction.  Data can be written to HDF5 with
:meth:`pysmd.closed_shell.simulation_data.SimulationData.save_data_to_hdf5`.

For the public API, see the autogenerated reference pages under
:doc:`api/pysmd.closed_shell`.
