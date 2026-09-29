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

"""Adapters connecting PySMD to the PySCF quantum-chemistry package.

The adapter classes retain PySCF's molecule and mean-field objects and call
PySCF directly for standard one- and two-electron quantities. PySMD-specific
operations, such as separated or linearized gradient components and array
backend conversion, are implemented in the adapter classes rather than by
modifying PySCF itself.

The module also provides a factory function that detects the mean-field type
and returns the appropriate interface class.

Usage
-----
Call the module-level factory function with a PySCF mean-field object:

    >>> from pysmd.interface.pyscf import PyscfDriver
    >>> qm_interface = PyscfDriver(pyscf_mf=mf)  # Automatically detects RHF, RKS, etc.

Or import specific classes directly:

    >>> from pysmd.interface.pyscf import RHF, RKS
    >>> qm_interface = RHF(pyscf_mf=mf)
    >>> qm_interface = RKS(pyscf_mf=mf)
"""

from typing import Any

from pyscf import dft, gto, scf
from pysmd.common import logger

# Import interface classes from new modular implementations
from pysmd.interface.pyscf.gpu import GPU4RHF, GPU4RKS, _load_gpu4pyscf
from pysmd.interface.pyscf.rhf import RHF
from pysmd.interface.pyscf.rks import RKS

log = logger.getLogger(__name__)


def PyscfDriver(
    pyscf_mf: object | None = None,
    pyscf_mol: gto.MoleBase | None = None,
    backend: str = "cpu",
    **kwargs: Any,
) -> RHF | RKS | GPU4RHF | GPU4RKS:
    """Create the appropriate PySMD adapter for a PySCF mean-field object.

    The dispatch pattern follows PySCF's mean-field factory style, but the
    returned object is a PySMD adapter. CPU ``RHF`` and ``RKS`` adapters use
    direct PySCF calls for standard quantities and add the PySMD-specific
    operations required by the shadow-MD driver. GPU selections return the
    corresponding GPU4PySCF adapters.

    Both pyscf_mf and pyscf_mol are optional (matching the interface class
    signatures), but at least one must be provided. If pyscf_mf is provided,
    the factory automatically detects its type and returns the appropriate
    interface. If only pyscf_mol is provided, additional kwargs (like 'xc')
    can be used to specify the method type.

    Parameters
    ----------
    pyscf_mf : object, optional
        PySCF or GPU4PySCF mean-field object. RHF and RKS are supported.
    pyscf_mol : gto.MoleBase, optional
        PySCF molecule object. Can be provided alone or with pyscf_mf.
        If pyscf_mf is also provided, it takes precedence.
    backend : {"cpu", "gpu", "auto"}, default="cpu"
        Electronic backend. Auto detects pyscf_mf and defaults molecules to CPU.
    **kwargs
        Additional keyword arguments to pass to the specific interface class.
        - 'xc' (str): Exchange-correlation functional for DFT methods (RKS, UKS)
        - Other method-specific parameters as needed

    Returns
    -------
    RHF | RKS | GPU4RHF | GPU4RKS
        An instance of the appropriate PySCF interface class. The specific
        type depends on the mean-field object provided:

        - scf.hf.RHF → RHF
        - dft.rks.RKS → RKS
        - gpu4pyscf.scf.RHF → GPU4RHF
        - gpu4pyscf.dft.RKS → GPU4RKS

        All returned objects inherit from QMSoftware and provide the same
        core interface methods.

    Raises
    ------
    TypeError
        If the mean-field object type is not recognized or supported
    ValueError
        If both pyscf_mf and pyscf_mol are None

    Examples
    --------
    >>> from pyscf import gto, scf, dft
    >>> from pysmd.interface.pyscf import PyscfDriver
    >>>
    >>> # Create a molecule
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz')
    >>>
    >>> # Example 1: With RHF mean-field object (automatic detection)
    >>> mf = scf.RHF(mol)
    >>> mf.kernel()
    >>> qm_interface = PyscfDriver(pyscf_mf=mf)
    >>> # Returns an RHF interface object

    >>> # Example 2: With DFT mean-field object
    >>> mf_dft = dft.RKS(mol)
    >>> mf_dft.xc = 'b3lyp'
    >>> mf_dft.kernel()
    >>> qm_interface = PyscfDriver(pyscf_mf=mf_dft)
    >>> # Returns an RKS interface object

    >>> # Example 3: With only molecule (specify method via kwargs)
    >>> qm_interface = PyscfDriver(pyscf_mol=mol, xc='pbe')
    >>> # Returns an RKS interface object

    >>> # Example 4: With only molecule (defaults to RHF)
    >>> qm_interface = PyscfDriver(pyscf_mol=mol)
    >>> # Returns an RHF interface object

    Notes
    -----
    - If pyscf_mf is provided, it takes precedence over pyscf_mol for
      determining the interface type
    - The function uses isinstance checks to determine the mean-field type
    - Check order: RKS → UKS → RHF → UHF (DFT first since they inherit from HF)
    - Currently supports: RHF, RKS (UHF and UKS support can be added)
    - Both parameters are optional to match interface class signatures, but
      at least one must be provided
    """

    if not isinstance(backend, str):
        raise TypeError("backend must be one of cpu, gpu, or auto.")
    backend = backend.lower()
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError(
            f"Unsupported electronic backend {backend!r}; expected cpu, "
            "gpu, or auto."
        )

    if pyscf_mf is not None:
        object_backend = (
            "gpu"
            if type(pyscf_mf).__module__.split(".", 1)[0] == "gpu4pyscf"
            else "cpu"
        )
        selected_backend = object_backend if backend == "auto" else backend
        if selected_backend != object_backend:
            raise TypeError(
                f"backend={backend!r} expects {selected_backend.upper()} "
                f"mean-field objects, but received {object_backend.upper()} "
                f"object {type(pyscf_mf).__module__}.{type(pyscf_mf).__name__}."
            )
        backend = selected_backend

        if backend == "gpu":
            gpu_scf, gpu_dft = _load_gpu4pyscf()
            if isinstance(pyscf_mf, gpu_dft.rks.RKS):
                return GPU4RKS(pyscf_mf=pyscf_mf, **kwargs)
            if isinstance(pyscf_mf, gpu_dft.uks.UKS):
                raise NotImplementedError("GPU4PySCF UKS is not implemented.")
            if isinstance(pyscf_mf, gpu_scf.rohf.ROHF):
                raise NotImplementedError("GPU4PySCF ROHF is not implemented.")
            if isinstance(pyscf_mf, gpu_scf.hf.RHF):
                return GPU4RHF(pyscf_mf=pyscf_mf, **kwargs)
            if isinstance(pyscf_mf, gpu_scf.uhf.UHF):
                raise NotImplementedError("GPU4PySCF UHF is not implemented.")
            raise TypeError(
                f"Unsupported GPU4PySCF mean-field type: "
                f"{type(pyscf_mf).__name__}. Currently supported: RHF, RKS."
            )

    # Validate that at least one argument is provided
    # This matches the interface class behavior (both optional, but need at least one)
    if pyscf_mf is None and pyscf_mol is None:
        raise ValueError(
            "At least one of pyscf_mf or pyscf_mol must be provided. "
            "This matches the interface class requirement where both parameters are "
            "optional but at least one must be specified."
        )

    # If mean-field object is provided, determine its type
    if pyscf_mf is not None:
        # Check for DFT methods first (they inherit from HF classes)
        if isinstance(pyscf_mf, dft.rks.RKS):
            log.info(">> PySCF factory detected RKS mean-field object")
            return RKS(pyscf_mf=pyscf_mf, pyscf_mol=None, **kwargs)

        elif isinstance(pyscf_mf, dft.uks.UKS):
            log.info(">> PySCF factory detected UKS mean-field object")
            raise NotImplementedError(
                "UKS interface not yet implemented. "
                "Please use RKS for restricted systems."
            )

        elif isinstance(pyscf_mf, scf.rohf.ROHF):
            raise NotImplementedError(
                "ROHF interface not yet implemented. Use RHF for closed-shell systems."
            )

        # Check for HF methods
        elif isinstance(pyscf_mf, scf.hf.RHF):
            log.info(">> PySCF factory detected RHF mean-field object")
            return RHF(pyscf_mf=pyscf_mf, pyscf_mol=None, **kwargs)

        elif isinstance(pyscf_mf, scf.uhf.UHF):
            log.info(">> PySCF factory detected UHF mean-field object")
            raise NotImplementedError(
                "UHF interface not yet implemented. "
                "Please use RHF for restricted systems."
            )

        else:
            raise TypeError(
                f"Unsupported mean-field type: {type(pyscf_mf).__name__}. "
                f"Currently supported types: RHF, RKS"
            )

    # If only molecule is provided, infer method type from kwargs
    else:
        if backend == "auto":
            backend = "cpu"
        if backend == "gpu":
            if "xc" in kwargs:
                return GPU4RKS(pyscf_mol=pyscf_mol, **kwargs)
            return GPU4RHF(pyscf_mol=pyscf_mol, **kwargs)

        # Try to infer from kwargs or default to RHF
        if 'xc' in kwargs:
            log.info(">> PySCF factory: xc specified, creating RKS interface")
            return RKS(pyscf_mol=pyscf_mol, pyscf_mf=None, **kwargs)
        else:
            log.info(">> PySCF factory: defaulting to RHF interface")
            return RHF(pyscf_mol=pyscf_mol, pyscf_mf=None, **kwargs)

__all__ = [
    "PyscfDriver",
    "RHF",
    "RKS",
    "GPU4RHF",
    "GPU4RKS",
]
