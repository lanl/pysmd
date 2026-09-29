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

"""Provide a small, backend-independent array and linear-algebra API.

PySMD uses this module instead of importing NumPy operations throughout the
calculation code. The active backend is process-wide and can be ``numpy``,
``torch`` (CPU, double precision), or ``cupy`` (GPU, double precision for
floating-point and complex arrays). Backend selection does not alter the
public PySMD calculation interfaces, but it must occur before constructing
objects that retain backend arrays.
"""

from typing import Any

import numpy
from scipy.special import expit as scipy_expit


class _NumpyArrayWithAttributes(numpy.ndarray):
    """NumPy array capable of retaining metadata across backend conversion."""


def _scalar_if_needed(value):
    if getattr(value, "ndim", None) == 0:
        return value.item()
    return value


class _NumpyBackend:
    """NumPy implementation of the PySMD array API."""

    linalg = numpy.linalg
    nan = numpy.nan

    def __getattr__(self, name):
        if name in {"dot", "array2string", "integer"}:
            raise AttributeError(f"The linear-algebra backend has no attribute {name!r}")
        return getattr(numpy, name)

    def array(self, value):
        return numpy.array(value)

    def asarray(self, value):
        return value if isinstance(value, numpy.ndarray) else numpy.asarray(value)

    def to_numpy(self, value):
        return value if isinstance(value, numpy.ndarray) else numpy.asarray(value)

    def copy(self, value):
        return numpy.copy(value)

    def expit(self, value):
        return scipy_expit(value)

    def matmul(self, left, right):
        return numpy.matmul(left, right)


class _TorchLinalg:
    """Torch implementations of the linalg operations used by PySMD."""

    def __init__(self, backend):
        self.backend = backend

    def eigh(self, value):
        return self.backend.torch.linalg.eigh(self.backend.asarray(value))

    def inv(self, value):
        return self.backend.torch.linalg.inv(self.backend.asarray(value))

    def norm(self, value):
        result = self.backend.torch.linalg.norm(self.backend.asarray(value))
        return _scalar_if_needed(result)


class _TorchBackend:
    """CPU double-precision Torch implementation of the PySMD array API."""

    nan = float("nan")

    def __init__(self, torch):
        self.torch = torch
        self.linalg = _TorchLinalg(self)

    def asarray(self, value):
        attributes = getattr(value, "__dict__", None)
        if self.torch.is_tensor(value):
            result = value.to(device="cpu")
        else:
            try:
                result = self.torch.as_tensor(value, device="cpu")
            except (TypeError, ValueError):
                result = self.torch.as_tensor(
                    numpy.ascontiguousarray(value), device="cpu"
                )

        if result.is_floating_point():
            result = result.to(dtype=self.torch.float64)
        elif result.is_complex():
            result = result.to(dtype=self.torch.complex128)
        if attributes:
            result.__dict__.update(attributes)
        return result

    def array(self, value):
        return self.asarray(value).clone()

    def to_numpy(self, value):
        if not self.torch.is_tensor(value):
            return value if isinstance(value, numpy.ndarray) else numpy.asarray(value)
        result = value.detach().cpu().resolve_conj().resolve_neg().numpy()
        attributes = getattr(value, "__dict__", None)
        if attributes:
            result = result.view(_NumpyArrayWithAttributes)
            result.__dict__.update(attributes)
        return result

    def copy(self, value):
        return self.asarray(value).clone()

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.torch.float64, device="cpu")

    def zeros_like(self, value):
        return self.torch.zeros_like(self.asarray(value), device="cpu")

    def diag(self, value):
        return self.torch.diag(self.asarray(value))

    def stack(self, values):
        return self.torch.stack(tuple(self.asarray(value) for value in values))

    def hstack(self, values):
        return self.torch.hstack(tuple(self.asarray(value) for value in values))

    def vstack(self, values):
        return self.torch.vstack(tuple(self.asarray(value) for value in values))

    def matmul(self, left, right):
        return self.torch.matmul(self.asarray(left), self.asarray(right))

    def einsum(self, equation, *operands, **kwargs):
        kwargs.pop("optimize", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported einsum keyword arguments: {unexpected}")
        result = self.torch.einsum(
            equation, *(self.asarray(operand) for operand in operands)
        )
        return _scalar_if_needed(result)

    def sum(self, value, axis=None):
        result = self.torch.sum(self.asarray(value), dim=axis)
        return _scalar_if_needed(result)

    def trace(self, value):
        return _scalar_if_needed(self.torch.trace(self.asarray(value)))

    def abs(self, value):
        return _scalar_if_needed(self.torch.abs(self.asarray(value)))

    def sqrt(self, value):
        return _scalar_if_needed(self.torch.sqrt(self.asarray(value)))

    def log(self, value):
        return _scalar_if_needed(self.torch.log(self.asarray(value)))

    def isfinite(self, value):
        return _scalar_if_needed(self.torch.isfinite(self.asarray(value)))

    def expit(self, value):
        return _scalar_if_needed(self.torch.sigmoid(self.asarray(value)))


class _CupyLinalg:
    """CuPy implementations of the linalg operations used by PySMD."""

    def __init__(self, backend):
        self.backend = backend

    def eigh(self, value):
        return self.backend.cupy.linalg.eigh(self.backend.asarray(value))

    def inv(self, value):
        return self.backend.cupy.linalg.inv(self.backend.asarray(value))

    def norm(self, value):
        result = self.backend.cupy.linalg.norm(self.backend.asarray(value))
        return _scalar_if_needed(result)


class _CupyBackend:
    """Double-precision CuPy implementation of the PySMD array API."""

    nan = float("nan")

    def __init__(self, cupy):
        self.cupy = cupy
        self.linalg = _CupyLinalg(self)
        self._array_with_attributes = type(
            "_CupyArrayWithAttributes", (cupy.ndarray,), {}
        )

    def __getattr__(self, name):
        if name in {"dot", "array2string", "integer"}:
            raise AttributeError(
                f"The linear-algebra backend has no attribute {name!r}"
            )
        return getattr(self.cupy, name)

    def _preserve_attributes(self, result, source):
        attributes = getattr(source, "__dict__", None)
        if attributes:
            if not isinstance(result, self._array_with_attributes):
                result = result.view(self._array_with_attributes)
            result.__dict__.update(
                {
                    key: self.asarray(item)
                    if isinstance(item, (numpy.ndarray, self.cupy.ndarray))
                    else item
                    for key, item in attributes.items()
                }
            )
        return result

    def asarray(self, value):
        if isinstance(value, self.cupy.ndarray):
            result = value
        else:
            result = self.cupy.asarray(value)

        if result.dtype.kind == "f" and result.dtype != self.cupy.float64:
            result = result.astype(self.cupy.float64)
        elif result.dtype.kind == "c" and result.dtype != self.cupy.complex128:
            result = result.astype(self.cupy.complex128)
        return self._preserve_attributes(result, value)

    def array(self, value):
        return self.asarray(value).copy()

    def to_numpy(self, value):
        if not isinstance(value, self.cupy.ndarray):
            return value if isinstance(value, numpy.ndarray) else numpy.asarray(value)
        result = self.cupy.asnumpy(value)
        attributes = getattr(value, "__dict__", None)
        if attributes:
            result = result.view(_NumpyArrayWithAttributes)
            result.__dict__.update(
                {
                    key: self.to_numpy(item)
                    if isinstance(item, self.cupy.ndarray)
                    else item
                    for key, item in attributes.items()
                }
            )
        return result

    def copy(self, value):
        return self.asarray(value).copy()

    def zeros(self, shape):
        return self.cupy.zeros(shape, dtype=self.cupy.float64)

    def zeros_like(self, value):
        return self.cupy.zeros_like(self.asarray(value))

    def diag(self, value):
        return self.cupy.diag(self.asarray(value))

    def stack(self, values):
        return self.cupy.stack(tuple(self.asarray(value) for value in values))

    def hstack(self, values):
        return self.cupy.hstack(tuple(self.asarray(value) for value in values))

    def vstack(self, values):
        return self.cupy.vstack(tuple(self.asarray(value) for value in values))

    def matmul(self, left, right):
        return self.cupy.matmul(self.asarray(left), self.asarray(right))

    def einsum(self, equation, *operands, **kwargs):
        result = self.cupy.einsum(
            equation,
            *(self.asarray(operand) for operand in operands),
            **kwargs,
        )
        return _scalar_if_needed(result)

    def sum(self, value, axis=None):
        return _scalar_if_needed(self.cupy.sum(self.asarray(value), axis=axis))

    def trace(self, value):
        return _scalar_if_needed(self.cupy.trace(self.asarray(value)))

    def abs(self, value):
        return _scalar_if_needed(self.cupy.abs(self.asarray(value)))

    def sqrt(self, value):
        return _scalar_if_needed(self.cupy.sqrt(self.asarray(value)))

    def log(self, value):
        return _scalar_if_needed(self.cupy.log(self.asarray(value)))

    def isfinite(self, value):
        return _scalar_if_needed(self.cupy.isfinite(self.asarray(value)))

    def expit(self, value):
        result = 1.0 / (1.0 + self.cupy.exp(-self.asarray(value)))
        return _scalar_if_needed(result)


_BACKENDS: dict[str, Any] = {
    "numpy": _NumpyBackend(),
    "torch": None,
    "cupy": None,
}
_backend: Any | None = None
_backend_name: str | None = None


def _get_active_backend():
    if _backend is None:
        raise RuntimeError("The linear-algebra backend has not been configured.")
    return _backend


class _BackendProxy:
    """Forward attribute access to the active backend."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_get_active_backend(), name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(dir(_get_active_backend())))


_backend_proxy = _BackendProxy()


def get_linalg_backend() -> _BackendProxy:
    """Return a proxy that forwards operations to the active backend.

    Raises
    ------
    RuntimeError
        If no backend has been selected yet.
    """
    _get_active_backend()
    return _backend_proxy


def get_linalg_backend_name() -> str:
    """Return the name of the active array and linear-algebra backend."""
    _get_active_backend()
    assert _backend_name is not None
    return _backend_name


def set_linalg_backend(name: str) -> None:
    """Select the process-wide array and linear-algebra backend.

    Parameters
    ----------
    name : {"numpy", "torch", "cupy"}
        Backend to activate. Optional backends are imported lazily when first
        selected.

    Raises
    ------
    ValueError
        If ``name`` is not a supported backend name.
    ImportError
        If an optional backend was requested but is unavailable.
    """
    global _backend, _backend_name

    if name not in _BACKENDS:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"Unsupported linear-algebra backend {name!r}. "
            f"Supported backends: {supported}."
        )

    backend = _BACKENDS[name]
    if backend is None:
        if name == "torch":
            try:
                import torch
            except ImportError as exc:
                raise ImportError(
                    "PyTorch backend requested, but torch is not installed."
                ) from exc
            backend = _TorchBackend(torch)
        elif name == "cupy":
            try:
                import cupy
            except ImportError as exc:
                raise ImportError(
                    "CuPy backend requested, but cupy is not importable. Install "
                    "the CuPy distribution matching the CUDA runtime, or add its "
                    "environment to PYTHONPATH."
                ) from exc
            backend = _CupyBackend(cupy)
        _BACKENDS[name] = backend
    _backend = backend
    _backend_name = name
