# PySMD: Open-source Driver for First-Principles Shadow Molecular Dynamics

LANL software release: O5139

## Installation

Install PySMD and its runtime dependencies from PyPI:

```console
python -m pip install pysmd
```

PySMD has been developed and tested primarily with PySCF 2.11. To use an
existing PySCF source tree through `PYTHONPATH`, install PySMD without
dependencies:

```console
export PYTHONPATH=/path/to/pyscf:${PYTHONPATH}
python -m pip install --no-deps pysmd
```

## Testing an installation

Install the optional test dependency and run the packaged test suite:

```console
python -m pip install "pysmd[test]"
python -m pytest --pyargs pysmd
```

## Linear algebra backend

PySMD configures NumPy as its process-wide linear algebra backend when the
package is imported. PyTorch is available as an optional backend when it is
installed:

```python
import pysmd
from pysmd.lib import linalg_helper

linalg_helper.set_linalg_backend("torch")
```

Configure the backend before creating PySMD calculation objects. The PyTorch
backend uses CPU tensors with double precision for floating-point and complex
inputs. Selecting `"torch"` raises `ImportError` if PyTorch is unavailable; it
does not install dependencies or change the previously active backend.

CuPy is also available as an optional device backend. Its distribution name
depends on the CUDA runtime, so PySMD does not declare it as a hard dependency:

```python
from pysmd.lib import linalg_helper

linalg_helper.set_linalg_backend("cupy")
```

CuPy preserves integer and Boolean input dtypes and promotes floating-point and
complex inputs to double precision. A CPU PySCF electronic-structure interface
can be combined with CuPy MD arrays by selecting CuPy before constructing the
interface. Numerical arrays are converted explicitly at the PySCF boundary.

## GPU4PySCF backend

RHF and RKS electronic-structure operations can use GPU4PySCF explicitly:

```python
from pyscf import gto
from gpu4pyscf import dft
from pysmd.interface.pyscf import PyscfDriver

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="cc-pvdz")
gpu_mf = dft.RKS(mol, xc="pbe")
interface = PyscfDriver(pyscf_mf=gpu_mf, backend="gpu")
```

The factory defaults to `backend="cpu"`. CPU mode accepts PySCF mean-field
objects only. GPU mode accepts GPU4PySCF mean-field objects only; PySMD does not
silently call `to_cpu()` or `to_gpu()` for a mismatched object. A PySCF molecule
is backend-neutral and can construct either implementation:

```python
cpu_rhf = PyscfDriver(pyscf_mol=mol)
cpu_rks = PyscfDriver(pyscf_mol=mol, backend="cpu", xc="pbe")
gpu_rhf = PyscfDriver(pyscf_mol=mol, backend="gpu")
gpu_rks = PyscfDriver(pyscf_mol=mol, backend="gpu", xc="pbe")
```

`backend="auto"` detects CPU or GPU from `pyscf_mf`. Molecule-only auto mode
remains CPU:

```python
interface = PyscfDriver(pyscf_mf=gpu_mf, backend="auto")
```

A GPU interface selects CuPy for PySMD arrays. Its
`electronic_backend`, `gradient_backend`, and `array_backend` attributes report
the active arrangement. GPU4PySCF 1.7.1 does not provide the asymmetric
linearized shadow-gradient contract required by PySMD, so gradients are
evaluated by a synchronized CPU PySCF delegate created with
`pyscf_mf.to_cpu()`. Construction emits one `RuntimeWarning` describing this
fallback. Electronic matrices, quadrature, and energies remain on the device.

GPU4PySCF and CuPy are optional because their compatible packages depend on the
CUDA runtime. Install both according to their upstream instructions. For local
source trees, make both packages importable, for example:

```console
export PYTHONPATH=/path/to/pyscf:/path/to/gpu4pyscf:${PYTHONPATH}
```

If GPU4PySCF or CuPy cannot be imported, PySMD raises an actionable
`ImportError` and leaves the previously selected array backend unchanged.
