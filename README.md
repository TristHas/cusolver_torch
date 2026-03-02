# cusolver-torch

Minimal Python package exposing cuSOLVER eigensolvers on CUDA torch tensors.

## Install

```bash
pip install -e .
```

From a git URL:

```bash
pip install "git+https://github.com/<you>/cusolver_torch.git"
```

## Quick usage

```python
import torch
from cusolver_torch import eigh

A = torch.randn(32, 128, 128, device="cuda", dtype=torch.float32)
A = (A + A.transpose(-1, -2)) * 0.5
w, v, info = eigh(A, driver="xsyev_batched", compute_vectors=True)
```

## API

`eigh(a, *, compute_vectors=True, lower=True, driver="syevd", tol=1e-7, max_sweeps=100, sort_eig=True, il=1, iu=-1, copy_input=True, deterministic_mode=0, return_meig=False)`

- `driver`: `"syevd" | "syevj" | "syevj_batched" | "syevdx" | "xsyev_batched"`
- `deterministic_mode`: `0` keep current handle mode, `1` deterministic, `2` allow non-deterministic.

`return_meig=True` returns `(w, v, info, meig)`.

## Notes

- Input `a` must be CUDA and shape `(N, N)` or `(B, N, N)`.
- Eigenvectors returned follow torch convention: `A @ V = V @ diag(w)`.
- Extension is built JIT on first import/call via `torch.utils.cpp_extension.load`.

### Optional env vars

- `CUSOLVER_TORCH_CUDA_ARCH_LIST` (defaults to current GPU capability)
- `CUSOLVER_TORCH_EXT_NAME` (default: `cusolver_torch_ext`)
- `CUSOLVER_TORCH_VERBOSE` (`1` enables verbose extension build logs)
