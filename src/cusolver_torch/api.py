from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "cusolver_torch_ext.cu"


def _default_arch_list() -> str:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"
    return "7.5"


def _find_nvidia_include_dirs() -> list[str]:
    include_dirs: list[str] = []

    # pip torch wheels expose nvidia/<lib>/include under site-packages
    nvidia_root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    for sub in ("cublas", "cusolver", "cusparse", "cuda_runtime"):
        p = nvidia_root / sub / "include"
        if p.exists():
            include_dirs.append(str(p))

    # conda toolkit often installs headers in targets/<triplet>/include
    prefix = Path(torch.__file__).resolve().parents[4]
    for triplet in ("x86_64-linux", "sbsa-linux"):
        p = prefix / "targets" / triplet / "include"
        if p.exists():
            include_dirs.append(str(p))

    # de-dup while preserving order
    out: list[str] = []
    seen = set()
    for x in include_dirs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _version_key(path: Path) -> tuple[int, ...]:
    parts = re.findall(r"\d+", path.name)
    if not parts:
        return tuple()
    return tuple(int(x) for x in parts)


def _pick_soname(libdir: Path, stem: str) -> str | None:
    """
    Return a link flag argument for one CUDA lib in `libdir`.
    Prefer unversioned .so, else highest versioned .so.*.
    """
    unversioned = libdir / f"lib{stem}.so"
    if unversioned.exists():
        return f"-l:{unversioned.name}"

    candidates = sorted(
        libdir.glob(f"lib{stem}.so*"),
        key=_version_key,
        reverse=True,
    )
    for c in candidates:
        if c.is_file():
            return f"-l:{c.name}"
    return None


def _find_linker_flags() -> list[str]:
    flags: list[str] = []

    def _flags_for_dir(libdir: Path) -> list[str] | None:
        cusolver = _pick_soname(libdir, "cusolver")
        cublas = _pick_soname(libdir, "cublas")
        cusparse = _pick_soname(libdir, "cusparse")
        if not (cusolver and cublas and cusparse):
            return None
        return [
            f"-L{libdir}",
            cusolver,
            cublas,
            cusparse,
            f"-Wl,-rpath,{libdir}",
        ]

    # conda targets lib path (preferred for soname links)
    prefix = Path(torch.__file__).resolve().parents[4]
    for triplet in ("x86_64-linux", "sbsa-linux"):
        libdir = prefix / "targets" / triplet / "lib"
        if libdir.exists():
            dflags = _flags_for_dir(libdir)
            if dflags is not None:
                return dflags

    # pip nvidia libs fallback
    nvidia_root = Path(torch.__file__).resolve().parent.parent / "nvidia"
    cublas_lib = nvidia_root / "cublas" / "lib"
    cusolver_lib = nvidia_root / "cusolver" / "lib"
    cusparse_lib = nvidia_root / "cusparse" / "lib"

    # some wheel layouts keep all three in separate dirs; add search/rpath for each
    for d in (cublas_lib, cusolver_lib, cusparse_lib):
        if d.exists():
            flags.append(f"-L{d}")
            flags.append(f"-Wl,-rpath,{d}")

    # Prefer exact filenames if discoverable, else generic -l names.
    cusolver = _pick_soname(cusolver_lib, "cusolver") if cusolver_lib.exists() else None
    cublas = _pick_soname(cublas_lib, "cublas") if cublas_lib.exists() else None
    cusparse = _pick_soname(cusparse_lib, "cusparse") if cusparse_lib.exists() else None
    flags.extend([cusolver or "-lcusolver", cublas or "-lcublas", cusparse or "-lcusparse"])
    return flags


@lru_cache(maxsize=1)
def _ext():
    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST", os.environ.get("CUSOLVER_TORCH_CUDA_ARCH_LIST", _default_arch_list())
    )
    name = os.environ.get("CUSOLVER_TORCH_EXT_NAME", "cusolver_torch_ext")
    verbose = os.environ.get("CUSOLVER_TORCH_VERBOSE", "0") == "1"

    return load(
        name=name,
        sources=[str(_SRC)],
        extra_include_paths=_find_nvidia_include_dirs(),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        extra_ldflags=_find_linker_flags(),
        with_cuda=True,
        verbose=verbose,
    )


def eigh(
    a: torch.Tensor,
    compute_vectors: bool = True,
    lower: bool = True,
    driver: str = "syevd",
    tol: float = 1e-7,
    max_sweeps: int = 100,
    sort_eig: bool = True,
    il: int = 1,
    iu: int = -1,
    copy_input: bool = True,
    deterministic_mode: int = 0,
    return_meig: bool = False,
):
    """cuSOLVER wrapper for symmetric eigendecomposition on CUDA torch tensors.

    Returns `(eigvals, eigvecs, info)` by default.
    If `return_meig=True`, returns `(eigvals, eigvecs, info, meig)`.
    """
    if not a.is_cuda:
        raise ValueError("Input tensor must be on CUDA")

    out = _ext().eigh_cuda(
        a,
        compute_vectors,
        lower,
        driver,
        tol,
        max_sweeps,
        sort_eig,
        il,
        iu,
        copy_input,
        deterministic_mode,
    )
    if return_meig:
        return out
    return out[0], out[1], out[2]
