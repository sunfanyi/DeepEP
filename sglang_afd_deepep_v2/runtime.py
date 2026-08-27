from __future__ import annotations

import functools
import importlib
import os
import shutil
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType

import torch

_TOPK_IDX_BITS = 64


def _resolve_cuda_root() -> Path:
    candidates: list[Path] = []
    if configured := os.environ.get("CUDA_NVCC_EXECUTABLE"):
        candidates.append(Path(configured).expanduser().parent.parent)
    for name in ("CUDA_HOME", "CUDA_PATH"):
        if configured := os.environ.get(name):
            candidates.append(Path(configured).expanduser())
    if nvcc := shutil.which("nvcc"):
        candidates.append(Path(nvcc).resolve().parent.parent)
    candidates.append(Path("/usr/local/cuda"))

    for root in dict.fromkeys(path.resolve(strict=False) for path in candidates):
        if (root / "bin" / "nvcc").is_file() and (root / "bin" / "cuobjdump").is_file():
            return root
    raise RuntimeError(
        "sglang-afd-deepep-v2 requires nvcc and cuobjdump for kernel JIT; "
        "set CUDA_HOME"
    )


def _packaged_nccl_root() -> Path | None:
    if not torch.version.cuda:
        return None
    package_name = f"nvidia-nccl-cu{torch.version.cuda.split('.', maxsplit=1)[0]}"
    try:
        package = distribution(package_name)
    except PackageNotFoundError:
        return None
    return Path(package.locate_file("nvidia/nccl")).resolve()


def _resolve_nccl_root() -> Path:
    candidates: list[Path] = []
    for name in ("EP_NCCL_ROOT_DIR", "NCCL_DIR"):
        if configured := os.environ.get(name):
            candidates.append(Path(configured).expanduser())
    if packaged_root := _packaged_nccl_root():
        candidates.append(packaged_root)
    candidates.extend([Path("/usr/local/nccl"), Path("/usr/local/cuda")])

    required_headers = (
        Path("include/nccl.h"),
        Path("include/nccl_device.h"),
        Path("include/nccl_device/core.h"),
    )
    for root in dict.fromkeys(path.resolve(strict=False) for path in candidates):
        if all((root / header).is_file() for header in required_headers):
            return root
    raise RuntimeError(
        "Cannot find NCCL device headers; set EP_NCCL_ROOT_DIR to the NCCL root"
    )


def _configure_topk_idx_bits() -> None:
    configured = os.environ.get("EP_NUM_TOPK_IDX_BITS")
    if configured is not None and configured != str(_TOPK_IDX_BITS):
        raise RuntimeError(
            f"EP_NUM_TOPK_IDX_BITS must be {_TOPK_IDX_BITS}, got {configured!r}"
        )
    os.environ.setdefault("EP_NUM_TOPK_IDX_BITS", str(_TOPK_IDX_BITS))


@functools.cache
def load_native() -> ModuleType:
    _configure_topk_idx_bits()
    module = importlib.import_module("sglang_afd_deepep_v2._C")
    package_root = Path(__file__).resolve().parent
    module.init_jit(
        str(package_root), str(_resolve_cuda_root()), str(_resolve_nccl_root())
    )
    return module


__all__ = ["load_native"]
