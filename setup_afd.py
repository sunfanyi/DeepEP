from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

from sglang_afd_deepep_v2 import __version__ as PACKAGE_VERSION

ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "sglang_afd_deepep_v2"
TOPK_IDX_BITS = 64


def _load_find_pkgs():
    module_path = ROOT / "deep_ep" / "utils" / "find_pkgs.py"
    spec = importlib.util.spec_from_file_location("deep_ep_find_pkgs", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load package discovery from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_nccl() -> tuple[Path, Path]:
    root = Path(_load_find_pkgs().find_nccl_root()).resolve()
    include_dir = root / "include"
    required_headers = (
        include_dir / "nccl.h",
        include_dir / "nccl_device.h",
        include_dir / "nccl_device" / "core.h",
    )
    missing = [path for path in required_headers if not path.is_file()]
    if missing:
        raise RuntimeError(
            "DeepEP V2 requires NCCL device headers; missing="
            + ", ".join(str(path) for path in missing)
        )

    lib_dir = root / "lib"
    if not lib_dir.is_dir() and (root / "lib64").is_dir():
        lib_dir = root / "lib64"
    candidates = [lib_dir / "libnccl.so", lib_dir / "libnccl.so.2"]
    library = next((path for path in candidates if path.is_file()), None)
    if library is None:
        library = next(iter(sorted(lib_dir.glob("libnccl.so.2*"))), None)
    if library is None:
        raise RuntimeError(f"No NCCL shared library found under {lib_dir}")
    return include_dir, library.resolve(strict=True)


NCCL_INCLUDE_DIR, NCCL_LIBRARY = _resolve_nccl()
if os.environ.get("EP_NUM_TOPK_IDX_BITS", "64") != str(TOPK_IDX_BITS):
    raise RuntimeError(f"EP_NUM_TOPK_IDX_BITS must be {TOPK_IDX_BITS}")
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    raise RuntimeError(
        "TORCH_CUDA_ARCH_LIST must be set when building the portable AFD wheel"
    )
if shutil.which("ninja") is None:
    raise RuntimeError("ninja is required to build the AFD wheel with CUDA RDC")
if not (ROOT / "third-party" / "fmt" / "include" / "fmt" / "format.h").is_file():
    raise RuntimeError("Initialize the third-party/fmt submodule before building")


class AFDBuildPy(build_py):
    def run(self) -> None:
        super().run()
        package_dir = Path(self.build_lib) / PACKAGE_NAME
        shutil.copytree(
            ROOT / "deep_ep" / "include",
            package_dir / "include",
            dirs_exist_ok=True,
        )
        licenses_dir = package_dir / "licenses"
        licenses_dir.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "LICENSE", licenses_dir / "LICENSE.deep_ep")
        shutil.copy2(
            ROOT / "third-party" / "fmt" / "LICENSE",
            licenses_dir / "LICENSE.fmt",
        )

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        source_root = ROOT / "deep_ep" / "include"
        target_root = Path(self.build_lib) / PACKAGE_NAME / "include"
        outputs.extend(
            str(target_root / path.relative_to(source_root))
            for path in source_root.rglob("*")
            if path.is_file()
        )
        outputs.extend(
            str(Path(self.build_lib) / PACKAGE_NAME / "licenses" / name)
            for name in ("LICENSE.deep_ep", "LICENSE.fmt")
        )
        return outputs


include_dirs = [
    str(ROOT / "deep_ep" / "include"),
    str(ROOT / "csrc"),
    str(ROOT / "third-party" / "fmt" / "include"),
    str(NCCL_INCLUDE_DIR),
]
if CUDA_HOME is not None:
    cccl_include = Path(CUDA_HOME) / "include" / "cccl"
    if cccl_include.is_dir():
        include_dirs.append(str(cccl_include))

common_defines = [
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
    f"-DEP_NUM_TOPK_IDX_BITS={TOPK_IDX_BITS}",
]

setuptools.setup(
    name="sglang-afd-deepep-v2",
    version=PACKAGE_VERSION,
    description="Isolated DeepEP V2 ElasticBuffer runtime for SGLang AFD",
    license="MIT",
    license_files=["LICENSE"],
    python_requires=">=3.10",
    packages=[PACKAGE_NAME],
    ext_modules=[
        CUDAExtension(
            name=f"{PACKAGE_NAME}._C",
            sources=[
                str(ROOT / "csrc" / "afd_python_api.cpp"),
                str(ROOT / "csrc" / "kernels" / "backend" / "nccl.cu"),
                str(ROOT / "csrc" / "kernels" / "backend" / "cuda_driver.cu"),
            ],
            include_dirs=include_dirs,
            extra_objects=[str(NCCL_LIBRARY)],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-Wno-deprecated-declarations",
                    "-Wno-unused-variable",
                    "-Wno-sign-compare",
                    "-Wno-reorder",
                    "-Wno-attributes",
                    *common_defines,
                ],
                "nvcc": [
                    "-O3",
                    "-Xcompiler",
                    "-O3",
                    "--extended-lambda",
                    "--diag-suppress=128,2417",
                    "-rdc=true",
                    "--ptxas-options=--register-usage-level=10",
                    *common_defines,
                ],
                "nvcc_dlink": ["-dlink"],
            },
            extra_link_args=["-lcuda"],
        )
    ],
    cmdclass={"build_ext": BuildExtension, "build_py": AFDBuildPy},
    zip_safe=False,
)
