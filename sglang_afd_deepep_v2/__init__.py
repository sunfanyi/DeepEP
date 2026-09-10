from __future__ import annotations

from typing import Any

__version__ = "2.0.0+018f21e.afd5"
UPSTREAM_COMMIT = "018f21ea62c6c2888d9b46806aad6d90b399b23d"
TOPK_IDX_BITS = 64

_NATIVE_EXPORTS = frozenset(
    {
        "ElasticBuffer",
        "EventHandle",
        "calculate_elastic_buffer_size",
        "topk_idx_t",
    }
)


def validate_runtime() -> None:
    from .runtime import load_native

    load_native()


def __getattr__(name: str) -> Any:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(name)

    from .runtime import load_native

    return getattr(load_native(), name)


__all__ = [
    "TOPK_IDX_BITS",
    "UPSTREAM_COMMIT",
    "validate_runtime",
    *sorted(_NATIVE_EXPORTS),
]
