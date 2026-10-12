"""Native ElasticBuffer extensions for attention-FFN disaggregation."""

from importlib.metadata import version

from ._C import (
    ElasticBuffer,
    EventHandle,
    calculate_elastic_buffer_size,
    topk_idx_t,
)

__version__ = version("sgl-deep-ep")

__all__ = [
    "ElasticBuffer",
    "EventHandle",
    "calculate_elastic_buffer_size",
    "topk_idx_t",
]
