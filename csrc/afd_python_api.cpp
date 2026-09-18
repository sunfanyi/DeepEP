#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>

#include "elastic/buffer.hpp"
#include "jit/api.hpp"
#include "utils/event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Isolated DeepEP V2 ElasticBuffer runtime for SGLang AFD";

    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);

    py::class_<deep_ep::EventHandle>(m, "EventHandle", py::module_local())
        .def(py::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    deep_ep::jit::register_apis(m);
    deep_ep::elastic::register_apis(m, true);
}
