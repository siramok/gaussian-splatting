#include "sample_mesh.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_gpu_mesh_sampling, m) {
  m.def("sample_mesh", &sample_mesh);
  m.def("render_volume_bench", &render_volume_bench);
}