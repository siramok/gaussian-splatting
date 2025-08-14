// sample_mesh_uniform.cpp

#include <iostream>
#include <numeric>
#include <cstring>

#include <viskores/cont/Initialize.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/filter/resampling/Probe.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperRayTracer.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>
#include <viskores/cont/Timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <viskores/rendering/MapperVolume.h>
#include <viskores/rendering/Camera.h>
#include <viskores/cont/ArrayRangeCompute.h>
#include <sstream>
#include <iomanip>

namespace py = pybind11;

// 512x512 volume rendering; time 100 render+save iterations
void render_volume_bench(
    py::array_t<int64_t> dims_arr,
    py::array_t<float>   origin_arr,
    py::array_t<float>   spacing_arr,
    py::array_t<double>  val_arr)
{
  // 0) CUDA device & timer
  auto &tracker = viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto cudaTag = viskores::cont::DeviceAdapterTagCuda();
  viskores::cont::Timer timer(cudaTag);

  // 1) Initialize
  timer.Start();
  viskores::cont::Initialize();
  timer.Stop();
  std::cout << "Initialize: " << timer.GetElapsedTime() << " s\n";

  // 2) Build uniform dataset
  timer.Reset(); timer.Start();

  auto dims_buf = dims_arr.request();
  if (dims_buf.ndim != 1 || dims_buf.size != 3)
    throw std::runtime_error("dims_arr must be length-3 [nx, ny, nz].");
  auto dims_ptr = static_cast<int64_t*>(dims_buf.ptr);
  viskores::Id nx = dims_ptr[0], ny = dims_ptr[1], nz = dims_ptr[2];

  auto orig_buf = origin_arr.request();
  if (orig_buf.ndim != 1 || orig_buf.size != 3)
    throw std::runtime_error("origin_arr must be length-3.");
  auto o_ptr = static_cast<float*>(orig_buf.ptr);

  auto sp_buf = spacing_arr.request();
  if (sp_buf.ndim != 1 || sp_buf.size != 3)
    throw std::runtime_error("spacing_arr must be length-3.");
  auto s_ptr = static_cast<float*>(sp_buf.ptr);

  auto val_buf = val_arr.request();
  auto v_ptr   = static_cast<double*>(val_buf.ptr);
  std::size_t totalPts = static_cast<std::size_t>(nx) * ny * nz;
  if (val_buf.size != static_cast<py::ssize_t>(totalPts))
    throw std::runtime_error("val_arr size must equal nx*ny*nz.");

  std::vector<viskores::Float64> val_vec(v_ptr, v_ptr + totalPts);
  auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));

  viskores::Id3 dims3{nx, ny, nz};
  viskores::Vec3f origin{ o_ptr[0], o_ptr[1], o_ptr[2] };
  viskores::Vec3f spacing{ s_ptr[0], s_ptr[1], s_ptr[2] };

  auto ds = viskores::cont::DataSetBuilderUniform::Create(dims3, origin, spacing, "coords");
  ds.AddPointField("value", valHandle);

  timer.Stop();
  std::cout << "ReadDataSet: " << timer.GetElapsedTime() << " s\n";

  // 3) Volume rendering setup (512x512) with linear opacity TF
  timer.Reset(); timer.Start();

  viskores::rendering::Actor actor(
      ds.GetCellSet(),
      ds.GetCoordinateSystem("coords"),
      ds.GetField("value"));
  viskores::Range range{0.0, 1.0};
  actor.SetScalarRange(range);

  viskores::rendering::Scene scene;
  scene.AddActor(actor);

  viskores::rendering::MapperVolume mapper;
  constexpr int kWidth  = 512;
  constexpr int kHeight = 512;
  viskores::rendering::CanvasRayTracer canvas(kWidth, kHeight);

  // Camera: look along +X so the image shows Y (horizontal) and Z (vertical)
  viskores::rendering::Camera camera;
  auto bounds = ds.GetCoordinateSystem("coords").GetBounds();
  // Compute center and an offset in +X
  const double cx = 0.5 * (bounds.X.Min + bounds.X.Max);
  const double cy = 0.5 * (bounds.Y.Min + bounds.Y.Max);
  const double cz = 0.5 * (bounds.Z.Min + bounds.Z.Max);
  const double dx = (bounds.X.Max - bounds.X.Min);
  const double offset = (dx > 0.0 ? 1.5 * dx : 1.0); // fallback offset if degenerate

  viskores::Vec3f pos{ static_cast<float>(bounds.X.Max + offset),
                       static_cast<float>(cy),
                       static_cast<float>(cz) };
  viskores::Vec3f look{ static_cast<float>(cx),
                        static_cast<float>(cy),
                        static_cast<float>(cz) };

  camera.SetPosition(pos);
  camera.SetLookAt(look);
  camera.SetViewUp(viskores::Vec3f{0.f, 0.f, 1.f}); // Z up ⇒ Y is horizontal

  viskores::rendering::View3D view(scene, mapper, canvas, camera);

  timer.Stop();
  std::cout << "Render setup: " << timer.GetElapsedTime() << " s\n";

  // 4) Benchmark: 100 renders + saves (includes file I/O)
  timer.Reset(); timer.Start();

  const int iters = 100;
  for (int i = 0; i < iters; ++i)
  {
    view.Paint();

    std::ostringstream oss;
    oss << "testi/test" << '_' << std::setw(3) << std::setfill('0') << i << ".png";
    view.SaveAs(oss.str());
  }

  timer.Stop();
  double total = timer.GetElapsedTime();
  std::cout << "100 volume renders+saves (512x512, view +X over YZ): " << total << " s\n";
  std::cout << "Avg per frame: " << (total / iters) << " s\n";
}


py::array_t<double> sample_mesh(
    py::array_t<int64_t> dims_arr,
    py::array_t<float>   origin_arr,
    py::array_t<float>   spacing_arr,
    py::array_t<double>  val_arr,
    py::array_t<float>   samp_arr)
{
  // 0) Pick CUDA device for timing async kernels
  auto &tracker = viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  auto cudaTag = viskores::cont::DeviceAdapterTagCuda();
  viskores::cont::Timer timer(cudaTag);

  // 1) Initialization
  timer.Start();
  viskores::cont::Initialize();
  timer.Stop();
  std::cout << "Initialize: " 
            << timer.GetElapsedTime() << " s\n";

  // 2) Read and wrap input arrays
  timer.Reset();
  timer.Start();

  // dims
  auto dims_buf = dims_arr.request();
  auto dims_ptr = static_cast<int64_t*>(dims_buf.ptr);
  viskores::Id nx = dims_ptr[0],
                   ny = dims_ptr[1],
                   nz = dims_ptr[2];

  // origin
  auto orig_buf = origin_arr.request();
  auto o_ptr = static_cast<float*>(orig_buf.ptr);

  // spacing
  auto sp_buf = spacing_arr.request();
  auto s_ptr = static_cast<float*>(sp_buf.ptr);

  // values
  auto val_buf = val_arr.request();
  auto v_ptr   = static_cast<double*>(val_buf.ptr);
  std::size_t totalPts = static_cast<std::size_t>(nx) * ny * nz;
  std::vector<viskores::Float64> val_vec(v_ptr, v_ptr + totalPts);
  auto valHandle = viskores::cont::make_ArrayHandleMove(std::move(val_vec));

  // build uniform DataSet
  viskores::Id3 dims3{nx, ny, nz};
  viskores::Vec3f origin{ o_ptr[0], o_ptr[1], o_ptr[2] };
  viskores::Vec3f spacing{ s_ptr[0], s_ptr[1], s_ptr[2] };
  auto inData = viskores::cont::DataSetBuilderUniform::Create(
    dims3, origin, spacing, "coords"
  );
  inData.AddPointField("value", valHandle);

  timer.Stop();
  std::cout << "ReadDataSet: " 
            << timer.GetElapsedTime() << " s\n";

  // 3) Build explicit point‐vertex grid for sampling locations
  timer.Reset();
  timer.Start();

  auto samp_buf = samp_arr.request();
  std::size_t n_samples = samp_buf.shape[0];
  auto samp_ptr = static_cast<float*>(samp_buf.ptr);

  std::vector<viskores::Vec<float,3>> sample_coords(n_samples);
  std::memcpy(
    sample_coords.data(),
    samp_ptr,
    n_samples * 3 * sizeof(float)
  );

  std::vector<viskores::Id> sample_conn(n_samples);
  std::iota(sample_conn.begin(), sample_conn.end(), 0);

  auto explicitGrid = viskores::cont::DataSetBuilderExplicit::Create(
    sample_coords,
    viskores::CellShapeTagVertex{},
    static_cast<viskores::IdComponent>(1),
    sample_conn,
    "sample_coords"
  );

  timer.Stop();
  std::cout << "Build explicit grid: " 
            << timer.GetElapsedTime() << " s\n";

  // 4) Probe filter setup
  timer.Reset();
  timer.Start();

  viskores::filter::resampling::Probe probe;
  probe.SetGeometry(explicitGrid);
  probe.SetInvalidValue(-1.0);

  timer.Stop();
  std::cout << "Probe setup: " 
            << timer.GetElapsedTime() << " s\n";

  // 5) Probe execution
  timer.Reset();
  timer.Start();

  viskores::cont::DataSet sampled = probe.Execute(inData);

  timer.Stop();
  std::cout << "Probe execute: " 
            << timer.GetElapsedTime() << " s\n";

  // 6) Retrieve and return result
  timer.Reset();
  timer.Start();

  const auto array = sampled.GetPointField("value").GetData();
  auto concrete = array.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Float64>>();
  concrete.SyncControlArray();
  auto readPortal = concrete.ReadPortal();

  std::size_t n = readPortal.GetNumberOfValues();
  py::array_t<double> result(n);
  auto out_ptr = result.mutable_data();
  for (std::size_t i = 0; i < n; ++i)
  {
    out_ptr[i] = readPortal.Get(i);
  }

  timer.Stop();
  std::cout << "Data retrieval: " 
            << timer.GetElapsedTime() << " s\n";

  return result;
}