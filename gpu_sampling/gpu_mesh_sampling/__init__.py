from ._gpu_mesh_sampling import sample_mesh, render_volume_bench
import numpy as np

def gpu_sample(
    dims: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    values: np.ndarray,
    samples: np.ndarray
):
    return sample_mesh(dims, origin, spacing, values, samples)

def gpu_volume_bench(
    dims: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    values: np.ndarray
):
    return render_volume_bench(dims, origin, spacing, values)