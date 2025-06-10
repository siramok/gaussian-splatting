import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import os
import argparse

SH_C0 = 0.28209479177387814

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def inverse_sigmoid(y):
    return np.log(y / (1.0 - y))

def apply_spherical_harmonics(value, cmap_name):
    vmin, vmax = value.min(), value.max()
    normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(value)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(normalized)[:, :3]
    f_dc_0 = np.clip(((colors[:, 0] - 0.5) / SH_C0 * 255), 0, 255).astype(np.uint8)
    f_dc_1 = np.clip(((colors[:, 1] - 0.5) / SH_C0 * 255), 0, 255).astype(np.uint8)
    f_dc_2 = np.clip(((colors[:, 2] - 0.5) / SH_C0 * 255), 0, 255).astype(np.uint8)
    return f_dc_0, f_dc_1, f_dc_2

def clean_and_process_ply(in_path, colormaps, constant_opacity):
    file_name = os.path.splitext(os.path.basename(in_path))[0]

    with open(in_path, 'rb') as f:
        plydata = PlyData.read(f)
    
    vertex_data = plydata['vertex'].data

    x = vertex_data['x'].astype(np.float64)
    y = vertex_data['y'].astype(np.float64)
    z = vertex_data['z'].astype(np.float64)
    value = vertex_data['value'].astype(np.float64)

    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(value) |
                   np.isinf(x) | np.isinf(y) | np.isinf(z) | np.isinf(value))

    cleaned_data = vertex_data[valid_mask]
    print(f"Cleaned vertices: {len(cleaned_data)} / Original vertices: {len(vertex_data)}")

    cleaned_data = cleaned_data.copy()
    cleaned_data['value'] = sigmoid(cleaned_data['value'].astype(np.float64)).astype(cleaned_data['value'].dtype)

    inverse_sigmoid_opacity = inverse_sigmoid(np.full(cleaned_data.shape, constant_opacity))

    for cmap_name in colormaps:
        f_dc_0, f_dc_1, f_dc_2 = apply_spherical_harmonics(cleaned_data['value'], cmap_name)

        new_dtype = cleaned_data.dtype.descr + [
            ('f_dc_0', 'u1'),
            ('f_dc_1', 'u1'),
            ('f_dc_2', 'u1'),
            ('opacity', 'f4')
        ]

        new_data = np.empty(cleaned_data.shape, dtype=new_dtype)
        for name in cleaned_data.dtype.names:
            new_data[name] = cleaned_data[name]

        new_data['f_dc_0'] = f_dc_0
        new_data['f_dc_1'] = f_dc_1
        new_data['f_dc_2'] = f_dc_2
        new_data['opacity'] = inverse_sigmoid_opacity.astype('f4')

        vertex_element = PlyElement.describe(new_data, 'vertex')
        os.makedirs('output', exist_ok=True)
        out_path = f'output/{file_name}_{cmap_name}.ply'
        PlyData([vertex_element]).write(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert transfer function .ply to standard 3DGS .ply format with inverse sigmoid opacity.")
    parser.add_argument("in_path", type=str, help="Path to input .ply file.")
    parser.add_argument("--opacity", type=float, default=0.005, help="Constant opacity value (after sigmoid, range (0,1)).")
    args = parser.parse_args()

    colormaps = ['viridis']
    clean_and_process_ply(args.in_path, colormaps, constant_opacity=args.opacity)