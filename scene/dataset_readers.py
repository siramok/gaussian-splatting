#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import gc
import random
import shutil
from typing import NamedTuple

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from plyfile import PlyData, PlyElement
from vtk import vtkMatrix3x3, vtkMatrix4x4
from sklearn.neighbors import NearestNeighbors
from skimage.filters import gaussian
from scipy.stats import entropy
import time
from pathlib import Path
import json
import numpy as np
from typing import NamedTuple
from vtk import vtkXMLPolyDataReader
import matplotlib.pyplot as plt

from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    mvt_matrix: np.array
    proj_matrix: np.array
    center: np.array
    colormap_id: int
    opacitymap_id: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    bounding_box: list
    mesh: pv.PolyData = None


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    values = np.vstack(vertices["value"]).T
    return BasicPointCloud(points=positions, values=values)


def storePly(path, xyz, values):
    values = values.reshape(-1, 1)
    print(xyz.shape, values.shape)
    if xyz.shape[0] != values.shape[0]:
        raise ValueError(
            f"Mismatch in number of points: mesh has {xyz.shape[0]} points, "
            f"but values has {values.shape[0]} entries."
        )
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("value", "f4"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, values), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def arrayFromVTKMatrix(vmatrix):
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray.astype(np.float32)


def random_dropout(mesh, values, dropout_percentage):
    num_points = mesh.n_points
    
    # Create a random boolean mask directly (more memory efficient)
    keep_percentage = 1 - dropout_percentage
    mask = np.random.random(num_points) < keep_percentage
    
    # Apply the mask to the points and values
    new_points = mesh.points[mask]
    new_values = mesh.point_data["value"][mask]
    
    # Create a new PyVista PolyData mesh
    new_mesh = pv.PolyData(new_points)
    new_mesh.point_data["value"] = new_values

    return new_mesh, new_values


def density_based_dropout(
    mesh, values, high_density_dropout, low_density_dropout, n_neighbors=10
):
    points = mesh.points

    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(points)
    distances, _ = nbrs.kneighbors(points)

    mean_distances = distances.mean(axis=1)
    density_scores = 1 - (mean_distances - mean_distances.min()) / (
        mean_distances.max() - mean_distances.min()
    )

    dropout_probs = (
        low_density_dropout
        + (high_density_dropout - low_density_dropout) * density_scores
    )
    keep_mask = np.random.random(len(points)) > dropout_probs

    new_mesh = pv.PolyData(points[keep_mask])
    new_values = values[keep_mask]

    return new_mesh, new_values


def storeRawPly(path, mesh, values):
    values = values.reshape(-1, 1)

    xyz = mesh.points  # Shape (N, 3)

    if xyz.shape[0] != values.shape[0]:
        raise ValueError(
            f"Mismatch in number of points: mesh has {xyz.shape[0]} points, "
            f"but values has {values.shape[0]} entries."
        )

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("value", "f4"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["value"] = values[:, 0]

    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def buildRawDataset(path, filename, colormaps, opacitymaps, num_control_points, resolution, spacing):
    # Directory setup
    image_dir = os.path.join(path, "images")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    # Window setup
    width = resolution
    height = resolution
    ratio = width / height
    pv.start_xvfb()
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]
    # print(pv.get_gpu_info())

    base_name = filename.rsplit(".", 1)[0]
    parts = base_name.split("_")
    dimensions = tuple(map(int, parts[-2].split("x")))
    data_type = parts[-1]
    print(f"Raw file dimensions: {dimensions}, data_type: {data_type}")

    dtype_map = {
        "uint8": np.uint8,
        "int8": np.int8,
        "uint16": np.uint16,
        "int16": np.int16,
        "uint32": np.uint32,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }

    values = np.fromfile(os.path.join(path, filename), dtype=dtype_map[data_type])

    # Ensure the size matches the dimensions
    if values.size != dimensions[0] * dimensions[1] * dimensions[2]:
        raise ValueError("Data size does not match the specified dimensions.")

    # Reshape data into a 3D array and normalize to [0, 1]
    values = values.astype(np.float32).reshape(dimensions, order='F')
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)

    mesh = pv.ImageData()
    mesh.dimensions = np.array(values.shape)
    mesh.spacing = spacing
    mesh.point_data["value"] = values.ravel(order="F").astype(np.float32)

    # Point scaling
    points_min = np.array(mesh.origin)
    points_max = points_min + (np.array(mesh.dimensions) - 1) * np.array(mesh.spacing)
    points_max_abs = max(np.max(np.abs(points_min)), np.max(np.abs(points_max)))

    if points_max_abs > 1:
        scale_factor = 1.0 / points_max_abs
        mesh.spacing = (scale_factor, scale_factor, scale_factor)

    # Get the focal point so that we can translate the mesh to the origin
    offset = list(pl.camera.focal_point)
    # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.origin = offset

    cam_infos = []
    image_counter = 0

    for opacitymap_id, opacitymap in enumerate(opacitymaps):
        for colormap_id, colormap in enumerate(colormaps):
            cmap = plt.cm.get_cmap(colormap, num_control_points)
            pl.clear()
            pl.add_volume(
                mesh,
                show_scalar_bar=False,
                scalars="value",
                cmap=cmap,
                opacity=opacitymap * 255,
                blending="composite",
                shade=False,
                diffuse=0.0,
                specular=0.0,
                ambient=1.0,
                opacity_unit_distance=None,
            )

            # Reset the camera position and focal point, since we translated the mesh
            pl.view_xy()
            pl.background_color = "black"
            pl.camera.clipping_range = (0.001, 1000.0)
            camera = pl.camera

            # Controls the camera orbit and capture frequency
            print(
                f"Generating images using {colormap} colormap, opacitymap {opacitymap_id}, {resolution}x{resolution} resolution, and {spacing} spacing"
            )
            azimuth_steps = 18
            elevation_steps = 7
            azimuth_range = range(0, 360, 360 // azimuth_steps)
            # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
            elevation_range = range(-35, 35, 70 // elevation_steps)

            for elevation in elevation_range:
                for azimuth in azimuth_range:
                    # Set new azimuth and elevation
                    camera.elevation = elevation
                    camera.azimuth = azimuth

                    # Produce a new render at the new camera position
                    pl.render()

                    # Save the render as a new image
                    image_name = f"{colormap}_omap{opacitymap_id}_{image_counter:05d}.png"
                    image_path = os.path.join(image_dir, image_name)
                    pl.screenshot(image_path)

                    mvt_matrix = np.linalg.inv(
                        arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
                    )

                    # Not sure why this is necessary
                    mvt_matrix[:3, 1:3] *= -1

                    R = mvt_matrix[:3, :3].T
                    T = mvt_matrix[:3, 3]

                    FovY = np.radians(camera.view_angle)
                    FovX = focal2fov(fov2focal(FovY, height), width)

                    proj_matrix = arrayFromVTKMatrix(
                        camera.GetCompositeProjectionTransformMatrix(ratio, 0.001, 1000.0)
                    )

                    # Not sure why this is necessary
                    proj_matrix[1, :] = -proj_matrix[1, :]
                    proj_matrix[2, :] = -proj_matrix[2, :]

                    # Not sure why this is necessary
                    y = camera.position[1]
                    if y < 0:
                        mvt_matrix[2, 1] *= -1
                    mvt_matrix[2, 3] = abs(mvt_matrix[2, 3])

                    center = mvt_matrix[:3, 3]

                    cam_info = CameraInfo(
                        uid=image_counter,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        depth_params=None,
                        image_path=image_path,
                        image_name=image_name,
                        depth_path="",
                        width=width,
                        height=height,
                        is_test=False,
                        mvt_matrix=mvt_matrix,
                        proj_matrix=proj_matrix,
                        center=center,
                        colormap_id=colormap_id,
                        opacitymap_id=opacitymap_id
                    )
                    cam_infos.append(cam_info)

                    image_counter += 1

    dropout_percentage = 0.995
    mesh_dropout, values_dropout = random_dropout(mesh, values, dropout_percentage)
    mesh_dropout.point_data["value"] = values_dropout.ravel()

    # Save the scaled and translated mesh as input.ply
    storeRawPly(
        os.path.join(path, "input.ply"),
        mesh_dropout,
        values_dropout.ravel(order="F"),
    )
        
    # Close the plotter explicitly
    if 'pl' in locals():
        pl.close()
        del pl
        
    # Delete large arrays to free memory
    if 'values' in locals():
        del values
    if 'values_dropout' in locals():
        del values_dropout
    if 'mesh_dropout' in locals():
        del mesh_dropout
        
    # Force garbage collection
    gc.collect()

    return cam_infos, mesh


def buildVtuDataset(path, colormaps, opacitymaps, num_control_points, resolution):
    # Directory setup
    image_dir = os.path.join(path, "images")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    # Window setup
    width = resolution
    height = resolution
    ratio = width / height
    pv.start_xvfb()
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]
    # print(pv.get_gpu_info())

    # Mesh loading
    mesh = pv.read(os.path.join(path, "data.vtu"))
    print(f"Mesh size: {len(mesh.points)} points")
    array_name = mesh.array_names[0]

    edges = mesh.extract_all_edges()
    edge_lengths = edges.compute_cell_sizes()["Length"]
    non_zero_lengths = edge_lengths[edge_lengths > 1e-10]
    min_edge_length = np.min(non_zero_lengths)

    print(f"Mesh's smallest edge length: {min_edge_length}")

    # Value rescaling
    values = mesh.get_array(array_name).astype(np.float32).reshape(-1, 1)
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.point_data[array_name] = values.ravel().astype(np.float32)

    # Point scaling
    points_min = np.min(mesh.points, axis=0)
    points_max = np.max(mesh.points, axis=0)
    points_max_abs = max(np.max(np.abs(points_min)), np.max(np.abs(points_max)))

    if points_max_abs > 1:
        scale_factor = -1.0 / points_max_abs
        mesh.scale(scale_factor, inplace=True)

    # Get the focal point so that we can translate the mesh to the origin
    offset = list(pl.camera.focal_point)
    # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.translate(offset, inplace=True)

    cam_infos = []
    image_counter = 0

    for opacitymap_id, opacitymap in enumerate(opacitymaps):
        for colormap_id, colormap in enumerate(colormaps):
            cmap = plt.cm.get_cmap(colormap, num_control_points)
            pl.clear()
            pl.add_volume(
                mesh,
                show_scalar_bar=False,
                scalars=array_name,
                cmap=cmap,
                opacity=opacitymap * 255,
                blending="composite",
                shade=False,
                diffuse=0.0,
                specular=0.0,
                ambient=1.0,
                opacity_unit_distance=min_edge_length,
            )

            # Reset the camera position and focal point, since we translated the mesh
            pl.view_xy()
            pl.background_color = "black"
            pl.camera.clipping_range = (0.001, 1000.0)
            camera = pl.camera

            # Controls the camera orbit and capture frequency
            print(
                f"Generating images using {colormap} colormap, opacitymap {opacitymap_id} and resolution {resolution}x{resolution}"
            )
            azimuth_steps = 36
            elevation_steps = 7
            azimuth_range = range(0, 360, 360 // azimuth_steps)
            # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
            elevation_range = range(-35, 35, 70 // elevation_steps)

            for elevation in elevation_range:
                for azimuth in azimuth_range:
                    # Set new azimuth and elevation
                    camera.elevation = elevation
                    camera.azimuth = azimuth

                    # Produce a new render at the new camera position
                    pl.render()

                    # Save the render as a new image
                    image_name = f"{colormap}_omap{opacitymap_id}_{image_counter:05d}.png"
                    image_path = os.path.join(image_dir, image_name)
                    pl.screenshot(image_path)

                    mvt_matrix = np.linalg.inv(
                        arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
                    )

                    # Not sure why this is necessary
                    mvt_matrix[:3, 1:3] *= -1

                    R = mvt_matrix[:3, :3].T
                    T = mvt_matrix[:3, 3]

                    FovY = np.radians(camera.view_angle)
                    FovX = focal2fov(fov2focal(FovY, height), width)

                    proj_matrix = arrayFromVTKMatrix(
                        camera.GetCompositeProjectionTransformMatrix(ratio, 0.001, 1000.0)
                    )

                    # Not sure why this is necessary
                    proj_matrix[1, :] = -proj_matrix[1, :]
                    proj_matrix[2, :] = -proj_matrix[2, :]

                    # Not sure why this is necessary
                    y = camera.position[1]
                    if y < 0:
                        mvt_matrix[2, 1] *= -1
                    mvt_matrix[2, 3] = abs(mvt_matrix[2, 3])

                    center = mvt_matrix[:3, 3]

                    cam_info = CameraInfo(
                        uid=image_counter,
                        R=R,
                        T=T,
                        FovY=FovY,
                        FovX=FovX,
                        depth_params=None,
                        image_path=image_path,
                        image_name=image_name,
                        depth_path="",
                        width=width,
                        height=height,
                        is_test=False,
                        mvt_matrix=mvt_matrix,
                        proj_matrix=proj_matrix,
                        center=center,
                        colormap_id=colormap_id,
                        opacitymap_id=opacitymap_id
                    )
                    cam_infos.append(cam_info)

                    image_counter += 1

    pl.close()

    # mesh_dropout, values_dropout = density_based_dropout(
    #     mesh, values, high_density_dropout=0.65, low_density_dropout=0.35
    # )
    mesh_dropout, values_dropout = random_dropout(mesh, values, 0.99)
    mesh_dropout.point_data[array_name] = values_dropout.ravel()

    # Save the scaled and translated mesh as input.ply
    storePly(
        os.path.join(path, "input.ply"),
        mesh_dropout.points,
        values_dropout,
    )

    return cam_infos, mesh_dropout


def getDirectppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        center = cam.center
        cam_centers.append(np.array([[center[0]], [center[1]], [center[2]]]))

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readRawSceneInfo(
    path, filename, colormaps, opacitymaps, num_control_points, resolution, spacing, eval, train_values, llffhold=8
):
    cam_infos, mesh = buildRawDataset(
        path, filename, colormaps, opacitymaps, num_control_points, resolution, spacing
    )

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # TODO: verify that this actually sets is_test to True
        test_cam_infos = [
            c._replace(is_test=True)
            for idx, c in enumerate(cam_infos)
            if idx % llffhold == 0
        ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    normalization = getDirectppNorm(train_cam_infos)

    ply_path = os.path.join(path, "input.ply")
    pcd = fetchPly(ply_path)

    print(mesh.points.shape)
    min_x, max_x = mesh.points[:, 0].min(), mesh.points[:, 0].max()
    min_y, max_y = mesh.points[:, 1].min(), mesh.points[:, 1].max()
    min_z, max_z = mesh.points[:, 2].min(), mesh.points[:, 2].max()
    bounding_box = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

    if train_values:
        del mesh
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=normalization,
            ply_path=ply_path,
            bounding_box=bounding_box
        )
    else:
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=normalization,
            ply_path=ply_path,
            bounding_box=bounding_box,
            mesh=mesh,
        )
    return scene_info


def readVtuSceneInfo(path, colormaps, opacitymaps, num_control_points, resolution, eval, train_values, llffhold=8):
    cam_infos, mesh = buildVtuDataset(path, colormaps, opacitymaps, num_control_points, resolution)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # TODO: verify that this actually sets is_test to True
        test_cam_infos = [
            c._replace(is_test=True)
            for idx, c in enumerate(cam_infos)
            if idx % llffhold == 0
        ]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    normalization = getDirectppNorm(train_cam_infos)

    ply_path = os.path.join(path, "input.ply")
    pcd = fetchPly(ply_path)

    min_x, max_x = mesh.points[:, 0].min(), mesh.points[:, 0].max()
    min_y, max_y = mesh.points[:, 1].min(), mesh.points[:, 1].max()
    min_z, max_z = mesh.points[:, 2].min(), mesh.points[:, 2].max()
    bounding_box = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]

    if train_values:
        del mesh
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=normalization,
            ply_path=ply_path,
            bounding_box=bounding_box
        )
    else:
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=normalization,
            ply_path=ply_path,
            bounding_box=bounding_box,
            mesh=mesh,
        )
    return scene_info
