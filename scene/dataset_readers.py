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
from tracemalloc import start
from turtle import st
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

from numba import njit
import time
import vtk


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
    narray = np.eye(matrixSize, dtype=np.float32)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray


@njit
def random_dropout_percentage_numba(points, values, dropout_percentage):
    keep_percentage = 1 - dropout_percentage
    num_points = points.shape[0]
    mask = np.random.random(num_points) < keep_percentage
    indices = np.where(mask)[0]
    return points[indices], values[indices]


@njit
def random_dropout_exact_numba(points, values, selected_indices):
    new_points = np.take(points, selected_indices, axis=0)
    new_values = np.take(values, selected_indices, axis=0)
    return new_points, new_values


def random_dropout_percentage(mesh, dropout_percentage):
    num_points = mesh.n_points

    # Create a random boolean mask directly (more memory efficient)
    keep_percentage = 1 - dropout_percentage
    mask = np.random.random(num_points) < keep_percentage

    # Apply the mask to the points and values
    new_points = mesh.points[mask]
    new_values = mesh.point_data["value"][mask]

    return new_points, new_values


def cell_centered_random_dropout_percentage(mesh, dropout_percentage):
    cell_data = mesh.point_data_to_cell_data()
    cell_centers = mesh.cell_centers().points

    num_points = mesh.n_cells

    # Create a random boolean mask directly (more memory efficient)
    keep_percentage = 1 - dropout_percentage
    mask = np.random.random(num_points) < keep_percentage

    # Apply the mask to the points and values
    new_points = cell_centers[mask]
    new_values = cell_data["value"][mask]

    return new_points, new_values


def random_dropout_exact(mesh, num_particles_to_keep):
    num_points = mesh.n_points

    if num_particles_to_keep > num_points:
        num_particles_to_keep = num_points

    # Randomly select indices without replacement
    selected_indices = np.random.choice(
        num_points, size=num_particles_to_keep, replace=False
    )

    # Extract selected points and associated values
    new_points = mesh.points[selected_indices]
    new_values = mesh.point_data["value"][selected_indices]

    return new_points, new_values


def true_random_exact(num_points):
    bounds = [
        (np.float32(-1.0), np.float32(0.0)),  # x bounds
        (np.float32(-1.0), np.float32(0.0)),  # y bounds
        (np.float32(2.0), np.float32(3.0)),  # z bounds
    ]

    # Initialize array for points
    new_points = np.zeros((num_points, 3), dtype=np.float32)

    # Generate random points within the specified bounds
    for i in range(3):
        lower, upper = bounds[i]
        new_points[:, i] = np.random.uniform(lower, upper, num_points)
    new_values = np.random.random(num_points)

    return new_points, new_values


def cell_centered_random_dropout_exact(mesh, num_particles_to_keep):
    num_points = mesh.n_cells

    if num_particles_to_keep > num_points:
        raise ValueError("num_particles_to_keep cannot exceed total number of points.")

    cell_data = mesh.point_data_to_cell_data()
    cell_centers = mesh.cell_centers().points

    # Randomly select indices without replacement
    selected_indices = np.random.choice(
        num_points, size=num_particles_to_keep, replace=False
    )

    # Extract selected points and associated values
    new_points = cell_centers[selected_indices]
    new_values = cell_data["value"][selected_indices]

    return new_points, new_values


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


@njit
def is_image_too_dark_numba(image, threshold=3):
    return np.max(image) < threshold


@njit
def generate_selected_grid_points(dimensions, spacing, origin, indices):
    nx, ny, nz = dimensions
    n = indices.shape[0]
    points = np.empty((n, 3), dtype=np.float32)

    for i in range(n):
        idx = indices[i]
        ix = idx % nx
        iy = (idx // nx) % ny
        iz = idx // (nx * ny)

        x = ix * spacing[0] + origin[0]
        y = iy * spacing[1] + origin[1]
        z = iz * spacing[2] + origin[2]

        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z

    return points


def dropout_points_and_values(values, dimensions, spacing, origin, dropout):
    total_points = values.shape[0]

    if isinstance(dropout, float) and 0.0 <= dropout <= 1.0:
        keep_count = int(total_points * (1 - dropout))
        print(f"Performing dropout of {dropout * 100:.1f}% ({keep_count} kept)")
    elif isinstance(dropout, int) and dropout > 0:
        keep_count = min(dropout, total_points)
        if keep_count < dropout:
            print(
                f"Requested {dropout} points, but only {total_points} available — using all."
            )
        else:
            print(f"Performing exact dropout of {dropout} points")
    else:
        print("Invalid or zero dropout — using all points.")
        keep_count = total_points

    if keep_count == total_points:
        selected_indices = np.arange(total_points)
    else:
        selected_indices = np.random.choice(total_points, keep_count, replace=False)
        selected_indices.sort()

    values_dropout = values[selected_indices]

    # Avoid division by zero
    range_val = values_dropout.max() - values_dropout.min()
    if range_val > 1e-8:
        values_dropout = (values_dropout - values_dropout.min()) / range_val
    else:
        values_dropout[:] = 0.0

    points_dropout = generate_selected_grid_points(
        dimensions, spacing, origin, selected_indices
    )

    return points_dropout, values_dropout


def buildRawDataset(
    path,
    filename,
    colormaps,
    opacitymaps,
    num_control_points,
    resolution,
    spacing,
    dropout,
    skip_train,
):
    start_time = time.time()

    # Directory setup
    image_dir = os.path.join(path, "images")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    # Window setup
    width = resolution
    height = resolution
    ratio = width / height
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]

    # Parse the filename
    base_name = filename.rsplit(".", 1)[0]
    parts = base_name.split("_")
    dimensions = tuple(map(int, parts[-2].split("x")))

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
    data_type = dtype_map[parts[-1]]

    # Load the raw data
    values = np.fromfile(os.path.join(path, filename), dtype=data_type)

    # Ensure the size matches the dimensions
    if values.size != dimensions[0] * dimensions[1] * dimensions[2]:
        raise ValueError("Data size does not match the specified dimensions.")

    mesh = pv.ImageData(dimensions=dimensions, spacing=spacing)
    mesh.point_data["value"] = values

    # Point scaling
    points_min = np.array(mesh.origin)
    points_max = points_min + (np.array(mesh.dimensions) - 1) * np.array(mesh.spacing)
    extents = points_max - points_min
    max_extent = np.max(extents)
    scale_factor = 1.0 / max_extent
    mesh.spacing = tuple(np.array(mesh.spacing) * scale_factor)

    # Get the focal point so that we can translate the mesh to the origin
    offset = list(pl.camera.focal_point)

    # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.origin = offset
    final_spacing = mesh.spacing
    final_origin = mesh.origin

    print(f"Time taken to load the dataset: {time.time() - start_time:.2f} seconds")
    print(f"Mesh memory: {mesh.actual_memory_size * 1024} bytes")

    cam_infos = []
    image_counter = 0
    skip_counter = 0

    colormap_cache = {
        name: plt.cm.get_cmap(name, num_control_points) for name in colormaps
    }

    pl.background_color = "black"
    camera = pl.camera
    camera.clipping_range = (0.001, 1000.0)
    opacity_unit_distance = 1.0 / 128.0

    # Controls the camera orbit and capture frequency
    azimuth_steps = 18
    elevation_steps = 7
    azimuth_range = np.linspace(0, 360, azimuth_steps, endpoint=False)
    # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
    elevation_range = np.linspace(-35, 35, elevation_steps, endpoint=True)

    for opacitymap_id, opacitymap in enumerate(opacitymaps):
        for colormap_id, colormap in enumerate(colormaps):
            start_time = time.time()
            print("Updating the volume")
            cmap = colormap_cache[colormap]

            pl.add_volume(
                mesh,
                name="volume_actor",
                show_scalar_bar=False,
                scalars="value",
                cmap=cmap,
                opacity=opacitymap * 255,
                blending="composite",
                shade=False,
                diffuse=0.0,
                specular=0.0,
                specular_power=0.0,
                ambient=1.0,
                culling=True,
                pickable=False,
                render=False,
                opacity_unit_distance=opacity_unit_distance,
            )
            pl.view_xy(render=False)
            print(
                f"Time taken to update the volume: {time.time() - start_time:.2f} seconds"
            )

            print(
                f"Generating images using {colormap} colormap, opacitymap {opacitymap_id}, {resolution}x{resolution} resolution, and {spacing} spacing"
            )

            cam_count = 0
            start_time = time.time()
            for elevation in elevation_range:
                for azimuth in azimuth_range:
                    if skip_train and cam_count % 8 != 0:
                        cam_count += 1
                        continue
                    cam_count += 1

                    # Set new azimuth and elevation
                    camera.elevation = elevation
                    camera.azimuth = azimuth

                    # Produce a new render at the new camera position
                    pl.render()

                    img = pl.screenshot(None, return_img=True)

                    if is_image_too_dark_numba(img):
                        skip_counter += 1
                        continue

                    # Save the render as a new image
                    image_name = (
                        f"{colormap}_omap{opacitymap_id}_{image_counter:05d}.png"
                    )
                    image_path = os.path.join(image_dir, image_name)
                    plt.imsave(image_path, img)

                    # Convert 4x4 VTK matrix to NumPy and invert
                    mvt_matrix = np.linalg.inv(
                        arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
                    )

                    # Y/Z flip (likely due to coordinate system handedness)
                    mvt_matrix[:3, 1:3] *= -1

                    # Extract rotation and translation
                    R = mvt_matrix[:3, :3].T  # transpose to match camera convention
                    T = mvt_matrix[:3, 3]

                    # FOV conversions
                    FovY = np.radians(camera.view_angle)
                    FovX = focal2fov(fov2focal(FovY, height), width)

                    # Projection matrix conversion and adjustment
                    proj_matrix = arrayFromVTKMatrix(
                        camera.GetCompositeProjectionTransformMatrix(
                            ratio, 0.001, 1000.0
                        )
                    )
                    # Y and Z flip
                    proj_matrix[1:3, :] *= -1

                    # Fix up modelview matrix if camera is flipped
                    if camera.position[1] < 0:
                        mvt_matrix[2, 1] *= -1
                    mvt_matrix[2, 3] = abs(mvt_matrix[2, 3])

                    # Get camera center in world space
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
                        opacitymap_id=opacitymap_id,
                    )
                    cam_infos.append(cam_info)

                    image_counter += 1

            # Force garbage collection
            gc.collect()
            print(
                f"Time taken to generate images: {time.time() - start_time:.2f} seconds"
            )

    pl.close()
    del pl
    print(f"Number of images generated: {image_counter}")
    print(f"Number of images skipped: {skip_counter}")

    if skip_train:
        dropout = 300000

    start_time = time.time()
    points_dropout, values_dropout = dropout_points_and_values(
        values=values,
        dimensions=dimensions,
        spacing=final_spacing,
        origin=final_origin,
        dropout=dropout,
    )
    print(f"Time taken to perform dropout: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    storePly(
        os.path.join(path, "input.ply"),
        points_dropout,
        values_dropout,
    )
    print(f"Time taken to store input.ply: {time.time() - start_time:.2f} seconds")

    return cam_infos, points_dropout, values_dropout


def buildVtuDataset(
    path, colormaps, opacitymaps, num_control_points, resolution, dropout, skip_train
):
    # Directory setup
    image_dir = os.path.join(path, "images")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    # Window setup
    width = resolution
    height = resolution
    ratio = width / height
    # pv.start_xvfb()
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]
    # print(pv.get_gpu_info())

    # Mesh loading
    mesh = pv.read(os.path.join(path, "data.vtu"))
    print(f"Mesh size: {len(mesh.points)} points")
    array_name = mesh.array_names[0]

    # Value rescaling
    values = mesh.get_array(array_name).astype(np.float32).reshape(-1, 1)
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.point_data[array_name] = values.ravel().astype(np.float32)

    # Point scaling
    points_min = np.min(mesh.points, axis=0)
    points_max = np.max(mesh.points, axis=0)
    points_range = points_max - points_min
    points_max_abs = np.max(points_range)

    if points_max_abs > 1:
        scale_factor = 1.0 / points_max_abs
        mesh.scale(scale_factor, inplace=True)

    new_points_min = np.min(mesh.points, axis=0)
    mesh.translate([-new_points_min[0], -new_points_min[1], 0], inplace=True)

    # Get the focal point so that we can translate the mesh to the origin
    offset = list(pl.camera.focal_point)
    # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.translate(offset, inplace=True)

    # # Get the focal point so that we can translate the mesh to the origin
    # offset = list(pl.camera.focal_point)
    # # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    # offset[2] -= 3
    # offset = [-x for x in offset]
    # mesh.translate(offset, inplace=True)
    print(f"Mesh memory: {mesh.actual_memory_size * 1024} bytes")

    edges = mesh.extract_all_edges()
    edge_lengths = edges.compute_cell_sizes()["Length"]
    non_zero_lengths = edge_lengths[edge_lengths > 1e-10]
    min_edge_length = np.min(non_zero_lengths)

    print(f"Mesh's smallest edge length: {min_edge_length}")

    cam_infos = []
    image_counter = 0
    throwaway_counter = 0

    for opacitymap_id, opacitymap in enumerate(opacitymaps):
        for colormap_id, colormap in enumerate(colormaps):
            start_time = time.time()
            cmap = plt.cm.get_cmap(colormap, num_control_points)
            pl.clear()
            volume_actor = pl.add_volume(
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
                opacity_unit_distance=(1.0 / 128.0),
            )
            print(
                f"Time taken to set up volume actor: {time.time() - start_time:.2f} seconds"
            )
            start_time = time.time()
            volume_property = volume_actor.GetProperty()
            opacity_unit_distance = volume_property.GetScalarOpacityUnitDistance(0)
            print(
                f"Time taken to get opacity unit distance: {time.time() - start_time:.2f} seconds"
            )
            print(f"Opacity unit distance: {opacity_unit_distance}")

            # Reset the camera position and focal point, since we translated the mesh
            pl.view_xy()
            pl.background_color = "black"
            pl.camera.clipping_range = (0.001, 1000.0)
            camera = pl.camera

            # Controls the camera orbit and capture frequency
            print(
                f"Generating images using {colormap} colormap, opacitymap {opacitymap_id} and resolution {resolution}x{resolution}"
            )
            azimuth_steps = 18
            elevation_steps = 7
            azimuth_range = range(0, 360, 360 // azimuth_steps)
            # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
            elevation_range = range(-35, 35, 70 // elevation_steps)
            cam_count = 0

            for elevation in elevation_range:
                for azimuth in azimuth_range:
                    if skip_train and cam_count % 8 != 0:
                        cam_count += 1
                        continue
                    cam_count += 1
                    # Set new azimuth and elevation
                    camera.elevation = elevation
                    camera.azimuth = azimuth

                    # Produce a new render at the new camera position
                    pl.render()

                    img = pl.screenshot(None, return_img=True)

                    if is_image_too_dark_numba(img):
                        throwaway_counter += 1
                        continue

                    # Save the render as a new image
                    image_name = (
                        f"{colormap}_omap{opacitymap_id}_{image_counter:05d}.png"
                    )
                    image_path = os.path.join(image_dir, image_name)
                    plt.imsave(image_path, img)

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
                        camera.GetCompositeProjectionTransformMatrix(
                            ratio, 0.001, 1000.0
                        )
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
                        opacitymap_id=opacitymap_id,
                    )
                    cam_infos.append(cam_info)

                    image_counter += 1

    pl.close()

    print(f"Number of images generated: {image_counter}")
    print(f"Number of images thrown away due to darkness: {throwaway_counter}")
    if skip_train:
        dropout = 300000
    if dropout is None:
        values_dropout = mesh.point_data_to_cell_data()["value"]
        points_dropout = mesh.cell_centers().points
    elif 0.0 <= dropout <= 1.0:
        points_dropout, values_dropout = random_dropout_percentage(mesh, dropout)
    else:
        points_dropout, values_dropout = random_dropout_exact(mesh, dropout)

    # Save the scaled and translated mesh as input.ply
    storePly(
        os.path.join(path, "input.ply"),
        points_dropout,
        values_dropout,
    )

    return cam_infos, mesh


def getDirectppNorm(cam_info):
    centers = np.array([cam.center for cam in cam_info])
    center = centers.mean(axis=0)
    distances = np.linalg.norm(centers - center, axis=1)
    diagonal = distances.max()
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate, "radius": radius}


@njit
def compute_bounding_box_numba(points):
    n = points.shape[0]
    min_x = points[0, 0]
    min_y = points[0, 1]
    min_z = points[0, 2]
    max_x = points[0, 0]
    max_y = points[0, 1]
    max_z = points[0, 2]

    for i in range(1, n):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if z < min_z:
            min_z = z
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        if z > max_z:
            max_z = z

    bbox = np.empty((3, 2), dtype=points.dtype)
    bbox[0, 0] = min_x
    bbox[0, 1] = max_x
    bbox[1, 0] = min_y
    bbox[1, 1] = max_y
    bbox[2, 0] = min_z
    bbox[2, 1] = max_z
    return bbox


def readRawSceneInfo(
    path,
    filename,
    colormaps,
    opacitymaps,
    num_control_points,
    resolution,
    spacing,
    eval,
    train_values,
    dropout,
    skip_train=False,
    llffhold=8,
):
    cam_infos, pts, vals = buildRawDataset(
        path,
        filename,
        colormaps,
        opacitymaps,
        num_control_points,
        resolution,
        spacing,
        dropout,
        skip_train,
    )

    if skip_train:
        train_cam_infos = []
        test_cam_infos = [c._replace(is_test=True) for c in cam_infos]
    else:
        if eval:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            # TODO: verify that this actually sets is_test to True
            test_cam_infos = [
                c._replace(is_test=True)
                for idx, c in enumerate(cam_infos)
                if idx % llffhold == 0
            ]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    start_time = time.time()
    if skip_train:
        normalization = getDirectppNorm(test_cam_infos)
    else:
        normalization = getDirectppNorm(train_cam_infos)
    print(f"Time taken to get normalization: {time.time() - start_time:.2f} seconds")

    ply_path = os.path.join(path, "input.ply")
    pcd = BasicPointCloud(points=pts, values=vals)

    start_time = time.time()
    bbox = compute_bounding_box_numba(pts)
    bounding_box = [(bbox[i, 0], bbox[i, 1]) for i in range(bbox.shape[0])]
    print(f"Time taken to compute bounding box: {time.time() - start_time:.2f} seconds")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=normalization,
        ply_path=ply_path,
        bounding_box=bounding_box,
    )
    return scene_info


def readVtuSceneInfo(
    path,
    colormaps,
    opacitymaps,
    num_control_points,
    resolution,
    eval,
    train_values,
    dropout,
    skip_train=False,
    llffhold=8,
):
    cam_infos, mesh = buildVtuDataset(
        path,
        colormaps,
        opacitymaps,
        num_control_points,
        resolution,
        dropout,
        skip_train,
    )

    if skip_train:
        train_cam_infos = []
        test_cam_infos = [c._replace(is_test=True) for c in cam_infos]
    else:
        if eval:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % llffhold != 0
            ]
            # TODO: verify that this actually sets is_test to True
            test_cam_infos = [
                c._replace(is_test=True)
                for idx, c in enumerate(cam_infos)
                if idx % llffhold == 0
            ]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    if skip_train:
        normalization = getDirectppNorm(test_cam_infos)
    else:
        normalization = getDirectppNorm(train_cam_infos)

    ply_path = os.path.join(path, "input.ply")
    pcd = fetchPly(ply_path)

    min_x, max_x = mesh.points[:, 0].min(), mesh.points[:, 0].max()
    min_y, max_y = mesh.points[:, 1].min(), mesh.points[:, 1].max()
    min_z, max_z = mesh.points[:, 2].min(), mesh.points[:, 2].max()
    bounding_box = [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
    print(f"Bounding box: {bounding_box}")

    if train_values:
        del mesh
        scene_info = SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=normalization,
            ply_path=ply_path,
            bounding_box=bounding_box,
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
