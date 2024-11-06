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
import shutil
from typing import NamedTuple

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from plyfile import PlyData, PlyElement
from vtk import vtkMatrix3x3, vtkMatrix4x4

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


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    mesh: pv.PolyData


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    values = np.vstack(vertices["value"]).T
    return BasicPointCloud(points=positions, values=values)


def storePly(path, xyz, values):
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


def readDirectCameras(path):
    # TODO: if images already exist, don't regenerate them. Requires figuring out how to save and load the camera info
    image_dir = os.path.join(path, "images")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    # TODO: make the width and height configurable?
    # Higher resolution images train a better model, but takes longer
    width = 1600
    height = 900
    ratio = width / height

    # This line makes headless rendering work
    pv.start_xvfb()

    # Initialize the pyvista window
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]

    # TODO: if the input.ply already exists, just load it directly
    mesh = pv.read(os.path.join(path, "data.vtu"))

    # Rescale the values to the range [0, 1]
    values = mesh.get_array("value").reshape(-1, 1)
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.get_array("value")[:] = values.ravel()

    # Scale mesh to the unit cube
    points_min = np.min(mesh.points, axis=0)
    points_max = np.max(mesh.points, axis=0)
    points_max_abs = max(np.max(np.abs(points_min)), np.max(np.abs(points_max)))
    if points_max_abs > 1:
        scale_factor = -1.0 / points_max_abs
        mesh.scale(scale_factor, inplace=True)

    colormap = LinearSegmentedColormap.from_list(
        "CustomColormap",
        [
            (1.0, 0.0, 0.0),  # Red
            (1.0, 1.0, 0.0),  # Yellow
            (0.0, 1.0, 0.0),  # Green
            (0.0, 1.0, 1.0),  # Cyan
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 0.0, 1.0),  # Pink
        ],
    )

    pl.add_volume(
        mesh,
        show_scalar_bar=False,
        scalars="value",
        cmap=colormap,
        opacity=0.5,
    )

    # Get the focal point so that we can translate the mesh to the origin
    offset = list(pl.camera.focal_point)
    # However, the renderer has a bug(s) if the the camera's z-position is too close to 0, this works around it
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.translate(offset, inplace=True)

    # Save the scaled and translated mesh as input.ply
    storePly(
        os.path.join(path, "input.ply"),
        mesh.points,
        values,
    )

    # Reset the camera position and focal point, since we translated the mesh
    pl.view_xy()
    pl.background_color = "black"
    pl.camera.clipping_range = (0.001, 1000.0)
    camera = pl.camera

    # Controls the camera orbit and capture frequency
    azimuth_steps = 36
    elevation_steps = 7
    azimuth_range = range(0, 360, 360 // azimuth_steps)
    # elevation is intentionally limited to avoid a render bug(s) that occurs when elevation is outside of [-35, 35]
    elevation_range = range(-35, 35, 70 // elevation_steps)

    cam_infos = []
    image_counter = 0
    for elevation in elevation_range:
        for azimuth in azimuth_range:
            # Set new azimuth and elevation
            camera.elevation = elevation
            camera.azimuth = azimuth

            # Produce a new render at the new camera position
            pl.render()

            # Save the render as a new image
            image_name = f"img_{image_counter:05d}.png"
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
            )
            cam_infos.append(cam_info)

            image_counter += 1

    pl.close()
    return cam_infos, mesh


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


def readDirectSceneInfo(path, eval, llffhold=8):
    cam_infos, mesh = readDirectCameras(path)

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

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=normalization,
        ply_path=ply_path,
        mesh=mesh,
    )
    return scene_info
