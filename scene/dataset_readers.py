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


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
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


def readVTU(path):
    # TODO: if the input.ply already exists, just load it directly
    mesh = pv.read(os.path.join(path, "data.vtu"))

    # Rescale the values to the range [0, 1]
    values = mesh.get_array("value").reshape(-1, 1)
    values_min = values.min()
    values_max = values.max()
    values = (values - values_min) / (values_max - values_min)
    mesh.get_array("value")[:] = values.ravel()

    # Scale mesh to the unit cube
    global_min = mesh.points.min()
    global_max = mesh.points.max()
    mesh.points[:] = (mesh.points - global_min) / (global_max - global_min)

    # Save the scaled mesh as input.ply
    storePly(
        os.path.join(path, "input.ply"),
        mesh.points,
        values,
    )

    return mesh


def readDirectSceneInfo(path):
    mesh = readVTU(path)

    ply_path = os.path.join(path, "input.ply")
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        ply_path=ply_path,
        mesh=mesh,
    )
    return scene_info
