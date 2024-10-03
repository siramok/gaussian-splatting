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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from matplotlib.colors import LinearSegmentedColormap

import pyvista as pv
import shutil
import matplotlib.pyplot as plt
from vtk import vtkMatrix4x4, vtkMatrix3x3


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


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    depths_params,
    images_folder,
    depths_folder,
    test_cam_names_list,
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split(".")[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = (
            os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png")
            if depths_folder != ""
            else ""
        )

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=depth_params,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=width,
            height=height,
            is_test=image_name in test_cam_names_list,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    values = np.vstack(vertices["value"]).T
    return BasicPointCloud(
        points=positions, normals=normals, values=values
    )


def storePly(path, xyz, values):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("value", "f4"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, values), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array(
                [depths_params[key]["scale"] for key in depths_params]
            )
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(
                f"Error: depth_params.json file not found at path '{depth_params_file}'."
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occurred when trying to open depth_params.json file: {e}"
            )
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [
                name for idx, name in enumerate(cam_names) if idx % llffhold == 0
            ]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, normals=np.zeros((num_pts, 3)), values=np.random.random((num_pts))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


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

    width = 800
    height = 800
    ratio = width / height

    # Prepare pyvista window
    pv.start_xvfb()
    pl = pv.Plotter(off_screen=True)
    pl.window_size = [width, height]

    # TODO: if the input.ply already exists, just load it directly
    mesh = pv.read(os.path.join(path, "data.vtu"))
    values = mesh.get_array("value").reshape(-1, 1)

    # Rescale the values to the range [0, 1]
    min_val = values.min()
    max_val = values.max()
    values = (values - min_val) / (max_val - min_val)
    mesh.get_array("value")[:] = values.ravel()

    # mesh = pv.read(os.path.join(path, "data.ply"))
    min_vals = np.min(mesh.points, axis=0)
    max_vals = np.max(mesh.points, axis=0)
    max_abs_val = max(np.max(np.abs(min_vals)), np.max(np.abs(max_vals)))

    if max_abs_val > 1:
        scale_factor = -1.0 / max_abs_val
        mesh.scale(scale_factor, inplace=True)
    # print(mesh)
    colormap = LinearSegmentedColormap.from_list("CustomColormap", [
        (1.0, 0.0, 0.0),  # Red
        (1.0, 1.0, 0.0),  # Yellow
        (0.0, 1.0, 0.0),  # Green
        (0.0, 1.0, 1.0),  # Cyan
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 0.0, 1.0)   # Pink
    ])

    pl.add_volume(mesh, show_scalar_bar=False, scalars="value", cmap=colormap, opacity=np.ones((256,)) * 30)
    offset = list(pl.camera.focal_point)
    offset[2] -= 3
    offset = [-x for x in offset]
    mesh.translate(offset, inplace=True)

    # Save the scaled and translated .ply
    storePly(
        os.path.join(path, "input.ply"),
        mesh.points,
        # np.random.rand(*values.shape),
        values
    )

    pl.background_color = "black"
    pl.view_xy()
    pl.camera.clipping_range = (0.001, 1000.0)
    camera = pl.camera

    cam_infos = []

    azimuth_steps = 36
    elevation_steps = 7
    azimuth_range = range(0, 360, 360 // azimuth_steps)
    elevation_range = range(-35, 35, 70 // elevation_steps)

    image_counter = 0

    for elevation in elevation_range:
        for azimuth in azimuth_range:
            camera.elevation = elevation
            camera.azimuth = azimuth

            pl.render()
            _, y, _ = camera.position
            # image_name = f"img_{image_counter:05d}_azi_{azimuth}_ele_{elevation}_x_{round(x, 8)}_y_{round(y, 8)}_z_{round(z, 8)}.png"
            image_name = f"img_{image_counter:05d}.png"
            image_path = os.path.join(f"{image_dir}", image_name)
            pl.screenshot(image_path)

            mvt_matrix = np.linalg.inv(
                arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
            )
            mvt_matrix[:3, 1:3] *= -1

            R = mvt_matrix[:3, :3].T
            T = mvt_matrix[:3, 3]

            FovY = np.radians(camera.view_angle)
            FovX = focal2fov(fov2focal(FovY, height), width)

            proj_matrix = arrayFromVTKMatrix(
                camera.GetCompositeProjectionTransformMatrix(ratio, 0.001, 1000.0)
            )

            proj_matrix[1, :] = -proj_matrix[1, :]
            proj_matrix[2, :] = -proj_matrix[2, :]

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
    sys.stdout.write("\n")
    return cam_infos


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
    cam_infos = readDirectCameras(path)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        for test_cam in test_cam_infos:
            test_cam.is_test = True
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
    )
    return scene_info


# def change_color_to_white(img, target_color):
#     img = img.convert("RGBA")
#     data = img.getdata()
#     new_data = []
#     for item in data:
#         # Change all pixels that match the target color to white
#         if item[:3] == target_color[:3]:  # Ignore alpha channel if present
#             new_data.append(
#                 (255, 255, 255, item[3]) if len(item) == 4 else (255, 255, 255)
#             )
#         else:
#             new_data.append(item)
#     img.putdata(new_data)
#     draw = ImageDraw.Draw(img)
#     width, height = img.size
#     # Define the region for the legend (adjust these coordinates as necessary)
#     legend_area = (width - 120, height - 400, width, height)
#     draw.rectangle(legend_area, fill=(255, 255, 255))
#     return img


# def readCinemaCameras(path):
#     csv_path = os.path.join(path, "data.csv")
#     df = pd.read_csv(csv_path)
#     df = df.loc[df["time"] == 0.0]
#     df = df.loc[df["opacity_transfer_function"] == 1]
#     # df = df[((df['theta'] >= 0) & (df['theta'] <= 35)) | ((df['theta'] >= 325) & (df['theta'] <= 360))]
#     # df = df[df['theta'] != 0]
#     df = df[
#         ((df["theta"] >= -35.0) & (df["theta"] <= 35.0))
#         | ((df["theta"] >= 325.0) & (df["theta"] <= 360.0))
#     ]
#     # df = df.loc[df["theta"] == 0.0]
#     # print(df)

#     with Image.open(os.path.join(path, df["FILE_png"].iloc[0])) as img:
#         width, height = img.size
#         bg_color = img.getpixel((0, 0))
#     ratio = width / height
#     pl = pv.Plotter(off_screen=True)
#     pl.window_size = [width, height]
#     mesh = pv.read(os.path.join(path, "data.ply"))
#     pl.add_mesh(mesh, show_scalar_bar=False, cmap=plt.cm.coolwarm_r)
#     pl.view_xy()
#     camera = pl.camera
#     # camera.position = [0.0, 0.0, 6.346065214951231]
#     # camera.focal_point = [0.0, 0.0, 5.0]

#     FovY = np.radians(camera.view_angle)
#     FovX = focal2fov(fov2focal(FovY, height), width)

#     cx = 0
#     cy = 0

#     cam_infos = []
#     image_counter = 0

#     for _, row in df.iterrows():
#         camera.elevation = row["theta"]
#         camera.azimuth = row["phi"]
#         y = camera.position[1]

#         image_name = row["FILE_png"]
#         image_path = os.path.join(path, image_name)
#         image = Image.open(image_path)
#         if bg_color != (255, 255, 255):
#             image = change_color_to_white(image, bg_color)

#         mvt_matrix = np.linalg.inv(
#             arrayFromVTKMatrix(camera.GetModelViewTransformMatrix())
#         )
#         mvt_matrix[:3, 1:3] *= -1

#         R = mvt_matrix[:3, :3].T
#         T = mvt_matrix[:3, 3]

#         proj_matrix = arrayFromVTKMatrix(
#             camera.GetCompositeProjectionTransformMatrix(ratio, 0.001, 1000.0)
#         )

#         proj_matrix[1, :] = -proj_matrix[1, :]
#         proj_matrix[2, :] = -proj_matrix[2, :]

#         if y < 0:
#             mvt_matrix[2, 1] *= -1
#         mvt_matrix[2, 3] = abs(mvt_matrix[2, 3])

#         center = mvt_matrix[:3, 3]

#         cam_info = CameraInfo(
#             uid=image_counter,
#             R=R,
#             T=T,
#             FovY=FovY,
#             FovX=FovX,
#             cx=cx,
#             cy=cy,
#             image=image,
#             image_path=image_path,
#             image_name=image_name,
#             width=width,
#             height=height,
#             mvt_matrix=mvt_matrix,
#             proj_matrix=proj_matrix,
#             center=center,
#         )
#         cam_infos.append(cam_info)

#         image_counter += 1

#     return cam_infos


# def readCinemaSceneInfo(path, eval, llffhold=8):
#     ply_path = os.path.join(path, "data.ply")
#     num_pts = 500_000
#     xyz = np.random.random((num_pts, 3)) * 2 - 1
#     xyz[:, 2] -= 4
#     shs = np.random.random((num_pts, 3)) / 255.0
#     pcd = BasicPointCloud(
#         points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
#     )
#     storePly(ply_path, xyz, SH2RGB(shs) * 255)

#     cam_infos = readCinemaCameras(path)

#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []

#     nerf_normalization = getVolumeppNorm(train_cam_infos)

#     scene_info = SceneInfo(
#         point_cloud=pcd,
#         train_cameras=train_cam_infos,
#         test_cameras=test_cam_infos,
#         nerf_normalization=nerf_normalization,
#         ply_path=ply_path,
#     )
#     return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Direct": readDirectSceneInfo,
}
