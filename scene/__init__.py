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

import json
import os
import random

from arguments import ModelParams
from scene.dataset_readers import readRawSceneInfo, readVtuSceneInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        opacity_tables,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        train_values = True,
        skip_train=False
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        opacitymaps = [o.cpu().numpy() for o in opacity_tables]

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        raw_files = [f for f in os.listdir(args.source_path) if f.endswith(".raw")]
        if os.path.exists(os.path.join(args.source_path, "data.vtu")) or os.path.exists(
            os.path.join(args.source_path, "data.vtui")
        ):
            scene_info = readVtuSceneInfo(
                args.source_path,
                args.colormaps,
                opacitymaps,
                args.num_control_points,
                args.resolution,
                args.eval,
                train_values,
                args.dropout,
                skip_train
            )
        elif len(raw_files) == 1:
            scene_info = readRawSceneInfo(
                args.source_path,
                raw_files[0],
                args.colormaps,
                opacitymaps,
                args.num_control_points,
                args.resolution,
                args.spacing,
                args.eval,
                train_values,
                args.dropout,
                skip_train
            )
        else:
            raise FileNotFoundError(
                "Could not recognize scene type! Ensure either a preprocessed dataset or raw data is available."
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file:
                with open(
                    os.path.join(self.model_path, "input.ply"), "wb"
                ) as dest_file:
                    dest_file.write(src_file.read())
                    json_cams = []
                    camlist = []
                    if scene_info.test_cameras:
                        camlist.extend(scene_info.test_cameras)
                    if scene_info.train_cameras:
                        camlist.extend(scene_info.train_cameras)
                    for id, cam in enumerate(camlist):
                        json_cams.append(camera_to_JSON(id, cam))
                    with open(
                        os.path.join(self.model_path, "cameras.json"), "w"
                    ) as file:
                        json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, False
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args, True
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                ),
                args.train_test_exp,
            )
        else:
            if not train_values:
                self.gaussians.create_from_pcd(
                    scene_info.point_cloud,
                    scene_info.train_cameras,
                    self.cameras_extent,
                    scene_info.bounding_box,
                    scene_info.mesh,
                )
            else:
                self.gaussians.create_from_pcd(
                    scene_info.point_cloud,
                    scene_info.train_cameras,
                    self.cameras_extent,
                    scene_info.bounding_box
                )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name)
            .detach()
            .cpu()
            .numpy()
            .tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
