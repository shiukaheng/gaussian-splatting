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
import random
import json
from typing import Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from split_gaussian_splatting.training_task import ProjectParams
    from split_gaussian_splatting.training_task import SimpleTrainerParams
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import SceneInfo, sceneLoadTypeCallbacks
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel

class Scene:

    def __init__(self, args : 'ProjectParams', shuffle=True, resolution_scales=[1.0], train_cam_limit = None, on_load_progress: Callable[[int, int], None] = None):

        self.args = args # The arguments passed to the program
        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_info: SceneInfo = self.parse_source(args)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        self.load_images(args, resolution_scales, train_cam_limit, self.scene_info, on_load_progress)

    def parse_source(self, args: 'ProjectParams'):
        if os.path.exists(os.path.join(args.source_path, "sparse")): # Presumed to be a COLMAP scene
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # Presumed to be a Blender scene
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        return scene_info

    def load_images(self, args: 'ProjectParams', resolution_scales, train_cam_limit, scene_info: SceneInfo, on_load_progress: Callable[[int, int], None] = None):
        total = len(resolution_scales) * (len(scene_info.train_cameras) + len(scene_info.test_cameras))
        loaded = 0
        for resolution_scale in resolution_scales:
            print(f"Loading {len(scene_info.train_cameras)} Train Cameras at scale {resolution_scale}")
            def increment():
                nonlocal loaded
                loaded += 1
                if on_load_progress:
                    on_load_progress(loaded, total)
            if train_cam_limit:
                print(f"Limiting train cameras to {train_cam_limit}")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras[:train_cam_limit], resolution_scale, args, on_load=increment)
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, on_load=increment)
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

    def initialize_camera_json(self, scene_info: SceneInfo):
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read()) # Copy the original ply file to a file called input.ply in the model path
            # Create two camera lists
        json_cams = []
        camlist = []
        if scene_info.test_cameras: # If we have test cameras, add them to the list
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras: # If we have train cameras, add them to the list
            camlist.extend(scene_info.train_cameras)
            # We now convert the cameras to JSON form, so we can save them to a file
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

    def create_gaussians(self, training_args: 'SimpleTrainerParams'):
        gaussian_model = GaussianModel(sh_degree=training_args.sh_degree)
        gaussian_model.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)
        gaussian_model.training_setup(training_args)
        return gaussian_model

    # Get the training cameras for a given scale
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # Get the test cameras for a given scale
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]