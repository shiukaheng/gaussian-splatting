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

from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import streamlit as st

class Scene:

    gaussians : GaussianModel
    def __init__(self, args : ModelParams, shuffle=True, resolution_scales=[1.0], train_cam_limit = None):

        self.model_path = args.model_path 
        self.loaded_iter = None

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = self.load_scene_info(args)

        # If we want to shuffle cameras
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # Get the radius of the scene and the translation to the center
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # For each resolution scale, load the cameras
        for resolution_scale in resolution_scales:
            print(f"Loading {len(scene_info.train_cameras)} Train Cameras at scale {resolution_scale}")
            if train_cam_limit:
                print(f"Limiting train cameras to {train_cam_limit}")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras[:train_cam_limit], resolution_scale, args)
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

    def load_scene_info(self, args):
        if os.path.exists(os.path.join(args.source_path, "sparse")): # Presumed to be a COLMAP scene
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # Presumed to be a Blender scene
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self.scene_info = scene_info
        return scene_info

    # Get the training cameras for a given scale
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # Get the test cameras for a given scale
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]