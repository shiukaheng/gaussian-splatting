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

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import streamlit as st

class Scene:

    gaussians : GaussianModel
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], train_cam_limit = None):

        self.model_path = args.model_path 
        self.loaded_iter = None
        self.gaussians = gaussians

        # If we want to load a previous iteration, we get the request iteration or the max
        # TODO: This should be in a separate method
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading ltrained model at iteration {}".format(self.loaded_iter))

        # Create a dictionary of cameras for each resolution scale
        # TODO: This should be stored in a separate Cameras class, especially when now we are considering dynamically changing cameras and gaussians independently
        self.train_cameras = {}
        self.test_cameras = {}

        # Load in different forms of scene_info
        # TODO: Better document what scene_info is and figure out where it should be stored
        if os.path.exists(os.path.join(args.source_path, "sparse")): # Presumed to be a COLMAP scene
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # Presumed to be a Blender scene
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # If we do not have a loaded iteration
        # TODO: We should really pre-convert it to our internal representation first so we don't have to write two versions of loading logic!
        if not self.loaded_iter:
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

        # If we want to shuffle cameras
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # Get the radius of the scene and the translation to the center
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # For each resolution scale, load the cameras
        # TODO: This should be a method in a Cameras class
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

        # If we have a loaded iteration, we load the already converted gaussians from PLY
        # TODO: Again, rewrite to not have to write two versions of loading logic, instead pre-convert to our internal representation
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # Otherwise, we create the gaussians from the point cloud
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    # Save the gaussians to a ply file
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # Get the training cameras for a given scale
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # Get the test cameras for a given scale
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]