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
from typing import Callable
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import SceneInfo, sceneLoadTypeCallbacks
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# The two important classes
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel

class Scene:

    def __init__(self, args : ModelParams, load_iteration=None, shuffle=True, resolution_scales=[1.0], train_cam_limit = None, on_load_progress: Callable[[int, int], None] = None):

        self.args = args # The arguments passed to the program
        self.model_path = args.model_path # The path to the model directory
        self.loaded_iter = None # The iteration we loaded from (or None if we did not load from a previous iteration)

        # Search for newest saved iteration, if requested (-1)
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_info: SceneInfo = self.parse_source(args)

        if not self.loaded_iter:
            self.initialize_camera_json(self.scene_info)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        # Get the radius of the scene and the translation to the center
        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        self.load_images(args, resolution_scales, train_cam_limit, self.scene_info, on_load_progress)

    def parse_source(self, args: 'Task'):
        if os.path.exists(os.path.join(args.source_path, "sparse")): # Presumed to be a COLMAP scene
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # Presumed to be a Blender scene
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"
        return scene_info

    def load_images(self, args: 'Task', resolution_scales, train_cam_limit, scene_info: SceneInfo, on_load_progress: Callable[[int, int], None] = None):
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

    def create_gaussians_from_source(self):
        gaussian_model = GaussianModel(self.args.sh_degree)
        gaussian_model.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)
        # gaussian_model.training_setup(self.args)
        return gaussian_model

    def create_gaussians(self):
        gaussian_model = GaussianModel(self.args.sh_degree)
        if self.loaded_iter:
            gaussian_model.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            # gaussian_model.training_setup(self.args)
            return gaussian_model
        else:
            return self.create_gaussians_from_source()

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
    
    # def subsetCameras(self):
    #     # Manipulate self.train_cameras[all scales] (an array of Cameras)
    #     # We should be able to use the translation property to filter the camera
    #     pass

    # def subsetGaussians(self):
    #     # Manipulate self.gaussian
    #     # See GaussianModel.create_from_pcd
    #     # Basically, gaussians are represented by one big tensor. We just have to filter them by attributes (like spatial attributes)
    #     pass

    # def splitScene(self):
    #     # How to do point to camera corespondence? We can estimate from position and camera frustum but its not optimal.
    #     # Check COLMAP loader line 211, seems to log 3D point IDs per camera. That is good news!

    #     # In order to split a new scene from an existing one, copying model_path,  loaded_iter,  gaussians, train_cameras, test_cameras, cameras_extent should be sufficient

    #     # image_extrinsics is a dict of Image classes (key is image id assigned by COLMAP), of which the value is a named tuple
    #     # which has the value point3D_ids

    #     # Gaussians right now dont have point ID.
    #     # Check BasicPointCloud properties and modify to include point ID
    #     # Also, fetchPly will need to be modified to also retrieve the point3D ID (if any! needs investigation)
    #     pass