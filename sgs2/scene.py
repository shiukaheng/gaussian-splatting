from dataclasses import dataclass
import json
import os
import random
from typing import Callable
from scene.dataset_readers import SceneInfo, sceneLoadTypeCallbacks
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos

@dataclass
class Scene:
    source_path: str
    images: str = "images"
    resolution: int = -1
    data_device: str = "cpu" # or "cuda"
    shuffle: bool = True
    resolution_scales=[1.0]
    train_cam_limit = None
    on_load_progress: Callable[[int, int], None] = None
    eval: bool = False

    def __post_init__(self):

        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_info: SceneInfo = self.parse_source()

        if self.shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        self.load_images()

    def parse_source(self):
        if os.path.exists(os.path.join(self.source_path, "sparse")): # Presumed to be a COLMAP scene
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.source_path, self.images, self.eval)
        elif os.path.exists(os.path.join(self.source_path, "transforms_train.json")): # Presumed to be a Blender scene
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](self.source_path, self.white_background, self.eval)
        else:
            assert False, "Could not recognize scene type!"
        return scene_info

    def load_images(self):
        total = len(self.resolution_scales) * (len(self.scene_info.train_cameras) + len(self.scene_info.test_cameras))
        loaded = 0
        for resolution_scale in self.resolution_scales:
            print(f"Loading {len(self.scene_info.train_cameras)} Train Cameras at scale {resolution_scale}")
            def increment():
                nonlocal loaded
                loaded += 1
                if self.on_load_progress:
                    self.on_load_progress(loaded, total)
            if self.train_cam_limit:
                print(f"Limiting train cameras to {self.train_cam_limit}")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras[:self.train_cam_limit], resolution_scale, self, on_load=increment)
            else:
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, self, on_load=increment)
            # self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, self)

    def initialize_camera_json(self):
        if not self.scene_info.ply_path:
            raise Exception("No ply file found!")
        with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read()) # Copy the original ply file to a file called input.ply in the model path
            # Create two camera lists
        json_cams = []
        camlist = []
        if self.scene_info.test_cameras: # If we have test cameras, add them to the list
            camlist.extend(self.scene_info.test_cameras)
        if self.scene_info.train_cameras: # If we have train cameras, add them to the list
            camlist.extend(self.scene_info.train_cameras)
            # We now convert the cameras to JSON form, so we can save them to a file
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

    # Get the training cameras for a given scale
    def get_train_cameras(self, scale=1.0):
        return self.train_cameras[scale]

    # Get the test cameras for a given scale
    def get_test_cameras(self, scale=1.0):
        return self.test_cameras[scale]