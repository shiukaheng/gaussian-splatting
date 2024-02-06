from dataclasses import dataclass
import json
import os
import random
from typing import Callable, List

from tqdm import tqdm
from scene.cameras import Camera
from scene.dataset_readers import CameraInfo, SceneInfo, sceneLoadTypeCallbacks
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
import networkx as nx

@dataclass
class Scene:
    source_path: str
    images: str = "images"
    resolution: int = -1
    data_device: str = "cpu" # or "cuda"
    shuffle: bool = False
    resolution_scales=[1.0]
    on_load_progress: Callable[[int, int], None] = None
    eval: bool = False
    camera_name_whitelist: List[str] = None
    camera_name_blacklist: List[str] = None
    white_background: bool = False

    def __post_init__(self):

        self.train_cameras = {}
        self.test_cameras = {}

        self.scene_info: SceneInfo = self.parse_source()

        if self.shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        self.update_filtered_cameras()
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
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.filtered_cameras, resolution_scale, self, on_load=increment)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, self)

    def update_filtered_cameras(self):
        if self.camera_name_whitelist:
            self.filtered_cameras = [x for x in self.scene_info.train_cameras if x.image_name in self.camera_name_whitelist]
        elif self.camera_name_blacklist:
            self.filtered_cameras = [x for x in self.scene_info.train_cameras if x.image_name not in self.camera_name_blacklist]
        else:
            self.filtered_cameras = self.scene_info.train_cameras

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
    def get_train_cameras(self, scale: float = 1.0) -> List[Camera]:
        return self.train_cameras[scale]

    # Get the test cameras for a given scale
    def get_test_cameras(self, scale: float = 1.0) -> List[Camera]:
        return self.test_cameras[scale]
    
    def __len__(self):
        # Return filtered camera count
        return len(self.filtered_cameras)
        
    def build_camera_graph(self):
        # Build a camera graph
        graph = nx.Graph()
        for camera_a in tqdm(self.filtered_cameras):
            for camera_b in self.filtered_cameras:
                # print(camera_a, camera_b)
                if camera_a.image_name != camera_b.image_name:
                    graph.add_edge(camera_a.image_name, camera_b.image_name, weight=view_similarity(camera_a, camera_b))
        return graph

def view_similarity(camera_a: CameraInfo, camera_b: CameraInfo):
    vec1, vec2 = camera_a.point3d_ids, camera_b.point3d_ids
    
    # Filter out '-1' from both vectors and convert them to sets
    set1 = set(vec1[vec1 != -1])
    set2 = set(vec2[vec2 != -1])
    
    # Find the intersection of both sets
    common_ids = set1.intersection(set2)
    
    # Return the count of common IDs
    return len(common_ids)