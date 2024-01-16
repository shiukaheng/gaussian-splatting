import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def getSceneInfoFromSource(source_path, images, white_background, eval):
    """
    Retrieves scene information based on the given source path, images, white background flag, and evaluation flag.

    Parameters:
    source_path (str): The path to the source directory.
    images (list): A list of image paths.
    white_background (bool): Flag indicating whether the background should be white.
    eval (bool): Flag indicating whether we hold data for evaluation. (Test images will be held out according to the LLFF holdout scheme)

    Returns:
    scene_info: The retrieved scene information.

    Raises:
    AssertionError: If the scene type cannot be recognized.
    """
    if os.path.exists(os.path.join(source_path, "sparse")): # Presumed to be a COLMAP scene
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, images, eval)
    elif os.path.exists(os.path.join(source_path, "transforms_train.json")): # Presumed to be a Blender scene
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](source_path, white_background, eval)
    else:
        assert False, "Could not recognize scene type!"

def initializeModelDir(ply_path, model_path, test_cameras, train_cameras):
    """
    Initializes the model directory by copying the original ply file to a file called input.ply in the model directory.
    It also collects all the used cameras into a single set, converts them to JSON form, and saves them to a file.

    Args:
        ply_path (str): The path to the original ply file.
        model_path (str): The path to the model directory.
        test_cameras (list): A list of test cameras.
        train_cameras (list): A list of train cameras.
    """
    with open(ply_path, 'rb') as src_file, open(os.path.join(model_path, "input.ply") , 'wb') as dest_file:
        dest_file.write(src_file.read())

    camset = set()
    if test_cameras:
        camset.update(test_cameras)
    if train_cameras:
        camset.update(train_cameras)

    camlist = list(camset)

    json_cams = []
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(model_path, "cameras.json"), 'w') as file:
        json.dump(json_cams, file)