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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import streamlit as st

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1: # No resolution specified
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    if cam_infos is None or len(cam_infos) == 0:
        return 
    
    bar = st.progress(0, text=f'Loading 0/{len(cam_infos)} cameras at {resolution_scale}x resolution')

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
        bar.progress((id + 1) / len(cam_infos), text=f'Loading {id + 1}/{len(cam_infos)} cameras at {resolution_scale}x resolution')
    

    return camera_list

# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
#     camera_list = []

#     # Define how many threads to use
#     num_threads = 4  # Adjust this based on your system's capabilities

#     # Function to be executed in parallel
#     def process_camera(id, cam_info):
#         return loadCam(args, id, cam_info, resolution_scale)

#     # Using ThreadPoolExecutor to parallelize
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         # Schedule the loadCam function to be executed for each camera
#         futures = [executor.submit(process_camera, id, c) for id, c in enumerate(cam_infos)]

#         # Wait for all futures to complete
#         for future in futures:
#             camera_list.append(future.result())

#     return camera_list

# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
#     camera_list = []

#     num_processes = 4  # Adjust based on your system

#     def process_camera(id, cam_info):
#         return loadCam(args, id, cam_info, resolution_scale)

#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = [executor.submit(process_camera, id, c) for id, c in enumerate(cam_infos)]

#         for future in futures:
#             camera_list.append(future.result())

#     return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
