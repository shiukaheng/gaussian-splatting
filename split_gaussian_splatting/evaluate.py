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

import torch
from tqdm import tqdm
from gaussian_renderer import render
from scene import Scene
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from split_gaussian_splatting.training_task import Task
from utils.loss_utils import ssim
import lpips
from utils.image_utils import psnr

lpips_fn = lpips.LPIPS(net='vgg').cuda()

def evaluate_camera(model: GaussianModel, task: Task, camera: Camera):

    bg_color = [1,1,1] if task.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gt = camera.original_image[0:3, :, :].cuda()
    pred = render(camera, model, task, background)["render"]

    s = ssim(pred, gt)
    p = psnr(pred, gt)
    l = lpips_fn(pred, gt)

    return s, p, l

def evaluate_scene(scene: Scene, model: GaussianModel, task: Task):
    with torch.no_grad():

        train_ssim_accum = 0
        train_psnr_accum = 0
        train_lpips_accum = 0
        test_ssim_accum = 0
        test_psnr_accum = 0
        test_lpips_accum = 0

        results = {
            'train': {},
            'test': {},
            'train_per_image': {},
            'test_per_image': {}
        }

        train_cameras = scene.getTrainCameras() or []
        test_cameras = scene.getTestCameras() or []

        # Evaluate training cameras
        print("Evaluating training cameras")
        for camera in tqdm(train_cameras):
            ssim_val, psnr_val, lpips_val = evaluate_camera(model, task, camera)
            ssim_val, psnr_val, lpips_val = ssim_val.item(), psnr_val.mean(dtype=float).item(), lpips_val.item()
            results['train_per_image'][camera.uid] = {'ssim': ssim_val, 'psnr': psnr_val, 'lpips': lpips_val}
            train_ssim_accum += ssim_val
            train_psnr_accum += psnr_val
            train_lpips_accum += lpips_val

        # Evaluate test cameras
        for camera in tqdm(test_cameras):
            ssim_val, psnr_val, lpips_val = evaluate_camera(model, task, camera)
            ssim_val, psnr_val, lpips_val = ssim_val.item(), psnr_val.mean(dtype=float).item(), lpips_val.item()
            results['test_per_image'][camera.uid] = {'ssim': ssim_val, 'psnr': psnr_val, 'lpips': lpips_val}
            test_ssim_accum += ssim_val
            test_psnr_accum += psnr_val
            test_lpips_accum += lpips_val

        # Calculate average metrics, but only if we have cameras
        if len(train_cameras) > 0:
            results['train']['ssim'] = train_ssim_accum / len(train_cameras)
            results['train']['psnr'] = train_psnr_accum / len(train_cameras)
            results['train']['lpips'] = train_lpips_accum / len(train_cameras)
        else:
            results['train']['ssim'] = 0
            results['train']['psnr'] = 0
            results['train']['lpips'] = 0
            print("No training cameras")

        if len(test_cameras) > 0:
            results['test']['ssim'] = test_ssim_accum / len(test_cameras)
            results['test']['psnr'] = test_psnr_accum / len(test_cameras)
            results['test']['lpips'] = test_lpips_accum / len(test_cameras)
        else:
            results['test']['ssim'] = 0
            results['test']['psnr'] = 0
            results['test']['lpips'] = 0
            print("No test cameras")

        torch.cuda.empty_cache()

    return results
