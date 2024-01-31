from typing import Callable
import torch
from tqdm import tqdm
from gaussian_renderer import render
from sgs2.helpers import PipelineParams
from sgs2.scene import Scene
from scene.cameras import Camera
from sgs2.gaussian import GaussianModel
from utils.loss_utils import ssim
import lpips
from utils.image_utils import psnr
# PIL
import PIL.Image
import numpy as np

lpips_fn = lpips.LPIPS(net='vgg').cuda()

def evaluate_camera(model: GaussianModel, camera: Camera, background: torch.Tensor = None, pipeline_params: PipelineParams = None):

    if background is None:
        background = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")

    if pipeline_params is None:
        pipeline_params = PipelineParams()

    with torch.no_grad():
        gt = camera.original_image[0:3, :, :].cuda()
        pred = render(camera, model, pipeline_params, background)["render"]

    s = ssim(pred, gt)
    p = psnr(pred, gt)
    l = lpips_fn(pred, gt)

    # return s, p, l, pred.cpu().numpy().transpose(1, 2, 0), gt.cpu().numpy().transpose(1, 2, 0)
    # Wrap with PIL to display in Streamlit
    return s, p, l, PIL.Image.fromarray((pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)), PIL.Image.fromarray((gt.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))


def evaluate_scene(scene: Scene, model: GaussianModel, pipeline_params: PipelineParams = None, progress_callback: Callable[[int, int], None] = None): # current, total
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

        train_cameras = scene.get_train_cameras() or []
        test_cameras = scene.get_test_cameras() or []

        # Evaluate training cameras
        print("Evaluating training cameras")
        for camera in tqdm(train_cameras):
            ssim_val, psnr_val, lpips_val, pred, gt = evaluate_camera(model, camera, pipeline_params=pipeline_params)
            ssim_val, psnr_val, lpips_val = ssim_val.item(), psnr_val.mean(dtype=float).item(), lpips_val.item()
            results['train_per_image'][camera.uid] = {'ssim': ssim_val, 'psnr': psnr_val, 'lpips': lpips_val, 'pred_image': pred, 'gt_image': gt}
            train_ssim_accum += ssim_val
            train_psnr_accum += psnr_val
            train_lpips_accum += lpips_val

            if progress_callback:
                progress_callback(len(results['train_per_image']), len(train_cameras) + len(test_cameras))

        # Evaluate test cameras
        for camera in tqdm(test_cameras):
            ssim_val, psnr_val, lpips_val, pred, gt = evaluate_camera(model, camera, pipeline_params=pipeline_params)
            ssim_val, psnr_val, lpips_val = ssim_val.item(), psnr_val.mean(dtype=float).item(), lpips_val.item()
            results['test_per_image'][camera.uid] = {'ssim': ssim_val, 'psnr': psnr_val, 'lpips': lpips_val, 'pred_image': pred, 'gt_image': gt}
            test_ssim_accum += ssim_val
            test_psnr_accum += psnr_val
            test_lpips_accum += lpips_val

            if progress_callback:
                progress_callback(len(results['test_per_image']) + len(results['train_per_image']), len(train_cameras) + len(test_cameras))

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
