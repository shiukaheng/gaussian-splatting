import subprocess
import time
import pandas as pd
import psutil
from pynvml import *

def monitor_command(command, interval=1):
    """
    Monitors the CPU usage, VRAM usage, and GPU utilization while running a command in a subprocess.

    Args:
        command (str): The command to be executed in a subprocess.
        interval (float, optional): The interval (in seconds) at which to collect data. Defaults to 1.

    Returns:
        pandas.DataFrame: A DataFrame containing the collected data, including timestamps, CPU usage,
        VRAM usage, and GPU utilization.
    """
    # Initialize NVML
    nvmlInit()

    # Start the command in a subprocess
    process = subprocess.Popen(command, shell=True)
    pid = process.pid

    # Prepare to collect data
    data = {
        'timestamp': [],
        'cpu_usage': [],
        'vram_usage': [],
        'gpu_utilization': []
    }

    # Get GPU handle (assuming single GPU)
    handle = nvmlDeviceGetHandleByIndex(0)

    try:
        while process.poll() is None:  # While process is running
            # Record timestamp
            data['timestamp'].append(time.time())

            # Monitor CPU usage
            proc = psutil.Process(pid)
            cpu_usage = proc.cpu_percent(interval=interval)
            data['cpu_usage'].append(cpu_usage)

            # Monitor VRAM and GPU Utilization
            gpu_info = nvmlDeviceGetUtilizationRates(handle)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            vram_usage = mem_info.used / 1024 / 1024  # Convert to MB
            gpu_utilization = gpu_info.gpu

            data['vram_usage'].append(vram_usage)
            data['gpu_utilization'].append(gpu_utilization)

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Cleanup NVML
        nvmlShutdown()

    # Convert data to pandas DataFrame
    df = pd.DataFrame(data)
    return df

def create_training_command(source_path, model_path='output/<random>', images='images', eval=False,
                            resolution=None, data_device='cuda', white_background=False, sh_degree=3,
                            convert_SHs_python=False, convert_cov3D_python=False, debug=False, debug_from=0,
                            iterations=30000, ip='127.0.0.1', port=6009, test_iterations='7000 30000',
                            save_iterations='7000 30000 <iterations>', checkpoint_iterations=None, start_checkpoint=None,
                            quiet=False, feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001,
                            position_lr_max_steps=30000, position_lr_init=0.00016, position_lr_final=0.0000016,
                            position_lr_delay_mult=0.01, densify_from_iter=500, densify_until_iter=15000,
                            densify_grad_threshold=0.0002, densification_interval=100, opacity_reset_interval=3000,
                            lambda_dssim=0.2, percent_dense=0.01):
    """
    Creates a command for training based on the provided parameters.
    """
    cmd = f"python train.py -s {source_path}"

    if model_path:
        cmd += f" -m {model_path}"
    if images:
        cmd += f" -i {images}"
    if eval:
        cmd += " --eval"
    if resolution is not None:
        cmd += f" -r {resolution}"
    if data_device:
        cmd += f" --data_device {data_device}"
    if white_background:
        cmd += " -w"
    cmd += f" --sh_degree {sh_degree}"
    if convert_SHs_python:
        cmd += " --convert_SHs_python"
    if convert_cov3D_python:
        cmd += " --convert_cov3D_python"
    if debug:
        cmd += " --debug"
    cmd += f" --debug_from {debug_from}"
    cmd += f" --iterations {iterations}"
    cmd += f" --ip {ip}"
    cmd += f" --port {port}"
    cmd += f" --test_iterations {test_iterations}"
    cmd += f" --save_iterations {save_iterations}"
    if checkpoint_iterations:
        cmd += f" --checkpoint_iterations {checkpoint_iterations}"
    if start_checkpoint:
        cmd += f" --start_checkpoint {start_checkpoint}"
    if quiet:
        cmd += " --quiet"
    cmd += f" --feature_lr {feature_lr}"
    cmd += f" --opacity_lr {opacity_lr}"
    cmd += f" --scaling_lr {scaling_lr}"
    cmd += f" --rotation_lr {rotation_lr}"
    cmd += f" --position_lr_max_steps {position_lr_max_steps}"
    cmd += f" --position_lr_init {position_lr_init}"
    cmd += f" --position_lr_final {position_lr_final}"
    cmd += f" --position_lr_delay_mult {position_lr_delay_mult}"
    cmd += f" --densify_from_iter {densify_from_iter}"
    cmd += f" --densify_until_iter {densify_until_iter}"
    cmd += f" --densify_grad_threshold {densify_grad_threshold}"
    cmd += f" --densification_interval {densification_interval}"
    cmd += f" --opacity_reset_interval {opacity_reset_interval}"
    cmd += f" --lambda_dssim {lambda_dssim}"
    cmd += f" --percent_dense {percent_dense}"