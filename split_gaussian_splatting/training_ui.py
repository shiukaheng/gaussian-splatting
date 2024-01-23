import base64
from io import BytesIO
import sys
from typing import List
import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException
import altair as alt
from split_gaussian_splatting.evaluate import evaluate_scene
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.trainers.grid_trainer import GridTrainer
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import Task
from PIL import Image
from gaussian_renderer import network_gui

def is_streamlit_app():
    """Check if the script is running in a Streamlit environment."""
    return 'streamlit' in sys.modules

def get_source_path():
    # Initialize the state if it's a Streamlit app
    if is_streamlit_app():
        if 'training_started' not in st.session_state:
            st.session_state.training_started = False
        if 'render_images' not in st.session_state:
            st.session_state.render_images = False

    c = st.container()
    c.subheader("Training configuration")
    directory = c.text_input("Source data directory:", value="./datasets/train")
    iterations = c.number_input("Iterations to train for:", value=100)

    # Checkbox for rendering images
    if c.checkbox("Render images (slow)"):
        st.session_state.render_images = True
    else:
        st.session_state.render_images = False
    
    # Create button and change the state when clicked
    if c.button("Start Training"):
        st.session_state.training_started = True
        return directory, iterations

    # If the button hasn't been clicked, return None
    if not st.session_state.training_started:
        return None
    
def train_models(path, iterations, methods=[SimpleTrainer]):

    network_gui.init("127.0.0.1", 6009)

    c = st.container()
    task = Task(source_path=path, iterations=iterations, data_device='cpu', densify_from_iter=0, densification_interval=50, opacity_reset_interval=300)
    with c.status("Dataset", expanded=True):
        bar = st.progress(0, text="Loading images...")
        scene = task.load_scene(on_load_progress=lambda cur, max: bar.progress(cur / max, text=f"Loading images... ({cur}/{max})"))
        # Animate remove progress bar and switch to "Loaded cameras."
        bar.empty()
        st.success(f"Loaded {len(list(scene.train_cameras.values())[-1] or [])} training cameras, {len(list(scene.test_cameras.values())[-1] or [])} test cameras.")
    
        # # For the first 5 images, we plot it out
        # training_cameras: List[Camera] = scene.getTrainCameras() or []
        # test_cameras: List[Camera] = scene.getTestCameras() or []
        # images = [camera.original_image[0:3, :, :].cpu().numpy().transpose(1, 2, 0) for camera in camera_selection]

        # for image in images:
        #     st.image(image, caption="Original image", use_column_width=True)
    models = {}

    s2 = c.status("Training", expanded=True)
    with s2:
        training_status = s2.empty()

        # Prepare columns for two altair charts
        col1, col2 = s2.columns(2)
        placeholder_gaussians = col1.empty()
        placeholder_memory = col2.empty()

        data_gaussians = pd.DataFrame(columns=['Iteration', 'Num Gaussians', 'Method'])
        data_memory = pd.DataFrame(columns=['Iteration', 'Memory Used (MB)', 'Method'])

        for i, method in enumerate(methods):
            train_progress = training_status.progress(0, text=f"Training {method.__name__}... (0/{iterations})")
            
            # Define callback for updating charts, passed to trainers
            def update_charts(iteration, num_gaussians, memory_used):

                train_progress.progress(iteration / iterations, text=f"Training {method.__name__}...  Model: {i + 1}/{len(methods)}, Iteration: {iteration}/{iterations}")
                
                nonlocal data_gaussians, data_memory

                # Append new data
                new_data_gaussians = {'Iteration': iteration, 'Num Gaussians': num_gaussians, 'Method': method.__name__}
                new_data_memory = {'Iteration': iteration, 'Memory Used (MB)': memory_used, 'Method': method.__name__}
                data_gaussians = pd.concat([data_gaussians, pd.DataFrame(new_data_gaussians, index=[0])], ignore_index=True)
                data_memory = pd.concat([data_memory, pd.DataFrame(new_data_memory, index=[0])], ignore_index=True)

                # Create Altair charts
                chart_gaussians = alt.Chart(data_gaussians).mark_line().encode(x='Iteration', y='Num Gaussians', color='Method')
                chart_memory = alt.Chart(data_memory).mark_line().encode(x='Iteration', y='Memory Used (MB)', color='Method')

                # Update the placeholders with new charts
                if iteration % 5 == 0:
                    placeholder_gaussians.altair_chart(chart_gaussians, use_container_width=True)
                    placeholder_memory.altair_chart(chart_memory, use_container_width=True)
            
            trainer: BaseTrainer = method(update_charts) # Create trainer with callback
            _, gaussian = trainer.train(task, scene)
            gaussian.archive_to_cpu()
            models[method.__name__] = gaussian
            
        training_status.success("Training complete!")

        return task, scene, models
    
def eval_models(task, scene, models):
    
    # Initialize tables; Highlight max for SSIM and PSNR, min for LPIPS
    train_agg = pd.DataFrame(columns=['Model', 'SSIM', 'PSNR', 'LPIPS'])
    test_agg = pd.DataFrame(columns=['Model', 'SSIM', 'PSNR', 'LPIPS'])

    per_image = pd.DataFrame(columns=['Model', 'SSIM', 'PSNR', 'LPIPS', 'Camera'])

    # Evaluate model status box
    with st.status("Evaluation", expanded=True):

        # Progress bar initialization
        eval_progress = st.empty()
        eval_progress.progress(0, text="Evaluating models...")
        results = {}

        for i, (name, model) in enumerate(models.items()):
            model.unarchive_to_cuda(task)
            results[name] = evaluate_scene(scene, model, task, progress_callback=lambda cur, max: eval_progress.progress(cur / max, text=f"Evaluating {name}... Model: {i + 1}/{len(models)}, Image: ({cur}/{max})"))
            train_agg = pd.concat([train_agg, pd.DataFrame({'Model': name, 'SSIM': results[name]["train"]["ssim"], 'PSNR': results[name]["train"]["psnr"], 'LPIPS': results[name]["train"]["lpips"]}, index=[0])], ignore_index=True)
            test_agg = pd.concat([test_agg, pd.DataFrame({'Model': name, 'SSIM': results[name]["test"]["ssim"], 'PSNR': results[name]["test"]["psnr"], 'LPIPS': results[name]["test"]["lpips"]}, index=[0])], ignore_index=True)
            per_image = pd.concat([per_image, pd.DataFrame({'Model': name, 'SSIM': [v['ssim'] for v in results[name]["train_per_image"].values()], 'PSNR': [v['psnr'] for v in results[name]["train_per_image"].values()], 'LPIPS': [v['lpips'] for v in results[name]["train_per_image"].values()], 'Camera': [k for k in results[name]["train_per_image"].keys()]})], ignore_index=True)

        train_agg_style = train_agg.style.highlight_max(
                color='green', axis=0, subset=['SSIM', 'PSNR']
            ).highlight_min(
                color='green', axis=0, subset=['LPIPS']
            )
        test_agg_style = test_agg.style.highlight_max(
                color='green', axis=0, subset=['SSIM', 'PSNR']
            ).highlight_min(
                color='green', axis=0, subset=['LPIPS']
            )

        eval_progress.success("Evaluation complete!")
        
        col1, col2 = st.columns(2)

        col1.subheader("Training set results")

        if len(scene.getTrainCameras() or []) > 0:
            col1.dataframe(train_agg_style)
        else:
            col1.write("No training cameras")

        col2.subheader("Test set results")

        if len(scene.getTestCameras() or []) > 0:
            col2.dataframe(test_agg_style)
        else:
            col2.write("No test cameras")

        st.subheader("Per-image results")

        metrics = ["SSIM", "PSNR", "LPIPS"]
        cols = st.columns(len(metrics))
        metric_cols = dict(zip(metrics, cols))
        for metric, col in metric_cols.items():
            # Boxplot to compare models
            col.subheader(f"{metric} comparison")
            col.altair_chart(alt.Chart(per_image).mark_boxplot().encode(x='Model', y=metric, color='Model'), use_container_width=True)
        
        if st.session_state.render_images:

            st.subheader("Full results")
            model_tabs = zip(models.keys(), st.tabs(models.keys()))

            def image_formatter(image: Image):  # Input is PIL image
                # Crop to square (centered)
                width, height = image.size
                min_dim = min(width, height)
                left = (width - min_dim) / 2
                top = (height - min_dim) / 2
                right = (width + min_dim) / 2
                bottom = (height + min_dim) / 2

                image = image.crop((left, top, right, bottom))

                # Resize to 256x256
                image = image.resize((256, 256))
                
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                return f"<img src='data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}'/>"

            for name, tab in model_tabs:
                rendered = pd.DataFrame(results[name]['train_per_image']).transpose().style.format({'pred_image': image_formatter, 'gt_image': image_formatter}).to_html(escape=False)
                tab.write(rendered, unsafe_allow_html=True)

        return results
    
def training_ui():

    try:

        st.set_page_config(page_title="Gaussian Splatting Training Visualization", page_icon="📷", layout="wide")
        st.title("📷 Gaussian Splatting Training Visualization")

        data = get_source_path()
        
        # Only proceed with training if the path is not None (i.e., the button has been pressed)
        if data:
            path, iterations = data
            task, scene, models = train_models(path, iterations, [SimpleTrainer, GridTrainer]) 
            # TODO: Refactor so it takes an instance of a trainer instead. This trainer can then take
            # training parameters such that we can compare performance of different training parameters too.
            # It probably could use a name parameter too.
            eval_models(task, scene, models)

    except (RerunException, StopException, KeyboardInterrupt):

        network_gui.close()