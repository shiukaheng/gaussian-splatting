import sys
import pandas as pd
import streamlit as st
import altair as alt
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.trainers.grid_trainer import GridTrainer

from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import Task

def is_streamlit_app():
    """Check if the script is running in a Streamlit environment."""
    return 'streamlit' in sys.modules

def get_source_path():
    # Initialize the state if it's a Streamlit app
    if is_streamlit_app():
        if 'training_started' not in st.session_state:
            st.session_state.training_started = False

    c = st.container()
    c.subheader("Training configuration")
    directory = c.text_input("Source data directory:", value="./datasets/train")
    iterations = c.number_input("Iterations to train for:", value=100)
    
    # Create button and change the state when clicked
    if c.button("Start Training"):
        st.session_state.training_started = True
        return directory, iterations

    # If the button hasn't been clicked, return None
    if not st.session_state.training_started:
        return None
    
def train_models(path, iterations, methods=[SimpleTrainer]):

    c = st.container()
    task = Task(source_path=path, iterations=iterations, data_device='cpu')
    with c.status("Dataset", expanded=True):
        bar = st.progress(0, text="Loading images...")
        scene = task.load_scene(on_load_progress=lambda cur, max: bar.progress(cur / max, text=f"Loading images... ({cur}/{max})"))
        # Animate remove progress bar and switch to "Loaded cameras."
        bar.empty()
        st.success(f"Loaded {len(list(scene.train_cameras.values())[-1] or [])} training cameras, {len(list(scene.test_cameras.values())[-1] or [])} test cameras.")

    models = {}

    s2 = c.status("Training", expanded=True)
    with s2:
        training_status = s2.empty()

        tabs = st.tabs([x.__name__ for x in methods])
        method_tabs = zip(tabs, methods)

        for i, (tab, method) in enumerate(method_tabs):
            train_progress = training_status.progress(0, text=f"Training {method.__name__}... (0/{iterations})")
            # Prepare columns for two altair charts
            col1, col2 = s2.columns(2)
            placeholder_gaussians = col1.empty()
            placeholder_memory = col2.empty()
            data_gaussians = pd.DataFrame(columns=['Iteration', 'Num Gaussians'])
            data_memory = pd.DataFrame(columns=['Iteration', 'Memory Used (MB)'])

            # Define callback for updating charts, passed to trainers
            def update_charts(iteration, num_gaussians, memory_used):
                train_progress.progress(iteration / iterations, text=f"Training {method.__name__}...  Model: {i + 1}/{len(methods)}, Iteration: {iteration}/{iterations}")
                if iteration % 10 != 0:
                    return
                nonlocal data_gaussians, data_memory
                # Append new data
                new_data_gaussians = {'Iteration': iteration, 'Num Gaussians': num_gaussians}
                new_data_memory = {'Iteration': iteration, 'Memory Used (MB)': memory_used}
                data_gaussians = pd.concat([data_gaussians, pd.DataFrame(new_data_gaussians, index=[0])], ignore_index=True)
                data_memory = pd.concat([data_memory, pd.DataFrame(new_data_memory, index=[0])], ignore_index=True)
                # Create Altair charts
                chart_gaussians = alt.Chart(data_gaussians).mark_line().encode(x='Iteration', y='Num Gaussians')
                chart_memory = alt.Chart(data_memory).mark_line().encode(x='Iteration', y='Memory Used (MB)')

                # Update the placeholders with new charts
                placeholder_gaussians.altair_chart(chart_gaussians, use_container_width=True)
                placeholder_memory.altair_chart(chart_memory, use_container_width=True)
            
            with tab:
                trainer: BaseTrainer = method(update_charts)
                models[method.__name__] = trainer.train(task, scene)
            
        training_status.success("Training complete!")

        return task, scene, models

    # pass
    return None, None, None

def main():
    st.set_page_config(page_title="Gaussian Splatting Training Visualization", page_icon="📷")
    st.title("📷 Gaussian Splatting Training Visualization")

    data = get_source_path()
    
    # Only proceed with training if the path is not None (i.e., the button has been pressed)
    if data:
        path, iterations = data
        task, scene, models = train_models(path, iterations, [SimpleTrainer, GridTrainer])
        # eval_models(task, scene, models)

if __name__ == "__main__":
    main()
