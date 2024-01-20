import pandas as pd
import streamlit as st
import altair as alt

from split_gaussian_splatting.training_task import Task
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer

def main():

    # UI: Add title
    st.title("Gaussian Splatting Training Visualization")

    st.subheader("Training configuration")

    # UI: Ask for directory
    directory = st.text_input("Enter the directory for training data:", value="./datasets/train")

    # Initialize DataFrames for charts
    data_gaussians = pd.DataFrame(columns=['Iteration', 'Num Gaussians'])
    data_memory = pd.DataFrame(columns=['Iteration', 'Memory Used (MB)'])
    placeholder_gaussians = None
    placeholder_memory = None

    # Define callback for updating charts, passed to SimpleTrainer
    def update_charts(iteration, num_gaussians, memory_used):
        if iteration % 10 != 0:
            return
        
        nonlocal data_gaussians, data_memory

        # Append new data
        new_data_gaussians = {'Iteration': iteration, 'Num Gaussians': num_gaussians}
        new_data_memory = {'Iteration': iteration, 'Memory Used (MB)': memory_used}

        # data_gaussians = data_gaussians.append(new_data_gaussians, ignore_index=True)
        # data_memory = data_memory.append(new_data_memory, ignore_index=True)

        # Append is deprecated, so we use concat instead
        data_gaussians = pd.concat([data_gaussians, pd.DataFrame(new_data_gaussians, index=[0])], ignore_index=True)
        data_memory = pd.concat([data_memory, pd.DataFrame(new_data_memory, index=[0])], ignore_index=True)
        
        # Create Altair charts
        chart_gaussians = alt.Chart(data_gaussians).mark_line().encode(x='Iteration', y='Num Gaussians')
        chart_memory = alt.Chart(data_memory).mark_line().encode(x='Iteration', y='Memory Used (MB)')

        # Update the placeholders with new charts
        placeholder_gaussians.altair_chart(chart_gaussians, use_container_width=True)
        placeholder_memory.altair_chart(chart_memory, use_container_width=True)

    # UI: Add button to start training
    if st.button("Start Training"):

        task = Task(source_path=directory, iterations=100, data_device='cpu')
        trainer = SimpleTrainer(update_charts)

        scene = task.get_initial_scene()

        f'Loaded {len(list(scene.train_cameras.values())[-1] or [])} training cameras, {len(list(scene.test_cameras.values())[-1] or [])} test cameras.'

        st.subheader("Training progress")
        # Initialize placeholders for charts
        col1, col2 = st.columns(2)
        with col1:
            placeholder_gaussians = st.empty()
        with col2:
            placeholder_memory = st.empty()
        trainer.train(task, scene)

    

if __name__ == "__main__":
    main()
