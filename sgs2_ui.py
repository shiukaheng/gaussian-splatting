import streamlit as st

from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import Task

def get_source_path():
    c = st.container()
    c.subheader("Training configuration")
    directory = c.text_input("Enter the directory for training data:", value="./datasets/train")
    return directory

def train_models(path, methods=[SimpleTrainer]):
    c = st.container()
    task = Task(source_path=path, iterations=100, data_device='cpu')
    scene = task.load_scene()
    c.subheader("Training progress")
    for method in methods:
        c.write(f"Training with {method}")
        c.write(f"Loaded {len(list(scene.train_cameras.values())[-1] or [])} training cameras, {len(list(scene.test_cameras.values())[-1] or [])} test cameras.")
        trainer = method(update_charts)
        trainer.train(task, scene, gaussian_model)

    pass

def eval_models(task, scene, models):
    pass

def main():
    st.title("Gaussian Splatting Training Visualization")
    path = get_source_path()
    task, scene, models = train_models(path)
    eval_models(task, scene, models)