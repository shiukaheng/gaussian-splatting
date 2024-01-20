from split_gaussian_splatting.evaluate import evaluate_scene
from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.trainers.grid_trainer import GridTrainer
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import Task

task = Task(source_path="./datasets/train", iterations=100, data_device='cpu')

grid_trainer = GridTrainer()
scene, gaussian_model = grid_trainer.train(task)

grid_eval = evaluate_scene(scene, gaussian_model, task)

simple_trainer = SimpleTrainer()
scene, gaussian_model = simple_trainer.train(task)

simple_eval = evaluate_scene(scene, gaussian_model, task)

print(simple_eval["train"])
print(grid_eval["train"])