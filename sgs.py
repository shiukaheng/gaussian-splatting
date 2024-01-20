from split_gaussian_splatting.trainers.base_trainer import BaseTrainer
from split_gaussian_splatting.trainers.grid_trainer import GridTrainer
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import Task

task = Task(source_path="./datasets/train", iterations=100, data_device='cpu')

trainer = GridTrainer()
trainer.train(task)