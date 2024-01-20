from split_gaussian_splatting.training_task import Task
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.evaluate import evaluate_scene

# Task is the description of what needs to be trained
task = Task(
    source_path='./datasets/train',
    iterations=200,
) 

trainer = SimpleTrainer()

# Scene includes the finished model, and loaded images
scene = trainer.train(task)