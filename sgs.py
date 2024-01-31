from split_gaussian_splatting.evaluate import evaluate_scene
from split_gaussian_splatting.trainers.grid_trainer import GridTrainer
from split_gaussian_splatting.trainers.simple_trainer import SimpleTrainer
from split_gaussian_splatting.training_task import ProjectParams, SimpleTrainerParams

# Define project parameters (what images we want to train on)
source = ProjectParams(source_path="./datasets/train")

# Create the Scene from the project parameters
scene = source.load_scene()

# Define trainer parameters (how we want to train)
trainer_params = SimpleTrainerParams(iterations=200, densify_from_iter=0, densification_interval=10)

# Create the GaussianModel from the scene
gaussian_model = scene.create_gaussians(trainer_params)

# Create the trainer
simple_trainer = SimpleTrainer(lambda x, y, z: print(x, y, z))
scene, gaussian_model = simple_trainer.train(trainer_params)

simple_eval = evaluate_scene(scene, gaussian_model, trainer_params)

'''
TODO: Separate rendering parameters in evaluation
TODO: Merge ProjectParams into scene
TODO: Merge SimpleTrainerParams into SimpleTrainer
'''

print(simple_eval["train"])