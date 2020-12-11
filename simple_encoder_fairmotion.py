import os
from math import inf
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from fairmotion.data import bvh
from fairmotion.data import amass_dip
from fairmotion.data import amass
from fairmotion.core.motion import Motion, Pose
from fairmotion.ops import motion as motion_ops
from fairmotion.tasks.motion_prediction import utils
from fairmotion.utils import utils as fairmotion_utils, conversions
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import DataLoader
from torchviz import make_dot
import torch.nn.functional as F

import motion_data
from motion_dataset import MotionDataset
from simple_encoder import SimpleAutoEncoder
from vae_encoder import VanillaVAE

device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability() != (3, 0) and
                                torch.cuda.get_device_capability()[0] >= 3 else "cpu")
batch_size = 8

print("Setting up train dataset...")
train_dataset = MotionDataset(motion_data.load_train())
print("Setting up train loader...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

print("Setting up test dataset...")
test_dataset = MotionDataset(
    motion_data.load_test(),
    std=train_dataset.std,
    mean=train_dataset.mean
)
print("Setting up test loader...")
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

# hidden_dimensions = [256, 128]
hidden_dimensions = [1024, 1024, 16]

model = None
model_type = "vae"
model_save_path = f"data/{model_type}_{'-'.join([str(x) for x in hidden_dimensions])}_batch-{batch_size}.pt"
losses_save_path = f"data/{model_type}_{'-'.join([str(x) for x in hidden_dimensions])}_batch-{batch_size}_losses.csv"
if model_type == "simple":
    print("Setting up simple model...")
    model = SimpleAutoEncoder(input_size=198, hidden_dimensions=hidden_dimensions).double().to(device)
    if os.path.exists(model_save_path):
        print("Loading saved best simple model...")
        model.load_state_dict(torch.load(model_save_path))
elif model_type == "vae":
    print("Setting up VAE model...")
    model = VanillaVAE(input_size=198, hidden_dimensions=hidden_dimensions).double().to(device)
    if os.path.exists(model_save_path):
        print("Loading saved best VAE model...")
        model.load_state_dict(torch.load(model_save_path))
else:
    raise Exception("Unknown model type")
best_test_loss = -inf
best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-5)


def test_eval():
    model.eval()
    with torch.no_grad():
        kld_loss = 0
        reconstruction_loss = 0
        flattened_pose_matrix_batch: torch.Tensor
        for flattened_pose_matrix_batch in test_loader:
            flattened_pose_matrix_batch = flattened_pose_matrix_batch.to(device)
            if model_type == "simple":
                outputs = model(flattened_pose_matrix_batch)
                reconstruction_loss += F.mse_loss(outputs, flattened_pose_matrix_batch, reduction='sum').item()
            elif model_type == "vae":
                mu, log_var = model.encode(flattened_pose_matrix_batch)
                reconstructed_input = model.decode(mu)
                reconstruction_loss += F.mse_loss(reconstructed_input, flattened_pose_matrix_batch,
                                                  reduction='sum').item()
                kld_loss += -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()).item()
            else:
                raise Exception("Unknown model type")

        return [
            (reconstruction_loss + kld_loss) / len(test_loader.dataset),
            reconstruction_loss / len(test_loader.dataset),
            kld_loss / len(test_loader.dataset)
        ]


should_train = True
if should_train:
    saved_model_visualization = False

    train_losses = []
    test_losses = []
    if os.path.exists(losses_save_path):
        print("Loading losses...")
        with open(losses_save_path, "r") as file:
            for line in file.readlines()[1:]:
                train_losses += [[float(string) for string in line.split(",")][:3]]
                test_losses += [[float(string) for string in line.split(",")][3:]]

    print("Starting training...")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_reconstruction_loss = 0
        train_kld_loss = 0
        flattened_pose_matrix_batch: torch.Tensor
        for flattened_pose_matrix_batch in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            # batch_features = batch_features.view(-1, 28 * 28).to(device)
            flattened_pose_matrix_batch = flattened_pose_matrix_batch.to(device)

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(flattened_pose_matrix_batch)

            if not saved_model_visualization:
                make_dot(outputs[0], params=dict(model.named_parameters())).save(
                    f"data/model-visualizations/{model_type}_mnist")
                saved_model_visualization = True

            # compute training reconstruction loss
            if model_type == "simple":
                reconstruction_loss = F.mse_loss(outputs, flattened_pose_matrix_batch, reduction='sum')
                total_loss = reconstruction_loss

                train_reconstruction_loss += reconstruction_loss.item()
                train_total_loss += total_loss.item()
            elif model_type == "vae":
                reconstructed_input, mu, log_var = outputs
                reconstruction_loss = F.mse_loss(reconstructed_input, flattened_pose_matrix_batch, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                total_loss = reconstruction_loss + kld_loss

                train_reconstruction_loss += reconstruction_loss.item()
                train_kld_loss += kld_loss.item()
                train_total_loss += total_loss.item()
            else:
                raise Exception("Unknown model type")

            # compute accumulated gradients
            total_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

        # compute the epoch training loss
        train_total_loss = train_total_loss / len(train_loader.dataset)
        train_reconstruction_loss = train_reconstruction_loss / len(train_loader.dataset)
        train_kld_loss = train_kld_loss / len(train_loader.dataset)

        # display the epoch training loss
        test_total_loss, test_reconstruction_loss, test_kld_loss = test_eval()

        if test_total_loss < best_test_loss:
            best_test_loss = test_total_loss
            best_model = model

        print(f"epoch : {epoch + 1}/{epochs}")
        train_losses += [[train_total_loss, train_reconstruction_loss, train_kld_loss]]
        test_losses += [[test_total_loss, test_reconstruction_loss, test_kld_loss]]
        print(
            f"train total loss = {train_total_loss:.6f}, train reconstruction loss = {train_reconstruction_loss:.6f}, train kld loss = {train_kld_loss:.6f}")
        print(
            f"test total loss = {test_total_loss:.6f}, test reconstruction loss = {test_reconstruction_loss:.6f}, test kld loss = {test_kld_loss:.6f}")

        # plt.plot([x for x, _, _ in train_losses[-20:]], label="train_total_loss")
        # plt.plot([x for x, _, _ in test_losses[-20:]], label="test_total_loss")
        plt.plot([x for _, x, _ in train_losses[-20:]], label="train_reconstruction_loss")
        # plt.plot([x for _, x, _ in test_losses[-20:]], label="test_reconstruction_loss")
        # plt.plot([x for _, _, x in train_losses[-20:]], label="train_kld_loss", linestyle="dashed")
        # plt.plot([x for _, _, x in test_losses[-20:]], label="test_kld_loss", linestyle="dashed")
        plt.title(f"{model_type}, layers {hidden_dimensions}, batch_size {batch_size}")
        plt.xlabel("epochs")
        plt.legend(loc='upper left')
        plt.show()

    print("Saving model...")
    torch.save(model.state_dict(), model_save_path)

    print("Saving losses...")
    with open(losses_save_path, "w") as file:
        file.write(
            "train_total_loss, train_reconstruction_loss, train_kld_loss, test_total_loss, test_reconstruction_loss, test_kld_loss\n")
        for train_loss, test_loss in zip(train_losses, test_losses):
            file.write(", ".join([str(x) for x in (train_loss + test_loss)]) + "\n")


def evaluate(motion: Motion, steps: List[int]) -> float:
    print(f"Evaluating in-painting of {motion.name}...")
    loss = 0
    for step in steps:
        latent_inpainted_motion = inpaint_motion(motion, step, "latent")
        linear_inpainted_motion = inpaint_motion(motion, step, "linear")
        motion_data.save(latent_inpainted_motion, f"data/generated/{model_type}/inpaint_{step}_{motion.name}.bvh")
        motion_data.save(linear_inpainted_motion, f"data/generated/linear/inpaint_{step}_{motion.name}.bvh")
        latent_step_loss = mse_loss(motion, latent_inpainted_motion)
        linear_step_loss = mse_loss(motion, linear_inpainted_motion)
        print(
            f"motion: {motion.name}, step: {step}, latent loss: {latent_step_loss:.6f}, linear loss: {linear_step_loss:.6f}, diff {linear_step_loss - latent_step_loss:.6f}")
        loss += latent_step_loss
    return loss / len(steps)


def inpaint_motion(motion: Motion, step: int, interpolation: str) -> Motion:
    """Returns a motion that has been inpainted by only preserving every `step` frames"""
    inpainted_motion = Motion.from_matrix(motion.to_matrix(), motion.skel)
    for begin, end in zip(range(0, inpainted_motion.num_frames(), step),
                          range(0, inpainted_motion.num_frames(), step)[1:]):
        for i in range(begin + 1, end):
            progress = (i - begin) / (end - begin)
            if interpolation == "latent":
                inpainted_motion.poses[i] = latent_interpolate(inpainted_motion.poses[begin],
                                                               inpainted_motion.poses[end],
                                                               motion.poses[i],  # for root position data
                                                               progress)
            elif interpolation == "linear":
                inpainted_motion.poses[i] = linear_interpolate(inpainted_motion.poses[begin],
                                                               inpainted_motion.poses[end],
                                                               motion.poses[i],  # for root position data
                                                               progress)
            else:
                raise Exception("Unknown interpolation")
    return inpainted_motion


def mse_loss(original_motion: Motion, inpainted_motion: Motion) -> float:
    error = 0
    count = 0
    for original_pose, inpainted_pose in zip(original_motion.poses, inpainted_motion.poses):
        original_matrix = original_pose.to_matrix().flatten()
        inpainted_matrix = inpainted_pose.to_matrix().flatten()
        for original_value, inpainted_value in zip(original_matrix, inpainted_matrix):
            count += 1
            error += pow(original_value - inpainted_value, 2)

    return error / count


def latent_interpolate(left: Pose, right: Pose, actual: Pose, progress: float) -> Pose:
    left_input = torch.tensor(train_dataset.normalize(conversions.T2R(left.to_matrix())).flatten()).to(device)
    right_input = torch.tensor(train_dataset.normalize(conversions.T2R(right.to_matrix())).flatten()).to(device)

    left_latent = model.encode(left_input) if model_type == "simple" else model.encode(left_input)[0]
    right_latent = model.encode(right_input) if model_type == "simple" else model.encode(right_input)[0]
    interpolated_latent_values: List[float] = []
    for i in range(len(left_latent)):
        interpolated_latent_values += [left_latent[i] + (right_latent[i] - left_latent[i]) * progress]
    interpolated_latent = torch.tensor(interpolated_latent_values).to(device)
    interpolated = model.decode(interpolated_latent)
    pose_matrix = conversions.Rp2T(
        train_dataset.unnormalize(interpolated.reshape((-1, 3, 3)).cpu()),
        conversions.T2p(actual.to_matrix())
    )
    return Pose.from_matrix(pose_matrix, left.skel)


def linear_interpolate(left: Pose, right: Pose, actual: Pose, progress: float) -> Pose:
    interpolated = Pose.interpolate(left, right, progress)
    pose_matrix = conversions.Rp2T(
        conversions.T2R(interpolated.to_matrix()),
        conversions.T2p(actual.to_matrix())
    )
    return Pose.from_matrix(pose_matrix, left.skel)


def save_viz(dataset):
    loader = DataLoader(dataset, batch_size=128, num_workers=4)
    print(f"Saving {dataset.motion.name[0:246]}")
    # Visualize results on untrained data
    original_motion = Motion(
        name=dataset.motion.name,
        skel=dataset.motion.skel,
        fps=dataset.motion.fps
    )
    autoencoded_motion = Motion(
        name=dataset.motion.name,
        skel=dataset.motion.skel,
        fps=dataset.motion.fps
    )

    with torch.no_grad():
        model.eval()
        flattened_pose_matrix_batch: torch.Tensor
        for flattened_pose_matrix_batch in loader:
            # we get a batch of 4x4 transform matrix for each joint

            flattened_pose_matrix_batch = flattened_pose_matrix_batch.to(device)

            output = None
            if model_type == "simple":
                output = model(flattened_pose_matrix_batch)
            elif model_type == "vae":
                mu, log_var = model.encode(flattened_pose_matrix_batch)
                reconstructed_input = model.decode(mu)
                output = reconstructed_input
            else:
                raise Exception("Unknown model type")

            flattened_pose_matrix: torch.Tensor
            for flattened_pose_matrix in output:
                pose_matrix = conversions.R2T(
                    dataset.unnormalize(flattened_pose_matrix.reshape((-1, 3, 3)).cpu())
                )
                autoencoded_motion.add_one_frame(t=None, pose_data=pose_matrix)
            for flattened_pose_matrix in flattened_pose_matrix_batch:
                pose_matrix = conversions.R2T(
                    dataset.unnormalize(flattened_pose_matrix.reshape((-1, 3, 3)).cpu())
                )
                original_motion.add_one_frame(t=None, pose_data=pose_matrix)

    motion_data.save(original_motion, f"data/generated/{model_type}/{autoencoded_motion.name}.orig.bvh")
    motion_data.save(autoencoded_motion, f"data/generated/{model_type}/{autoencoded_motion.name}.auto.bvh")


should_evaluate = True
if should_evaluate:
    model.eval()
    with torch.no_grad():
        motion = motion_data.load(motion_data.test_motion_paths()[0])
        loss = evaluate(motion, [1, 2, 4, 8, 16, 32, 64, 128])  # [1, 2, 4, 8, 16])

        motion = motion_data.load(motion_data.train_motion_paths()[0])
        loss = evaluate(motion, [1, 2, 4, 8, 16, 32, 64, 128])  # [1, 2, 4, 8, 16])

for path in motion_data.test_motion_paths()[:2]:
    save_viz(MotionDataset(motion_data.load(path), std=train_dataset.std, mean=train_dataset.mean))
for path in motion_data.train_motion_paths()[:2]:
    save_viz(MotionDataset(motion_data.load(path), std=train_dataset.std, mean=train_dataset.mean))
