import os
from math import inf

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

import motion_data
from motion_dataset import MotionDataset
from simple_encoder import SimpleAutoEncoder
from vae_encoder import VanillaVAE

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

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

model = None
model_type = "vae"
if model_type == "simple":
    print("Setting up simple model...")
    model = SimpleAutoEncoder(input_size=198, hidden_dimensions=[128, 64, 32]).double().to(device)
    if os.path.exists(f"data/best_{model_type}.pt"):
        print("Loading saved best simple model...")
        model.load_state_dict(torch.load(f"data/best_{model_type}.pt"))
elif model_type == "vae":
    print("Setting up VAE model...")
    model = VanillaVAE(input_size=198, hidden_dimensions=[128, 64, 32]).double().to(device)
    if os.path.exists(f"data/best_{model_type}.pt"):
        print("Loading saved best VAE model...")
        model.load_state_dict(torch.load(f"data/best_{model_type}.pt"))
else:
    raise Exception("Unknown model type")
best_test_loss = -inf
best_model = model
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


def test_eval():
    model.eval()
    with torch.no_grad():
        loss = 0
        flattened_pose_matrix_batch: torch.Tensor
        for flattened_pose_matrix_batch in test_loader:
            flattened_pose_matrix_batch = flattened_pose_matrix_batch.to(device)
            if model_type == "simple":
                outputs = model(flattened_pose_matrix_batch)
                loss = criterion(outputs, flattened_pose_matrix_batch).item()
            elif model_type == "vae":
                mu, log_var = model.encode(flattened_pose_matrix_batch)
                reconstructed_input = model.decode(mu)
                losses = model.loss_function(
                    reconstructed_input, flattened_pose_matrix_batch, mu, log_var,
                    kld_weight=batch_size / len(train_dataset)
                )
                loss += losses["loss"]
            else:
                raise Exception("Unknown model type")

        return loss / len(test_loader)


saved_model_visualization = False
print("Starting training...")
epochs = 20
for epoch in range(epochs):
    model.train()
    loss = 0
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
            make_dot(outputs[0], params=dict(model.named_parameters())).save(f"data/model-visualizations/{model_type}_mnist")
            saved_model_visualization = True

        # compute training reconstruction loss
        if model_type == "simple":
            train_loss = criterion(outputs, flattened_pose_matrix_batch)
        elif model_type == "vae":
            reconstructed_input, mu, log_var = outputs
            train_loss = model.loss_function(
                reconstructed_input, flattened_pose_matrix_batch, mu, log_var,
                kld_weight=batch_size / len(train_dataset)
            )["loss"]
        else:
            raise Exception("Unknown model type")

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    test_loss = test_eval()

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model = model

    print("epoch : {}/{}, train loss = {:.6f}, test loss = {:.6f}".format(epoch + 1, epochs, loss, test_loss))

print("Saving best model...")
torch.save(best_model.state_dict(), f"data/best_{model_type}.pt")


def save_viz(dataset):
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
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
        best_model.eval()
        flattened_pose_matrix_batch: torch.Tensor
        for flattened_pose_matrix_batch in loader:
            # we get a batch of 4x4 transform matrix for each joint

            input = flattened_pose_matrix_batch.to(device)

            output = None
            if model_type == "simple":
                output = best_model(input)
            elif model_type == "vae":
                mu, log_var = best_model.encode(flattened_pose_matrix_batch)
                reconstructed_input = best_model.decode(mu)
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


for path in motion_data.test_motion_paths()[:2]:
    save_viz(MotionDataset(motion_data.load(path), std=train_dataset.std, mean=train_dataset.mean))
for path in motion_data.train_motion_paths()[:2]:
    save_viz(MotionDataset(motion_data.load(path), std=train_dataset.std, mean=train_dataset.mean))
