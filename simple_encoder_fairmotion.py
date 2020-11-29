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
from fairmotion.utils import utils as fairmotion_utils
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import DataLoader

import motion_data
from motion_dataset import MotionDataset
from simple_encoder import SimpleAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MotionDataset(motion_data.load("data/motions/CMU/01/01_01_poses.npz"))
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

test_dataset = MotionDataset(motion_data.load("data/motions/CMU/01/01_02_poses.npz"))
test_loader = DataLoader(
    test_dataset, batch_size=32, num_workers=4, pin_memory=True
)

model = SimpleAutoEncoder(input_size=352, hidden_size=1024, encoded_size=100).double().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 10
for epoch in range(epochs):
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

        # compute training reconstruction loss
        train_loss = criterion(outputs, flattened_pose_matrix_batch)

        # compute accumulated gradients
        train_loss.backward()

        # perform parameter update based on current gradients
        optimizer.step()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()

    # compute the epoch training loss
    loss = loss / len(train_loader)

    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

def save_viz(dataset, loader):
    print(f"Saving {dataset.motion.name}")
    # Visualize results on untrained data
    original_motion = Motion(name=dataset.motion.name, skel=dataset.motion.skel, fps=dataset.motion.fps)
    autoencoded_motion = Motion(name=dataset.motion.name, skel=dataset.motion.skel, fps=dataset.motion.fps)

    with torch.no_grad():
        flattened_pose_matrix_batch: torch.Tensor
        for flattened_pose_matrix_batch in loader:
            # we get a batch of 4x4 transform matrix for each joint

            input = flattened_pose_matrix_batch.to(device)
            output = model(input)

            flattened_pose_matrix_batch: torch.Tensor
            for flattened_pose_matrix in output:
                autoencoded_motion.add_one_frame(t=None, pose_data=flattened_pose_matrix.reshape((-1, 4, 4)).cpu())
            for flattened_pose_matrix in flattened_pose_matrix_batch:
                original_motion.add_one_frame(t=None, pose_data=flattened_pose_matrix.reshape((-1, 4, 4)).cpu())

    motion_data.save(original_motion, f"data/generated/{autoencoded_motion.name}.orig.bvh")
    motion_data.save(autoencoded_motion, f"data/generated/{autoencoded_motion.name}.auto.bvh")


save_viz(train_dataset, train_loader)
save_viz(test_dataset, test_loader)
