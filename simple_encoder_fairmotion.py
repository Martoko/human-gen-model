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

train_dataset = MotionDataset(motion_data.load("data/motions/CMU/01/01_01_poses.npz"))
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)

generated_motion = Motion(name=train_dataset.motion.name, skel=train_dataset.motion.skel, fps=train_dataset.motion.fps)

pose: torch.Tensor
for flattened_pose_matrix_batch in train_loader:
    # we get a batch of 4x4 transform matrix for each joint

    input = flattened_pose_matrix_batch
    # manipulate pose matrix here
    output = flattened_pose_matrix_batch

    flattened_pose_matrix_batch: torch.Tensor
    for flattened_pose_matrix in output:
        generated_motion.add_one_frame(t=None, pose_data=flattened_pose_matrix.reshape((-1, 4, 4)))

motion_data.save(generated_motion, f"data/generated/{generated_motion.name}.bvh")