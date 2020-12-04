from typing import List, Optional, Any, Union
import torch
from fairmotion.core.motion import Motion, Pose
from torch.utils.data import Dataset
from fairmotion.ops import motion as motion_ops
import numpy as np
from fairmotion.utils import constants


# TODO: Normalize coordinates, see the fairmotion example https://github.com/facebookresearch/fairmotion/blob/509035c7e9a90046bd05ebc8ca1c254d51e08e28/fairmotion/tasks/motion_prediction/utils.py#L98
# TODO: AngleAxis would be nice to have
class MotionDataset(Dataset):
    motion: Motion
    pose_matrices: np.ndarray
    mean: float
    std: float

    def __init__(self, motion: Motion, mean: Optional[float] = None, std: Optional[float] = None):
        print("Getting rotations of the motion...")
        self.motion = motion
        self.pose_matrices = motion.rotations()

        print("Calculating mean and std...")
        self.mean = np.mean(self.pose_matrices, axis=(0, 1)) if mean is None else mean
        self.std = np.std(self.pose_matrices, axis=(0, 1)) if std is None else std

        # Ideas for how to normalize:
        # 1. Normalize across all values
        # 2. Normalize across each matrix
        # 3. Normalize across each matrix, but separate the root matrix and the rest
        # 4. Normalize across each matrix but with each matrix in world space
        # 5. Only use local rotations, no positional values

        print("MEAN")
        print(self.mean)
        print("STD")
        print(self.std)

    def __len__(self):
        return len(self.pose_matrices)

    def __getitem__(self, i):
        return self.normalize(self.pose_matrices[i]).flatten()

    def normalize(self, matrix):
        return (matrix - self.mean) / (self.std + constants.EPSILON)

    def unnormalize(self, matrix):
        return matrix * (self.std + constants.EPSILON) + self.mean
