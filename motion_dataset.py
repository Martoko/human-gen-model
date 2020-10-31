import torch
from fairmotion.core.motion import Motion, Pose
from torch.utils.data import Dataset


# TODO: Multiple input files
# TODO: Normalize coordinates, see the fairmotion example
class MotionDataset(Dataset):
    motion: Motion
    transform: callable

    def __init__(self, motion: Motion, transform=None):
        self.motion = motion
        self.transform = transform

    def __len__(self):
        return len(self.motion.poses)

    def __getitem__(self, i):
        pose: Pose = self.motion.poses[i]
        sample = pose.to_matrix().flatten()
        if self.transform:
            sample = self.transform(sample)
        return sample
