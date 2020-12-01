from typing import List
import torch
from fairmotion.core.motion import Motion, Pose
from torch.utils.data import Dataset
from fairmotion.ops import motion as motion_ops


# TODO: Normalize coordinates, see the fairmotion example https://github.com/facebookresearch/fairmotion/blob/509035c7e9a90046bd05ebc8ca1c254d51e08e28/fairmotion/tasks/motion_prediction/utils.py#L98
# TODO: AngleAxis would be nice to have
class MotionDataset(Dataset):
    motion: Motion
    transform: callable

    def __init__(self, motions: List[Motion]):
        combined_motion = None
        for motion in motions:
            if combined_motion is None:
                combined_motion = motion
            else:
                combined_motion = motion_ops.append(combined_motion, motion)

        self.motion = combined_motion

    def __len__(self):
        return len(self.motion.poses)

    def __getitem__(self, i):
        pose: Pose = self.motion.poses[i]
        sample = pose.to_matrix().flatten()
        return sample
