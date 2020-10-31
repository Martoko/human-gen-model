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
import motion_data

motion = motion_data.load("data/motions/CMU/01/01_01_poses.npz")

len(motion.rotations())  # 2751 framecount
motion.num_frames()  # 2751 framecount

len(motion.rotations()[0])  # 22 joint count
len(motion.skel.joints)  # 22 joint count
len(motion.rotations()[0][0])  # 3x3 matrix of rotation

generated_motion = Motion(name=motion.name, skel=motion.skel, fps=motion.fps)

pose: Pose
for pose in motion.poses:
    pose_matrix = pose.to_matrix()

    # manipulate pose matrix here

    generated_motion.add_one_frame(t=None, pose_data=pose_matrix)

motion_data.save(generated_motion, f"data/generated/{generated_motion.name}.bvh")
