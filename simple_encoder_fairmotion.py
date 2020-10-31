import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from fairmotion.data import bvh
from fairmotion.data import amass_dip
from fairmotion.data import amass
from fairmotion.ops import motion as motion_ops
from fairmotion.tasks.motion_prediction import utils
from fairmotion.utils import utils as fairmotion_utils
from human_body_prior.body_model.body_model import BodyModel

NEUTRAL_BODY_MODEL_PATH = "data/body-models/smplh/neutral/model.npz"
FEMALE_BODY_MODEL_PATH = "data/body-models/smplh/female/model.npz"
MALE_BODY_MODEL_PATH = "data/body-models/smplh/male/model.npz"

NEUTRAL_BODY_MODEL = BodyModel(model_type="smplh", bm_path=NEUTRAL_BODY_MODEL_PATH, num_betas=10)
FEMALE_BODY_MODEL = BodyModel(model_type="smplh", bm_path=FEMALE_BODY_MODEL_PATH, num_betas=10)
MALE_BODY_MODEL = BodyModel(model_type="smplh", bm_path=MALE_BODY_MODEL_PATH, num_betas=10)

motion = amass.load("data/motions/CMU/01/01_01_poses.npz", bm=FEMALE_BODY_MODEL)
bvh.save(motion, "data/generated/01_01_poses.bvh")
