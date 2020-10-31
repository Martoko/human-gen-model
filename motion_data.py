import os.path
from fairmotion.data import amass
from fairmotion.data import bvh
from fairmotion.motion.motion import Motion
from human_body_prior.body_model.body_model import BodyModel

NEUTRAL_BODY_MODEL_PATH = "data/body-models/smplh/neutral/model.npz"
FEMALE_BODY_MODEL_PATH = "data/body-models/smplh/female/model.npz"
MALE_BODY_MODEL_PATH = "data/body-models/smplh/male/model.npz"

NEUTRAL_BODY_MODEL = BodyModel(model_type="smplh", bm_path=NEUTRAL_BODY_MODEL_PATH, num_betas=10)
FEMALE_BODY_MODEL = BodyModel(model_type="smplh", bm_path=FEMALE_BODY_MODEL_PATH, num_betas=10)
MALE_BODY_MODEL = BodyModel(model_type="smplh", bm_path=MALE_BODY_MODEL_PATH, num_betas=10)


def load(path: str) -> Motion:
    motion = amass.load(path, bm=NEUTRAL_BODY_MODEL)
    motion.name = os.path.splitext(os.path.basename(path))[0]
    return motion


def save(motion: Motion, path: str) -> None:
    bvh.save(motion, path)
