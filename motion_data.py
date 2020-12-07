import math
import os.path
import pickle
from typing import List

from fairmotion.data import amass
from fairmotion.data import bvh
from fairmotion.motion.motion import Motion, Pose
from fairmotion.ops import motion as motion_ops
from human_body_prior.body_model.body_model import BodyModel

NEUTRAL_BODY_MODEL_PATH = "data/body-models/smplh/neutral/model.npz"
FEMALE_BODY_MODEL_PATH = "data/body-models/smplh/female/model.npz"
MALE_BODY_MODEL_PATH = "data/body-models/smplh/male/model.npz"

NEUTRAL_BODY_MODEL = BodyModel(model_type="smplh", bm_path=NEUTRAL_BODY_MODEL_PATH, num_betas=10)
FEMALE_BODY_MODEL = BodyModel(model_type="smplh", bm_path=FEMALE_BODY_MODEL_PATH, num_betas=10)
MALE_BODY_MODEL = BodyModel(model_type="smplh", bm_path=MALE_BODY_MODEL_PATH, num_betas=10)


def all_motion_paths() -> List[str]:
    paths = []
    for path, directory_names, file_names in os.walk("data/motions"):
        for file_name in file_names:
            if file_name[-4:] == ".npz":
                paths.append(os.path.join(path, file_name))
    paths.sort()
    return paths


def train_motion_paths() -> List[str]:
    paths = all_motion_paths()
    return paths[:math.floor(len(paths) / 10 * 9)]


def test_motion_paths() -> List[str]:
    paths = all_motion_paths()
    return paths[math.floor(len(paths) / 10 * 9):]


def load_train() -> Motion:
    # TODO: Loading all data (as opposed to only a few entries) is an overnight task
    if os.path.exists("data/pre-processed-motions/train.bin"):
        print("Loading cached train data...")
        with open("data/pre-processed-motions/train.bin", "rb") as file:
            motion = pickle.load(file)
    else:
        print("Loading train data...")
        motion = load_multiple(train_motion_paths()[:100])

        print("Saving cached copy of train data...")
        if not os.path.exists("data/pre-processed-motions"):
            os.mkdir("data/pre-processed-motions")
        with open("data/pre-processed-motions/train.bin", "wb") as file:
            pickle.dump(motion, file)
    return motion


def load_test() -> Motion:
    # TODO: Loading all data (as opposed to only a few entries) is an overnight task
    if os.path.exists("data/pre-processed-motions/test.bin"):
        print("Loading cached test data...")
        with open("data/pre-processed-motions/test.bin", "rb") as file:
            motion = pickle.load(file)
    else:
        print("Loading test data...")
        motion = load_multiple(test_motion_paths()[:10])

        print("Saving cached copy of test data...")
        if not os.path.exists("data/pre-processed-motions"):
            os.mkdir("data/pre-processed-motions")
        with open("data/pre-processed-motions/test.bin", "wb") as file:
            pickle.dump(motion, file)
    return motion


def load_multiple(paths: List[str]) -> Motion:
    motion = None

    for i, path in enumerate(paths):
        print(f"Loading motions {i + 1}/{len(paths)}...")
        motion = load(path) if motion is None else motion_ops.append(motion, load(path))

    return motion


def load(path: str) -> Motion:
    motion = amass.load(path, bm=NEUTRAL_BODY_MODEL)
    motion.name = os.path.splitext(os.path.basename(path))[0]
    return motion


def save(motion: Motion, path: str) -> None:
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    bvh.save(motion, path)
