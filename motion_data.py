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

# [SUBJECT, TRIAL, TAGS]
WALKING_LIKE_MOTIONS_INFO = [
    [2, 1, ["walk"]],
    [2, 2, ["walk"]],
    [5, 1, ["walk"]],
    [6, 1, ["walk"]],
    [7, 1, ["walk"]],
    [7, 2, ["walk"]],
    [7, 3, ["walk"]],
    [7, 4, ["slow walk"]],
    [7, 5, ["slow walk"]],
    [7, 6, ["walk"]],
    [7, 7, ["walk"]],
    [7, 8, ["walk"]],
    [7, 9, ["walk"]],
    [7, 10, ["walk"]],
    [7, 11, ["walk"]],
    [7, 12, ["brisk walk"]],
    [8, 1, ["walk"]],
    [8, 2, ["walk"]],
    [8, 3, ["walk"]],
    [8, 4, ["slow walk"]],
    [8, 5, ["walk/stride"]],
    [8, 6, ["walk"]],
    [8, 7, ["walk", "exaggerated stride"]],
    [8, 8, ["walk"]],
    [8, 9, ["walk"]],
    [8, 10, ["walk"]],
    [8, 11, ["slow walk/stride"]],
    [9, 12, ["navigate - walk forward", "backward", "sideways"]],
    [10, 4, ["walk"]],
    [12, 1, ["walk"]],
    [12, 2, ["walk"]],
    [12, 3, ["walk"]],
    [15, 1, ["walk/wander"]],
    [15, 3, ["walk/wander"]],
    [15, 9, ["walk/wander"]],
    [15, 14, ["walk/wander"]],
    [16, 11, ["walk", "veer left"]],
    [16, 12, ["walk", "veer left"]],
    [16, 13, ["walk", "veer right"]],
    [16, 14, ["walk", "veer right"]],
    [16, 15, ["walk"]],
    [16, 16, ["walk"]],
    [16, 17, ["walk", "90-degree left turn"]],
    [16, 18, ["walk", "90-degree left turn"]],
    [16, 19, ["walk", "90-degree right turn"]],
    [16, 20, ["walk", "90-degree right turn"]],
    [16, 21, ["walk"]],
    [16, 22, ["walk"]],
    [16, 23, ["walk", "veer left"]],
    [16, 24, ["walk", "veer left"]],
    [16, 25, ["walk", "veer right"]],
    [16, 26, ["walk", "veer right"]],
    [16, 27, ["walk", "90-degree left turn"]],
    [16, 28, ["walk", "90-degree left turn"]],
    [16, 29, ["walk", "90-degree right turn"]],
    [16, 30, ["walk", "90-degree right turn"]],
    [16, 31, ["walk"]],
    [16, 32, ["walk"]],
    [16, 33, ["slow walk", "stop"]],
    [16, 34, ["slow walk", "stop"]],
    [16, 47, ["walk"]],
    [16, 58, ["walk"]],
    [17, 1, ["walk with anger", "frustration"]],
    [17, 2, ["walk with anger", "frustration"]],
    [17, 3, ["walk stealthily"]],
    [17, 4, ["walk stealthily"]],
    [17, 5, ["walk/hobble"]],
    [17, 6, ["whistle", "walk jauntily"]],
    [17, 7, ["whistle", "walk jauntily"]],
    [17, 8, ["muscular", "heavyset person's walk"]],
    [17, 9, ["muscular", "heavyset person's walk"]],
    [26, 1, ["walk"]],
    [27, 1, ["walk"]],
    [29, 1, ["walk"]],
    [32, 1, ["walk"]],
    [32, 2, ["walk"]],
    [35, 1, ["walk"]],
    [35, 2, ["walk"]],
    [35, 3, ["walk"]],
    [35, 4, ["walk"]],
    [35, 5, ["walk"]],
    [35, 6, ["walk"]],
    [35, 7, ["walk"]],
    [35, 8, ["walk"]],
    [35, 9, ["walk"]],
    [35, 10, ["walk"]],
    [35, 11, ["walk"]],
    [35, 12, ["walk"]],
    [35, 13, ["walk"]],
    [35, 14, ["walk"]],
    [35, 15, ["walk"]],
    [35, 16, ["walk"]],
    [35, 28, ["walk"]],
    [35, 29, ["walk"]],
    [35, 30, ["walk"]],
    [35, 31, ["walk"]],
    [35, 32, ["walk"]],
    [35, 33, ["walk"]],
    [35, 34, ["walk"]],
    [36, 2, ["walk forward", "turn around", "walk back"]],
    [36, 3, ["walk forward", "turn around", "walk back"]],
    [36, 9, ["walk forward", "turn around", "walk back"]],
    [37, 1, ["slow walk"]],
    [38, 1, ["walk"]],
    [38, 2, ["walk"]],
    [38, 4, ["walk around", "frequent turns", "cyclic walk along a line"]],
    [39, 1, ["walk"]],
    [39, 2, ["walk"]],
    [39, 3, ["walk"]],
    [39, 4, ["walk"]],
    [39, 5, ["walk"]],
    [39, 6, ["walk"]],
    [39, 7, ["walk"]],
    [39, 8, ["walk"]],
    [39, 9, ["walk"]],
    [39, 10, ["walk"]],
    [39, 11, ["walk forward", "turn around", "walk back"]],
    [39, 12, ["walk"]],
    [39, 13, ["walk"]],
    [39, 14, ["walk"]],
    [40, 2, ["navigate - walk forward", "backward", "on a diagonal"]],
    [40, 3, ["navigate - walk forward", "backward", "on a diagonal"]],
    [40, 4, ["navigate - walk forward", "backward", "on a diagonal"]],
    [40, 5, ["navigate - walk forward", "backward", "on a diagonal"]],
    [41, 2, ["navigate - walk forward", "backward", "sideways", "on a diagonal"]],
    [41, 3, ["navigate - walk forward", "backward", "sideways", "on a diagonal"]],
    [41, 4, ["navigate - walk forward", "backward", "on a diagonal"]],
    [41, 5, ["navigate - walk forward", "backward", "on a diagonal"]],
    [41, 6, ["navigate - walk forward", "backward", "on a diagonal"]],
    [43, 1, ["walk"]],
    [45, 1, ["walk"]],
    [46, 1, ["walk"]],
    [47, 1, ["walk forward", "turn around", "walk back"]],
    [49, 1, ["walk"]],
    [55, 4, ["walk"]],
    [56, 1, ["walk around"]]
]

WALKING_LIKE_MOTIONS_PATH = [f"data/motions/CMU/{subject:02d}/{subject:02d}_{trial:02d}_poses.npz" for
                             subject, trial, tags in WALKING_LIKE_MOTIONS_INFO]
for path in WALKING_LIKE_MOTIONS_PATH:
    if not os.path.exists(path):
        print(f"WARNING: Cannot find motion at {path}")
        continue
WALKING_LIKE_MOTIONS_PATH = [path for path in WALKING_LIKE_MOTIONS_PATH if os.path.exists(path)]


def all_motion_paths() -> List[str]:
    paths = []
    for path, directory_names, file_names in os.walk("data/motions"):
        for file_name in file_names:
            if file_name[-4:] == ".npz":
                paths.append(os.path.join(path, file_name))
    paths.sort()
    return paths


def train_motion_paths() -> List[str]:
    paths = WALKING_LIKE_MOTIONS_PATH
    return paths[:math.floor(len(paths) / 10 * 9)]


def test_motion_paths() -> List[str]:
    paths = WALKING_LIKE_MOTIONS_PATH
    return paths[math.floor(len(paths) / 10 * 9):]


def load_train() -> Motion:
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

    print(f"Loaded {motion.length()} seconds of motion")

    return motion


def load(path: str) -> Motion:
    motion = amass.load(path, bm=NEUTRAL_BODY_MODEL)
    motion.name = os.path.splitext(os.path.basename(path))[0]
    return motion


def save(motion: Motion, path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "" and not os.path.exists(dir):
        os.mkdir(dir)
    bvh.save(motion, path)
