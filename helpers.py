# coding: utf-8
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
import yaml
import torch
import numpy as np

from torch import nn, Tensor
from dtw import dtw
from logging import Logger
from typing import Optional

class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model.
    """
    if os.path.isdir(model_dir):
        if model_continue:
            return model_dir
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production")
    return logger

def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))

def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size: int) -> Tensor:
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(path="configs/default.yaml") -> dict:
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def bpe_postprocess(string) -> str:
    return string.replace("@@ ", "")

def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    list_of_files = glob.glob("{}/*{}.ckpt".format(ckpt_dir,post_fix))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint

def freeze_params(module: nn.Module) -> None:
    for _, p in module.named_parameters():
        p.requires_grad = False

def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def calculate_dtw(references, hypotheses):
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    dtw_scores = []
    for i, ref in enumerate(references):
        _, ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)
        d = d/acc_cost_matrix.shape[0]
        dtw_scores.append(d)
    return dtw_scores


def getSkeletalModelStructure():
    return (
        (1, 0),
        (1, 1),     # center (self-edge) -> will be ignored by parent builder
        (1, 2),

        (2, 3),
        (3, 4),

        (1, 5),
        (5, 6),
        (6, 7),

        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (8, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (8, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (8, 21),
        (21, 22),
        (22, 23),
        (23, 24),

        (8, 25),
        (25, 26),
        (26, 27),
        (27, 28),

        (4, 29),
        (29, 30),
        (30, 31),
        (31, 32),
        (32, 33),

        (29, 34),
        (34, 35),
        (35, 36),
        (36, 37),

        (29, 38),
        (38, 39),
        (39, 40),
        (40, 41),

        (29, 42),
        (42, 43),
        (43, 44),
        (44, 45),

        (29, 46),
        (46, 47),
        (47, 48),
        (48, 49),
    )


# ---------------- PINN helper: build parents dict ----------------
def build_parents_from_skeleton(num_joints: int = 50) -> dict[int, int]:
    """
    Build {joint: parent_joint} from getSkeletalModelStructure().
    - root joints remain -1
    - ignores self-edges (p == c) like (1,1)
    If multiple edges assign a parent to the same child, last one wins (should not happen normally).
    """
    parents = {j: -1 for j in range(num_joints)}
    for p, c in getSkeletalModelStructure():
        p, c = int(p), int(c)
        if p == c:
            continue
        if 0 <= c < num_joints:
            parents[c] = p
    return parents


def get_hand_joint_indices() -> tuple[list, list]:
    left = list(range(8, 29))
    right = list(range(29, 50))
    return left, right

def make_joint_channel_masks(num_joints: int = 50, dims: int = 3, device="cpu"):
    """
    Returns (mask_body, mask_hand) as [1, 1, num_joints*dims] boolean tensors.
    """
    handL, handR = get_hand_joint_indices()
    hand_set = set(handL + handR)
    body_joints = [j for j in range(num_joints) if j not in hand_set]

    def joints_to_mask(jidx):
        m = torch.zeros(num_joints * dims, dtype=torch.bool, device=device)
        for j in jidx:
            start = j * dims
            m[start:start + dims] = True
        return m.view(1, 1, -1)

    mask_hand = joints_to_mask(list(hand_set))
    mask_body = joints_to_mask(body_joints)
    return mask_body, mask_hand
