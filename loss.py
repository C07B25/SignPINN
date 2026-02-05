# coding: utf-8
import torch
import torch.nn as nn
from typing import Tuple, Optional

from helpers import getSkeletalModelStructure, build_parents_from_skeleton
from pinn_losses import PINNLoss, PINNConfig


# --------- Hand/Body indices and masks ---------

def _hand_joint_indices() -> Tuple[list, list]:
    left = list(range(8, 29))
    right = list(range(29, 50))
    return left, right


def _make_joint_channel_masks(num_joints: int = 50, dims: int = 3, device="cpu"):
    handL, handR = _hand_joint_indices()
    hand = set(handL + handR)
    body = [j for j in range(num_joints) if j not in hand]

    def jmask(idx):
        m = torch.zeros(num_joints * dims, dtype=torch.bool, device=device)
        for j in idx:
            m[j * dims:(j + 1) * dims] = True
        return m.view(1, 1, -1)

    return jmask(body), jmask(list(hand))


def _make_bone_masks(device="cpu"):
    """
    Bone masks aligned with get_length_direct() output:
      lengths: [B,T,num_bones]
      directs: [B,T,3*num_bones] where each bone contributes a contiguous (x,y,z) block.
    """
    skel = getSkeletalModelStructure()  # [(p,c), ...]
    num_bones = len(skel)

    handL, handR = _hand_joint_indices()
    hand_set = set(handL + handR)

    # A "hand bone" if either endpoint is in hand joints
    is_hand_bone = [((p in hand_set) or (c in hand_set)) for (p, c) in skel]

    len_hand = torch.tensor(is_hand_bone, dtype=torch.bool, device=device).view(1, 1, num_bones)
    len_body = ~len_hand

    dir_hand = torch.zeros(3 * num_bones, dtype=torch.bool, device=device)
    for i, h in enumerate(is_hand_bone):
        if h:
            dir_hand[i * 3:(i + 1) * 3] = True
    dir_hand = dir_hand.view(1, 1, -1)
    dir_body = ~dir_hand

    return len_body, len_hand, dir_body, dir_hand


def get_length_direct(trg: torch.Tensor):
    """
    trg: [B,T,150] (50 joints * 3 dims)
    Returns:
      lengths: [B,T,num_bones]
      directs: [B,T,3*num_bones] (unit vectors concatenated per bone)
    Convention:
      skeleton = (parent, child)
      vec = child - parent   (standard)
    """
    B, T, D = trg.shape
    assert D == 150, f"Expected trg last dim=150, got {D}"

    trg_reshaped = trg.view(B, T, 50, 3)                 # [B,T,J,3]
    trg_list = trg_reshaped.split(1, dim=2)              # list of [B,T,1,3]
    trg_list_squeeze = [t.squeeze(dim=2) for t in trg_list]  # list of [B,T,3]

    skeletons = getSkeletalModelStructure()  # [(p,c), ...]

    length = []
    direct = []
    tiny = torch.finfo(trg.dtype).tiny

    for (p, c) in skeletons:
        # Standard: vec = child - parent
        vec = trg_list_squeeze[c] - trg_list_squeeze[p]              # [B,T,3]
        Skeleton_length = torch.norm(vec, p=2, dim=2, keepdim=True)   # [B,T,1]
        result_length = Skeleton_length                               # [B,T,1]
        result_direct = vec / (Skeleton_length + tiny)                # [B,T,3]

        direct.append(result_direct)
        length.append(result_length)

    # stack lengths: list of [B,T,1] -> [B,T,1,num_bones] -> squeeze dim=2 -> [B,T,num_bones]
    lengths = torch.stack(length, dim=-1).squeeze(2)

    # stack directs: list of [B,T,3] -> [B,T,num_bones,3] -> reshape to [B,T,3*num_bones]
    directs = torch.stack(direct, dim=2).reshape(B, T, -1)

    return lengths, directs


class Loss(nn.Module):
    """
    Existing loss + optional PINN regularizer.

    Base:
      L = L_joint + lambda_bone * (L_bone_dir + optional L_bone_len)

    Add:
      + pinn_weight * PINN(pred_skel, mask)

    Backward compatible:
      forward(preds, targets) still works.
      forward(preds, targets, mask_bt) also works (preferred).
    """

    def __init__(self, cfg, target_pad=0.0):
        super(Loss, self).__init__()

        self.loss = cfg["training"].get("loss", "l1").lower()
        self.bone_loss = cfg["training"].get("bone_loss", "mse").lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif self.loss == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            self.criterion = nn.L1Loss(reduction="none")

        if self.bone_loss == "l1":
            self.criterion_bone = nn.L1Loss(reduction="none")
        elif self.bone_loss == "mse":
            self.criterion_bone = nn.MSELoss(reduction="none")
        else:
            self.criterion_bone = nn.MSELoss(reduction="none")

        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})

        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

        # ---- Existing weights (kept) ----
        self.w_body_joint = float(train_cfg.get("body_joint_weight", 1.0))
        self.w_hand_joint = float(train_cfg.get("hand_joint_weight", 1.0))

        self.lambda_bone = float(train_cfg.get("lambda_bone", 0.1))
        self.w_body_bonedir = float(train_cfg.get("body_bonedir_weight", 1.0))
        self.w_hand_bonedir = float(train_cfg.get("hand_bonedir_weight", 1.0))
        self.w_body_bonelen = float(train_cfg.get("body_bonelen_weight", 0.0))
        self.w_hand_bonelen = float(train_cfg.get("hand_bonelen_weight", 0.0))

        # ---- NEW: PINN config ----
        self.use_pinn = bool(train_cfg.get("use_pinn", False))
        self.pinn_weight = float(train_cfg.get("pinn_weight", 0.0))

        self.pinn = None
        if self.use_pinn and self.pinn_weight > 0.0:
            parents = build_parents_from_skeleton(num_joints=50)

            pinn_cfg = PINNConfig(
                lambda_bone=float(train_cfg.get("pinn_lambda_bone", 1.0)),
                lambda_vel=float(train_cfg.get("pinn_lambda_vel", 0.1)),
                lambda_acc=float(train_cfg.get("pinn_lambda_acc", 0.05)),
                lambda_fk=float(train_cfg.get("pinn_lambda_fk", 0.5)),
                dt=float(train_cfg.get("pinn_dt", 1.0)),
                rest_from=str(train_cfg.get("pinn_rest_from", "first_valid")),
                detach_rest=bool(train_cfg.get("pinn_detach_rest", True)),
                use_huber=bool(train_cfg.get("pinn_use_huber", True)),
                huber_delta=float(train_cfg.get("pinn_huber_delta", 1.0)),
            )
            self.pinn = PINNLoss(parents=parents, num_joints=50, cfg=pinn_cfg)

    @staticmethod
    def _weighted_mean_abs_or_sq(diff, weights, valid_mask, kind: str):
        if kind == "l1":
            e = torch.abs(diff)
        else:
            e = diff ** 2

        w = (weights * valid_mask).float()
        denom = w.sum().clamp_min(1e-9)
        return (e * w).sum() / denom

    def forward(self, preds, targets, mask_bt: Optional[torch.Tensor] = None):
        """
        preds, targets: [B, T, 150(+...)]
        mask_bt: (B,T) bool for valid frames (preferred). If None, inferred from targets.
        """
        device = preds.device

        preds_xyz = preds[:, :, :150]
        targets_xyz = targets[:, :, :150]

        # ---- frame validity ----
        if mask_bt is None:
            frame_valid = (targets[:, :, 0] != self.target_pad)  # (B,T)
        else:
            frame_valid = mask_bt.to(dtype=torch.bool, device=device)

        valid_150 = frame_valid.unsqueeze(-1).expand(-1, -1, 150)  # (B,T,150)

        # ---- joints weights ----
        _, mask_hand_150 = _make_joint_channel_masks(device=device)

        W_joint = (
            self.w_body_joint * (~mask_hand_150).float() +
            self.w_hand_joint * (mask_hand_150).float()
        )  # (1,1,150)

        L_joint = self._weighted_mean_abs_or_sq(
            diff=(preds_xyz - targets_xyz),
            weights=W_joint,
            valid_mask=valid_150,
            kind=self.loss
        )

        # ---- bones: length + direction ----
        pred_len, pred_dir = get_length_direct(preds_xyz)
        targ_len, targ_dir = get_length_direct(targets_xyz)

        valid_len = frame_valid.unsqueeze(-1).expand(-1, -1, pred_len.shape[-1])
        valid_dir = frame_valid.unsqueeze(-1).expand(-1, -1, pred_dir.shape[-1])

        mask_len_body, mask_len_hand, mask_dir_body, mask_dir_hand = _make_bone_masks(device=device)

        # If skeleton size differs unexpectedly, fall back to "all body" (safe)
        if mask_len_body.shape[-1] != pred_len.shape[-1]:
            mask_len_hand = torch.zeros(1, 1, pred_len.shape[-1], dtype=torch.bool, device=device)
            mask_len_body = ~mask_len_hand
        if mask_dir_body.shape[-1] != pred_dir.shape[-1]:
            mask_dir_hand = torch.zeros(1, 1, pred_dir.shape[-1], dtype=torch.bool, device=device)
            mask_dir_body = ~mask_dir_hand

        W_len = (
            self.w_body_bonelen * (~mask_len_hand).float() +
            self.w_hand_bonelen * (mask_len_hand).float()
        )
        W_dir = (
            self.w_body_bonedir * (~mask_dir_hand).float() +
            self.w_hand_bonedir * (mask_dir_hand).float()
        )

        L_bone_dir = self._weighted_mean_abs_or_sq(
            diff=(pred_dir - targ_dir),
            weights=W_dir,
            valid_mask=valid_dir,
            kind=self.bone_loss
        )

        if (self.w_body_bonelen > 0.0 or self.w_hand_bonelen > 0.0):
            L_bone_len = self._weighted_mean_abs_or_sq(
                diff=(pred_len - targ_len),
                weights=W_len,
                valid_mask=valid_len,
                kind=self.bone_loss
            )
        else:
            L_bone_len = torch.zeros((), device=device)

        L_bone = L_bone_dir + L_bone_len

        loss = L_joint + self.lambda_bone * L_bone

        # ---- NEW: PINN regularizer ----
        if self.pinn is not None:
            skel_pred = preds_xyz.view(preds_xyz.shape[0], preds_xyz.shape[1], 50, 3)
            pinn_dict = self.pinn(skel_pred, mask=frame_valid)
            loss = loss + self.pinn_weight * pinn_dict["total"]

        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss
