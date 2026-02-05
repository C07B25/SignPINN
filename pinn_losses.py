# coding: utf-8
"""
PINN regularizers for Sign-IDD (mask-aware, stable, fast)
---------------------------------------------------------
Inputs:
  - skel: (B, T, J, 3) or (B, T, 3J)
  - mask: (B, T) boolean/0-1, True for valid frames

Loss terms:
  1) bone length consistency: enforce per-sequence rest-length consistency across time
  2) smoothness: velocity + acceleration
  3) FK consistency (meaningful): reconstruct child using parent + unit_dir * rest_len

Notes:
  - FK in your reference was identity => always 0. This file fixes that.
  - Rest length is detached by default to avoid trivial shrink-to-zero solutions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class PINNConfig:
    lambda_bone: float = 1.0
    lambda_vel: float = 0.1
    lambda_acc: float = 0.05
    lambda_fk: float = 0.5

    eps: float = 1e-8
    dt: float = 1.0

    # rest length estimation
    rest_from: str = "first_valid"     # "first_valid" or "mean_valid"
    detach_rest: bool = True           # important for stability

    # robust penalty
    use_huber: bool = True
    huber_delta: float = 1.0


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim), min=eps))


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return x.mean()
    # broadcast mask to x
    m = mask
    while m.ndim < x.ndim:
        m = m.unsqueeze(-1)
    m = m.to(dtype=x.dtype)
    num = (x * m).sum()
    den = m.sum().clamp_min(eps)
    return num / den


def _huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    ax = x.abs()
    q = torch.minimum(ax, torch.tensor(delta, device=x.device, dtype=x.dtype))
    l = ax - q
    return 0.5 * q * q + delta * l


class PINNLoss(nn.Module):
    def __init__(self, parents: Dict[int, int], num_joints: int = 50, cfg: PINNConfig = PINNConfig()):
        super().__init__()
        self.parents = parents
        self.num_joints = num_joints
        self.cfg = cfg

        child, parent = [], []
        for j in range(num_joints):
            p = parents.get(j, -1)
            if p is None or p == -1 or p == j:
                continue
            child.append(j)
            parent.append(p)

        if len(child) == 0:
            raise ValueError("PINNLoss: no valid bones from parents dict.")

        self.register_buffer("_child", torch.tensor(child, dtype=torch.long), persistent=False)
        self.register_buffer("_parent", torch.tensor(parent, dtype=torch.long), persistent=False)

    def _ensure_shape(self, skel: torch.Tensor) -> torch.Tensor:
        if skel.ndim == 3:
            B, T, D = skel.shape
            exp = 3 * self.num_joints
            if D != exp:
                raise ValueError(f"PINNLoss: expected last dim {exp}, got {D}")
            return skel.view(B, T, self.num_joints, 3)
        if skel.ndim == 4:
            if skel.shape[2] != self.num_joints or skel.shape[3] != 3:
                raise ValueError(f"PINNLoss: expected (B,T,{self.num_joints},3), got {tuple(skel.shape)}")
            return skel
        raise ValueError(f"PINNLoss: unsupported shape {tuple(skel.shape)}")

    def _bones(self, skel: torch.Tensor) -> torch.Tensor:
        # (B,T,Nb,3)
        return skel[:, :, self._child, :] - skel[:, :, self._parent, :]

    def _rest_lengths(self, bone_len: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        bone_len: (B,T,Nb)
        returns rest: (B,1,Nb)
        """
        cfg = self.cfg
        B, T, Nb = bone_len.shape

        if mask is None:
            if cfg.rest_from == "first_valid":
                rest = bone_len[:, :1, :]
            else:
                rest = bone_len.mean(dim=1, keepdim=True)
        else:
            m = mask.to(dtype=bone_len.dtype)  # (B,T)
            if cfg.rest_from == "first_valid":
                idx = (m > 0.5).float().argmax(dim=1)  # (B,)
                rest = bone_len[torch.arange(B, device=bone_len.device), idx, :].unsqueeze(1)  # (B,1,Nb)
            else:
                mt = m.unsqueeze(-1)  # (B,T,1)
                rest = (bone_len * mt).sum(dim=1, keepdim=True) / mt.sum(dim=1, keepdim=True).clamp_min(cfg.eps)

        if cfg.detach_rest:
            rest = rest.detach()
        return rest

    def bone_length_loss(self, skel: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        bones = self._bones(skel)                         # (B,T,Nb,3)
        bl = _safe_norm(bones, dim=-1, eps=cfg.eps)       # (B,T,Nb)
        rest = self._rest_lengths(bl, mask)               # (B,1,Nb)
        diff = bl - rest                                  # (B,T,Nb)

        if cfg.use_huber:
            per = _huber(diff, cfg.huber_delta)
        else:
            per = diff * diff

        return _masked_mean(per, mask, eps=cfg.eps)

    def velocity_loss(self, skel: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        vel = (skel[:, 1:] - skel[:, :-1]) / cfg.dt       # (B,T-1,J,3)
        vm = _safe_norm(vel, dim=-1, eps=cfg.eps)         # (B,T-1,J)
        if cfg.use_huber:
            vm = _huber(vm, cfg.huber_delta)

        m = None
        if mask is not None:
            m = (mask[:, 1:] & mask[:, :-1]).to(dtype=torch.bool)   # (B,T-1)
        return _masked_mean(vm, m, eps=cfg.eps)

    def acceleration_loss(self, skel: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        vel = (skel[:, 1:] - skel[:, :-1]) / cfg.dt        # (B,T-1,J,3)
        acc = (vel[:, 1:] - vel[:, :-1]) / cfg.dt          # (B,T-2,J,3)
        am = _safe_norm(acc, dim=-1, eps=cfg.eps)          # (B,T-2,J)
        if cfg.use_huber:
            am = _huber(am, cfg.huber_delta)

        m = None
        if mask is not None:
            m = (mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]).to(dtype=torch.bool)  # (B,T-2)
        return _masked_mean(am, m, eps=cfg.eps)

    def forward_kinematics_loss(self, skel: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct child: parent + unit_dir * rest_len
        Penalize ||child - recon||^2
        """
        cfg = self.cfg
        bones = self._bones(skel)                               # (B,T,Nb,3)
        bl = _safe_norm(bones, dim=-1, eps=cfg.eps)             # (B,T,Nb)
        rest = self._rest_lengths(bl, mask)                     # (B,1,Nb)

        unit = bones / bl.unsqueeze(-1).clamp_min(cfg.eps)      # (B,T,Nb,3)

        parent_pos = skel[:, :, self._parent, :]                # (B,T,Nb,3)
        recon = parent_pos + unit * rest.unsqueeze(-1)          # (B,T,Nb,3)
        true_child = skel[:, :, self._child, :]                 # (B,T,Nb,3)

        diff = true_child - recon                               # (B,T,Nb,3)
        per = (diff * diff).sum(dim=-1)                         # (B,T,Nb)

        if cfg.use_huber:
            per = _huber(per, cfg.huber_delta)

        return _masked_mean(per, mask, eps=cfg.eps)

    def forward(self, skel: torch.Tensor, mask: Optional[torch.Tensor] = None):
        skel = self._ensure_shape(skel)
        cfg = self.cfg

        L_bone = self.bone_length_loss(skel, mask)
        L_vel = self.velocity_loss(skel, mask) if cfg.lambda_vel != 0 else skel.new_tensor(0.0)
        L_acc = self.acceleration_loss(skel, mask) if cfg.lambda_acc != 0 else skel.new_tensor(0.0)
        L_fk = self.forward_kinematics_loss(skel, mask) if cfg.lambda_fk != 0 else skel.new_tensor(0.0)

        total = (
            cfg.lambda_bone * L_bone
            + cfg.lambda_vel * L_vel
            + cfg.lambda_acc * L_acc
            + cfg.lambda_fk * L_fk
        )

        return {
            "total": total,
            "bone": L_bone,
            "velocity": L_vel,
            "acceleration": L_acc,
            "fk": L_fk,
        }
