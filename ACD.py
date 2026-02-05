# # coding: utf-8
# import math
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import torch.nn as nn
# import numpy as np
# from encoder import Encoder
# from ACD_Denoiser import ACD_Denoiser
# from embeddings import Embeddings
# from prediction import validate_on_data
# from helpers import make_model_dir, make_logger, ConfigurationError, get_latest_checkpoint, load_checkpoint, symlink_update, load_config, set_seed, log_cfg
# from typing import Optional, List, Tuple
# from constants import PAD_TOKEN, TARGET_PAD
# import torch.nn.functional as F
#
# from collections import namedtuple
# from tqdm.auto import tqdm
#
# ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
#
# def exists(x):
#     return x is not None
#
# def default(val, d):
#     if exists(val):
#         return val
#     return d() if callable(d) else d
#
# def extract(a, t, x_shape):
#     """extract the appropriate  t  index for a batch of indices"""
#     batch_size = t.shape[0]
#     out = a.gather(-1, t)
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
#
# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
#     alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0, 0.999)
#
#
# class ACD(nn.Module):
#     def __init__(self, args, trg_vocab):
#         super(ACD, self).__init__()
#
#         self.args = args
#         self.trg_vocab = trg_vocab
#         timesteps = args["diffusion"].get('timesteps', 1000)
#         sampling_timesteps = args["diffusion"].get('sampling_timesteps', 5)
#
#         betas = cosine_beta_schedule(timesteps)
#         alphas = 1. - betas
#         alphas_cumprod = torch.cumprod(alphas, dim=0)
#         alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
#         timesteps, = betas.shape
#
#         self.num_timesteps = int(timesteps)
#         self.sampling_timesteps = default(sampling_timesteps, timesteps)
#         assert self.sampling_timesteps <= timesteps
#         self.is_ddim_sampling = self.sampling_timesteps < timesteps
#         self.ddim_sampling_eta = 1.
#         self.self_condition = False
#         self.scale = args["diffusion"].get('scale', 1.0)
#         self.box_renewal = True
#         self.use_ensemble = True
#
#         # register buffer helper
#         register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
#
#         register_buffer('betas', betas)
#         register_buffer('alphas_cumprod', alphas_cumprod)
#         register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
#
#         register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
#         register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
#         register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
#         register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
#         register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
#
#         posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
#         register_buffer('posterior_variance', posterior_variance)
#         register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
#         register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
#         register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
#
#         self.ACD_Denoiser = ACD_Denoiser(
#             num_layers=args["diffusion"].get('num_layers', 2),
#             num_heads=args["diffusion"].get('num_heads', 4),
#             hidden_size=args["diffusion"].get('hidden_size', 512),
#             ff_size=args["diffusion"].get('ff_size', 512),
#             dropout=args["diffusion"].get('dropout', 0.1),
#             emb_dropout=args["diffusion"]["embeddings"].get('dropout', 0.1),
#             vocab_size=len(trg_vocab),
#             freeze=False,
#             trg_size=args.get('trg_size', 150),
#             decoder_trg_trg_=True
#         )

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from tqdm.auto import tqdm

from ACD_Denoiser import ACD_Denoiser
from ID import ID

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def extract(a, t, x_shape):
    """
    extract the appropriate  t  index for a batch of indices
    a: [T]
    t: [B] long
    returns [B, 1, 1] broadcastable to x_shape
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # [B]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (https://openreview.net/forum?id=-NEXDKk8gZ)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


class ACD(nn.Module):

    def __init__(self, args, trg_vocab):
        super().__init__()

        # -------- Schedules (two-rate: Body vs Hand) --------
        timesteps = args["diffusion"].get('timesteps', 1000)
        sampling_timesteps = args["diffusion"].get('sampling_timesteps', 5)
        hand_beta_scale = args["diffusion"].get('hand_beta_scale', 0.6)  # <1 => hands corrupted slower

        base_betas = cosine_beta_schedule(timesteps)       # Body schedule
        betas_B = base_betas
        betas_H = (base_betas * hand_beta_scale).clamp(max=0.999)

        def cumprod_stats(betas):
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
            return betas, alphas, alphas_cumprod, alphas_cumprod_prev

        betas_B, alphas_B, alphas_cumprod_B, alphas_cumprod_prev_B = cumprod_stats(betas_B)
        betas_H, alphas_H, alphas_cumprod_H, alphas_cumprod_prev_H = cumprod_stats(betas_H)

        timesteps, = betas_B.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args["diffusion"].get('scale', 1.0)
        self.box_renewal = True
        self.use_ensemble = True

        # register buffer helper
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # store both schedules
        register_buffer('betas_B', betas_B)
        register_buffer('alphas_cumprod_B', alphas_cumprod_B)
        register_buffer('alphas_cumprod_prev_B', alphas_cumprod_prev_B)
        register_buffer('sqrt_alphas_cumprod_B', torch.sqrt(alphas_cumprod_B))
        register_buffer('sqrt_one_minus_alphas_cumprod_B', torch.sqrt(1. - alphas_cumprod_B))
        register_buffer('sqrt_recip_alphas_cumprod_B', torch.sqrt(1. / alphas_cumprod_B))
        register_buffer('sqrt_recipm1_alphas_cumprod_B', torch.sqrt(1. / alphas_cumprod_B - 1))

        register_buffer('betas_H', betas_H)
        register_buffer('alphas_cumprod_H', alphas_cumprod_H)
        register_buffer('alphas_cumprod_prev_H', alphas_cumprod_prev_H)
        register_buffer('sqrt_alphas_cumprod_H', torch.sqrt(alphas_cumprod_H))
        register_buffer('sqrt_one_minus_alphas_cumprod_H', torch.sqrt(1. - alphas_cumprod_H))
        register_buffer('sqrt_recip_alphas_cumprod_H', torch.sqrt(1. / alphas_cumprod_H))
        register_buffer('sqrt_recipm1_alphas_cumprod_H', torch.sqrt(1. / alphas_cumprod_H - 1))

        # posterior (Body)
        posterior_variance_B = betas_B * (1. - alphas_cumprod_prev_B) / (1. - alphas_cumprod_B)
        register_buffer('posterior_variance_B', posterior_variance_B)
        register_buffer('posterior_log_variance_clipped_B', torch.log(posterior_variance_B.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1_B', betas_B * torch.sqrt(alphas_cumprod_prev_B) / (1. - alphas_cumprod_B))
        register_buffer('posterior_mean_coef2_B', (1. - alphas_cumprod_prev_B) * torch.sqrt(alphas_B) / (1. - alphas_cumprod_B))

        # posterior (Hand)
        posterior_variance_H = betas_H * (1. - alphas_cumprod_prev_H) / (1. - alphas_cumprod_H)
        register_buffer('posterior_variance_H', posterior_variance_H)
        register_buffer('posterior_log_variance_clipped_H', torch.log(posterior_variance_H.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1_H', betas_H * torch.sqrt(alphas_cumprod_prev_H) / (1. - alphas_cumprod_H))
        register_buffer('posterior_mean_coef2_H', (1. - alphas_cumprod_prev_H) * torch.sqrt(alphas_H) / (1. - alphas_cumprod_H))

        # -------- Denoiser --------
        # ✅ FIX: robust parsing so missing model.diffusion.embeddings doesn't crash
        diff_cfg = args.get("diffusion", {})
        diff_emb_cfg = diff_cfg.get("embeddings", {})
        emb_dropout = diff_emb_cfg.get("dropout", 0.1)

        self.ACD_Denoiser = ACD_Denoiser(
            num_layers=diff_cfg.get('num_layers', 2),
            num_heads=diff_cfg.get('num_heads', 4),
            hidden_size=diff_cfg.get('hidden_size', 512),
            ff_size=diff_cfg.get('ff_size', 512),
            dropout=diff_cfg.get('dropout', 0.1),
            emb_dropout=emb_dropout,
            vocab_size=len(trg_vocab),
            freeze=False,
            trg_size=args.get('trg_size', 150),
            decoder_trg_trg_=True
        )

    # ---------- Group-aware helpers ----------

    def _sigmas_pair(self, t, x_shape):
        """
        Return σ_t^B, σ_t^H as [B] scalars:
          σ_t^C := 1 - ᾱ_t^C
        """
        acB = extract(self.alphas_cumprod_B, t, x_shape).squeeze(-1).squeeze(-1)  # [B]
        acH = extract(self.alphas_cumprod_H, t, x_shape).squeeze(-1).squeeze(-1)  # [B]
        return (1. - acB).float(), (1. - acH).float()

    def predict_noise_from_start_grouped(self, x_t, t, x0):
        """
        ε̂ = (sqrt(1/ᾱ_t) * x_t - x0) / sqrt(1/ᾱ_t - 1)
        but mixed per-channel using hand/body schedules.
        """
        # NOTE: x_t, x0 in xyz space (B,T,150)
        x_shape = x_t.shape

        # build per-channel schedule factors: body for body channels, hand for hand channels
        # Here we assume ID() expands and denoiser works in iconicity space; noise prediction is in xyz.
        # This method is kept for compatibility; you may not use it directly in current pipeline.
        sqrt_recip = extract(self.sqrt_recip_alphas_cumprod_B, t, x_shape)
        sqrt_recipm1 = extract(self.sqrt_recipm1_alphas_cumprod_B, t, x_shape)
        return (sqrt_recip * x_t - x0) / (sqrt_recipm1 + 1e-8)

    # ---------- Core diffusion math (body schedule used for xyz space) ----------

    def q_sample(self, x_start, t, noise=None):
        """
        x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
        We use body schedule in xyz space (hand schedule is applied inside iconicity forward/noising if needed).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod_B, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod_B, t, x_start.shape) * noise
        )

    def model_predictions(self, is_train, x, encoder_output, t, src_mask, trg_mask):
        """
        Given current noisy sample x (B,T,150), return:
          - pred_noise (ε̂)
          - x_start (x̂₀)
        """
        x_t = ID(x)  # (B,T,50*7) iconicity / dir+len expansion
        x_t = x_t / self.scale

        # Optional: condition denoiser with σ^B, σ^H
        sigma_B, sigma_H = self._sigmas_pair(t, x.shape)  # [B], [B]

        # Call denoiser; pass σ if its forward supports it
        try:
            pred_pose = self.ACD_Denoiser(
                encoder_output=encoder_output,
                trg_embed=x_t,
                src_mask=src_mask,
                trg_mask=trg_mask,
                t=t,
                sigma_B=sigma_B,
                sigma_H=sigma_H
            )
        except TypeError:
            # Backward-compatible: older denoiser without sigma args
            pred_pose = self.ACD_Denoiser(
                encoder_output=encoder_output,
                trg_embed=x_t,
                src_mask=src_mask,
                trg_mask=trg_mask,
                t=t
            )

        # pred_pose is x_start in xyz space (expected)
        x_start = pred_pose[:, :, :150]
        pred_noise = self.predict_noise_from_start_grouped(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, is_train, x, encoder_output, t, src_mask, trg_mask):
        preds = self.model_predictions(is_train, x, encoder_output, t, src_mask, trg_mask)
        x_start = preds.pred_x_start

        model_mean = (
            extract(self.posterior_mean_coef1_B, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2_B, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance_B, t, x.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped_B, t, x.shape)

        return model_mean, posterior_variance, posterior_log_variance_clipped, x_start

    @torch.no_grad()
    def p_sample(self, is_train, x, encoder_output, t, src_mask, trg_mask):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            is_train, x=x, encoder_output=encoder_output, t=t, src_mask=src_mask, trg_mask=trg_mask
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_start

    @torch.no_grad()
    def p_sample_loop(self, is_train, encoder_output, shape, src_mask, trg_mask):
        device = self.betas_B.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(is_train, img, encoder_output, t, src_mask, trg_mask)
        return img

    @torch.no_grad()
    def ddim_sample(self, is_train, encoder_output, shape, src_mask, trg_mask):
        device = self.betas_B.device
        b = shape[0]
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='ddim sampling', total=len(time_pairs)):
            t = torch.full((b,), time, device=device, dtype=torch.long)

            preds = self.model_predictions(is_train, img, encoder_output, t, src_mask, trg_mask)
            x_start = preds.pred_x_start
            pred_noise = preds.pred_noise

            if time_next < 0:
                img = x_start
                continue

            alpha = extract(self.alphas_cumprod_B, t, img.shape)
            alpha_next = extract(self.alphas_cumprod_B, torch.full((b,), time_next, device=device, dtype=torch.long), img.shape)

            sigma = eta * torch.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            c = torch.sqrt(1 - alpha_next - sigma ** 2)

            noise = torch.randn_like(img)

            img = x_start * torch.sqrt(alpha_next) + c * pred_noise + sigma * noise

        return img

    def forward(self, is_train, encoder_output, input_3d, src_mask, trg_mask):
        """
        input_3d: (B,T,150) ground truth poses (train) or dummy (inference)
        returns:
          - during train: predicted x0 in xyz space (B,T,150)
          - during inference: sampled xyz pose (B,T,150)
        """
        if is_train:
            x_start = input_3d
            b, t, d = x_start.shape
            device = x_start.device

            t_idx = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t_idx, noise=noise)

            preds = self.model_predictions(is_train, x_noisy, encoder_output, t_idx, src_mask, trg_mask)
            return preds.pred_x_start

        # inference / sampling
        shape = input_3d.shape
        if self.is_ddim_sampling:
            return self.ddim_sample(is_train, encoder_output, shape, src_mask, trg_mask)
        return self.p_sample_loop(is_train, encoder_output, shape, src_mask, trg_mask)
