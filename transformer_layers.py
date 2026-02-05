# coding: utf-8
"""
Fixed and modernized transformer/attention layers.

Key fixes and improvements:
- Robust mask broadcasting for both causal/attention masks and padding masks.
- Apply both key padding masks and optional attention masks to scores BEFORE softmax
  (using a large negative value instead of -inf to avoid NaNs).
- Consistent tensor shapes and comments.
- Corrected positional encoding shapes and sinusoidal implementation.
- Defensive programming with clear error messages when masks cannot be broadcast.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from constants import TARGET_PAD


_LARGE_NEG = -1e9  # used instead of -inf to avoid NaNs when entire row is masked


def _prepare_mask_for_scores(mask: Tensor, scores: Tensor) -> Tensor:
    """
    Turn a mask (which may be 2D: [B, Lk] or [B, Lq] or 3D [B, Lq, Lk]) into a boolean
    tensor that can be broadcasted against `scores` (shape [B, num_heads, Lq, Lk]).

    Returned mask has shape that can broadcast to scores; True means **keep**.
    """
    if mask is None:
        return None

    # Ensure boolean
    mask = mask.bool()

    # scores shape: [B, H, Lq, Lk]
    _, _, Lq, Lk = scores.shape

    # If mask is 2D (B, Lk) -> key padding mask: expand to (B, 1, 1, Lk)
    if mask.dim() == 2 and mask.size(1) == Lk:
        return mask.unsqueeze(1).unsqueeze(2)

    # If mask is 2D (B, Lq) -> query mask: expand to (B, 1, Lq, 1)
    if mask.dim() == 2 and mask.size(1) == Lq:
        return mask.unsqueeze(1).unsqueeze(-1)

    # If mask is 3D and matches (B, Lq, Lk)
    if mask.dim() == 3 and mask.size(1) == Lq and mask.size(2) == Lk:
        return mask.unsqueeze(1)  # -> (B,1,Lq,Lk)

    # If mask is 4D and already matches scores shape (B,H,Lq,Lk) return as-is
    if mask.dim() == 4 and mask.shape == scores.shape:
        return mask

    # try to be flexible: add singleton dims on the left until it can broadcast
    while mask.dim() < 4:
        mask = mask.unsqueeze(1)

    # now check if last two dims can broadcast with scores' last two dims
    if not (mask.size(-2) in (1, Lq) and mask.size(-1) in (1, Lk)):
        raise RuntimeError(f"Cannot broadcast mask of shape {mask.shape} to attention scores shape {scores.shape}")

    return mask


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0, "`size` must be divisible by `num_heads`"

        self.head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * self.head_size)
        self.v_layer = nn.Linear(size, num_heads * self.head_size)
        self.q_layer = nn.Linear(size, num_heads * self.head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.target_pad = TARGET_PAD

    def forward(self,
                k: Tensor,
                v: Tensor,
                q: Tensor,
                mask: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        k: [B, Lk, D]
        v: [B, Lk, D]
        q: [B, Lq, D]

        mask: optional attention mask (causal or custom). Typical shapes:
            - [B, Lq, Lk] OR
            - [B, Lq] OR
            - [Lq, Lk] (will be broadcasted across batch)

        padding_mask: optional key padding mask with shape [B, Lk] (True where token is VALID)

        Returns: context vectors of shape [B, Lq, D]
        """

        batch_size = q.size(0)

        # project to multi-head space
        k_proj = self.k_layer(k)  # [B, Lk, H*hs]
        v_proj = self.v_layer(v)
        q_proj = self.q_layer(q)

        # reshape -> [B, H, L, hs]
        def _reshape(x):
            B, L, _ = x.size()
            return x.view(B, L, self.num_heads, self.head_size).transpose(1, 2)

        k_heads = _reshape(k_proj)
        v_heads = _reshape(v_proj)
        q_heads = _reshape(q_proj)

        # scale queries
        q_heads = q_heads / math.sqrt(self.head_size)

        # compute attention scores [B, H, Lq, Lk]
        scores = torch.matmul(q_heads, k_heads.transpose(2, 3))

        # Prepare masks for scores and apply before softmax to zero out on softmax
        # Convert both mask (causal/custom) and padding_mask (keys) into broadcastable masks
        combined_mask = None

        if padding_mask is not None:
            # padding_mask: True where token is VALID -> we want keep True
            key_mask = _prepare_mask_for_scores(padding_mask, scores)
            combined_mask = key_mask if combined_mask is None else (combined_mask & key_mask)

        if mask is not None:
            att_mask = _prepare_mask_for_scores(mask, scores)
            combined_mask = att_mask if combined_mask is None else (combined_mask & att_mask)

        if combined_mask is not None:
            # combined_mask True means KEEP, so invert for masked_fill
            scores = scores.masked_fill(~combined_mask, _LARGE_NEG)

        # softmax
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # guard against any NaNs that could arise when rows are all -LARGE_NEG
        if torch.isnan(attention).any():
            attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)

        # compute context [B, H, Lq, hs]
        context = torch.matmul(attention, v_heads)

        # reshape back to [B, Lq, H*hs]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)

        output = self.output_layer(context)
        return output


class PositionwiseFeedForward(nn.Module):

    def __init__(self, input_size, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings for diffusion/time embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        # time: [B] or [B, 1]
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        exponents = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float) * -emb)
        # time may be [B] -> make [B,1]
        time = time.float().unsqueeze(1)
        args = time * exponents.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class PositionalEncoding(nn.Module):

    def __init__(self, size: int = 0, max_len: int = 200000, mask_count: bool = False):
        super(PositionalEncoding, self).__init__()
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dim")

        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, size]

        self.register_buffer('pe', pe)
        self.dim = size
        self.mask_count = mask_count

    def forward(self, emb: Tensor) -> Tensor:
        # emb: [B, L, D]
        L = emb.size(1)
        return emb + self.pe[:, :L, :]


class TransformerEncoderLayer(nn.Module):

    def __init__(self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask=mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class TransformerDecoderLayer(nn.Module):

    def __init__(self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1, decoder_trg_trg: bool = True):
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        self.decoder_trg_trg = decoder_trg_trg

    def forward(self,
                x: Tensor = None,
                memory: Tensor = None,
                src_mask: Optional[Tensor] = None,
                trg_mask: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None) -> (Tensor, Tensor):

        # decoder/target self-attention
        h1 = self.x_layer_norm(x)

        # Target-Target Self Attention (causal + padding handled by masks)
        if self.decoder_trg_trg:
            h1 = self.trg_trg_att(h1, h1, h1, mask=trg_mask, padding_mask=padding_mask)
        h1 = self.dropout(h1) + x

        # Source-Target Attention: keys/values from memory (encoder output), queries from h1
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask, padding_mask=None)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o, h2
