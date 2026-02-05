# coding: utf-8
# import torch.nn as nn
# import torch
# import math
# from torch import Tensor

# from helpers import freeze_params, subsequent_mask
# from transformer_layers import PositionalEncoding, TransformerDecoderLayer

# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings

# class ACD_Denoiser(nn.Module):

#     def __init__(self,
#                  num_layers: int = 2,
#                  num_heads: int = 4,
#                  hidden_size: int = 512,
#                  ff_size: int = 2048,
#                  dropout: float = 0.1,
#                  emb_dropout: float = 0.1,
#                  vocab_size: int = 1,
#                  freeze: bool = False,
#                  trg_size: int = 150,
#                  decoder_trg_trg_: bool = True,
#                  **kwargs):
#         super(ACD_Denoiser, self).__init__()

#         self.in_feature_size = trg_size + (trg_size // 3) * 4
#         self.out_feature_size = trg_size

#         self.pos_drop = nn.Dropout(p=emb_dropout)
#         self.trg_embed = nn.Linear(self.in_feature_size, hidden_size)
#         self.pe = PositionalEncoding(hidden_size, mask_count=True)
#         self.emb_dropout = nn.Dropout(p=emb_dropout)

#         if num_layers == 2:

#             self.layers_pose_condition = TransformerDecoderLayer(
#                 size=hidden_size, ff_size=ff_size, num_heads=num_heads,
#                 dropout=dropout, decoder_trg_trg=decoder_trg_trg_)

#             self.layer_norm_mid = nn.LayerNorm(hidden_size, eps=1e-6)
#             self.output_layer_mid = nn.Linear(hidden_size, self.in_feature_size, bias=False)
#             self.o1_embed = nn.Linear(trg_size, hidden_size)
#             self.o2_embed = nn.Linear((trg_size // 3) * 4, hidden_size)

#             self.layers_mha_ac = TransformerDecoderLayer(
#                 size=hidden_size, ff_size=ff_size, num_heads=num_heads,
#                 dropout=dropout, decoder_trg_trg=decoder_trg_trg_)

#         self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

#         self.time_mlp = nn.Sequential(
#             SinusoidalPositionEmbeddings(hidden_size),
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.GELU(),
#             nn.Linear(hidden_size * 2, hidden_size),
#         )

#         # Output layer to be the size of joints vector + 1 for counter (total is trg_size)
#         self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

#         if freeze:
#             freeze_params(self)

#     def forward(self,
#                 t,
#                 trg_embed: Tensor = None,
#                 encoder_output: Tensor = None,
#                 src_mask: Tensor = None,
#                 trg_mask: Tensor = None,
#                 **kwargs):

#         assert trg_mask is not None, "trg_mask required for Transformer"
#         time_embed = self.time_mlp(t)[:, None, :].repeat(1, encoder_output.shape[1], 1)
#         condition = encoder_output + time_embed
#         condition = self.pos_drop(condition)

#         trg_embed = self.trg_embed(trg_embed)
#         # add position encoding to word embedding
#         x = self.pe(trg_embed)
#         # Dropout if given
#         x = self.emb_dropout(x)

#         padding_mask = trg_mask
#         # Create subsequent mask for decoding
#         sub_mask = subsequent_mask(
#             trg_embed.size(1)).type_as(trg_mask)

#         x, h = self.layers_pose_condition(x=x, memory=condition,
#                              src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)

#         x = self.layer_norm_mid(x)
#         x = self.output_layer_mid(x)
#         o_reshaped = x.view(x.shape[0], x.shape[1], 50, 7)
#         o_1, o_2 = torch.split(o_reshaped, [3, 4], dim=-1)
#         o_1 = o_1.reshape(o_1.shape[0], o_1.shape[1], 50 * 3)
#         o_2 = o_2.reshape(o_2.shape[0], o_2.shape[1], 50 * 4)
#         o_1 = self.o1_embed(o_1)
#         o_2 = self.o2_embed(o_2)

#         x, h = self.layers_mha_ac(x=o_1, memory=o_2,
#                      src_mask=sub_mask, trg_mask=sub_mask, padding_mask=padding_mask)

#         # Apply a layer normalisation
#         x = self.layer_norm(x)
#         # Output layer turns it back into vectors of size trg_size
#         output = self.output_layer(x)

#         return output

#     def __repr__(self):
#         return "%s(num_layers=%r, num_heads=%r)" % (
#             self.__class__.__name__, len(self.layers),
#             self.layers[0].trg_trg_att.num_heads)






# # Ankita mam code
# # coding: utf-8
import torch
import torch.nn as nn
import math
from torch import Tensor

from helpers import freeze_params, subsequent_mask
from transformer_layers import PositionalEncoding, TransformerDecoderLayer


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: Tensor) -> Tensor:
        # time: [B] (long or float)
        device = time.device
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        # ensure float
        time = time.float()
        # [B, half_dim]
        angles = time[:, None] * freq[None, :]
        # [B, dim]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ACD_Denoiser(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_size: int = 512,
        ff_size: int = 2048,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        vocab_size: int = 1,
        freeze: bool = False,
        trg_size: int = 150,
        decoder_trg_trg_: bool = True,
        **kwargs
    ):
        super(ACD_Denoiser, self).__init__()

        # remember for repr
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Input features = joints (trg_size=150) + iconicity/bone dir+len (50*4)
        # total in_feature_size = 150 + 200 = 350 (= 50 * 7)
        self.in_feature_size = trg_size + (trg_size // 3) * 4
        self.out_feature_size = trg_size

        # Embedding for target features
        self.pos_drop = nn.Dropout(p=emb_dropout)
        self.trg_embed = nn.Linear(self.in_feature_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # Two-layer decoder stack (as in original)
        if num_layers == 2:
            self.layers_pose_condition = TransformerDecoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                decoder_trg_trg=decoder_trg_trg_,
            )

            self.layer_norm_mid = nn.LayerNorm(hidden_size, eps=1e-6)
            self.output_layer_mid = nn.Linear(hidden_size, self.in_feature_size, bias=False)
            self.o1_embed = nn.Linear(trg_size, hidden_size)                 # joints part (50*3)
            self.o2_embed = nn.Linear((trg_size // 3) * 4, hidden_size)      # bones part (50*4)

            self.layers_mha_ac = TransformerDecoderLayer(
                size=hidden_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                decoder_trg_trg=decoder_trg_trg_,
            )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # --- time embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        # NEW: small projector to inject [sigma_B, sigma_H] (2 scalars) into the time embedding
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Output head -> predict x0 joints (trg_size)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        t: Tensor,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        sigma_B: Tensor = None,   # NEW (optional): [B]
        sigma_H: Tensor = None,   # NEW (optional): [B]
        **kwargs,
    ) -> Tensor:

        assert trg_mask is not None, "trg_mask required for Transformer"

        # --- time conditioning ---
        # base time embedding: [B, hidden]
        t_base = self.time_mlp(t)
        # add two-rate noise indicators; default to zeros for backward-compat
        if sigma_B is None or sigma_H is None:
            # type/shape safety
            sigma_B = torch.zeros_like(t, dtype=t_base.dtype)
            sigma_H = torch.zeros_like(t, dtype=t_base.dtype)
        # concat and project back to hidden
        t_aug = torch.stack([sigma_B, sigma_H], dim=-1)            # [B, 2]
        t_cond = self.time_proj(torch.cat([t_base, t_aug], dim=-1))  # [B, hidden]
        # broadcast over time dimension of encoder_output
        time_embed = t_cond[:, None, :].repeat(1, encoder_output.shape[1], 1)

        # conditioning: encoder outputs + time embedding
        condition = encoder_output + time_embed
        condition = self.pos_drop(condition)

        # target stream
        trg_embed = self.trg_embed(trg_embed)
        x = self.pe(trg_embed)
        x = self.emb_dropout(x)

        padding_mask = trg_mask
        # causal mask for target self-attn
        sub_mask = subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        # cross-attend target stream to conditioning
        x, _ = self.layers_pose_condition(
            x=x,
            memory=condition,
            src_mask=src_mask,
            trg_mask=sub_mask,
            padding_mask=padding_mask,
        )

        # mid projection to split (joints vs bones) and re-embed
        x = self.layer_norm_mid(x)
        x = self.output_layer_mid(x)                     # [B,T,350]
        o_reshaped = x.view(x.shape[0], x.shape[1], 50, 7)
        o_1, o_2 = torch.split(o_reshaped, [3, 4], dim=-1)   # joints(3) vs bones(4)
        o_1 = o_1.reshape(o_1.shape[0], o_1.shape[1], 50 * 3)
        o_2 = o_2.reshape(o_2.shape[0], o_2.shape[1], 50 * 4)
        o_1 = self.o1_embed(o_1)
        o_2 = self.o2_embed(o_2)

        # second decoder layer mixes the two streams
        x, _ = self.layers_mha_ac(
            x=o_1,
            memory=o_2,
            src_mask=sub_mask,
            trg_mask=sub_mask,
            padding_mask=padding_mask,
        )

        # final norm + linear head -> joints x0
        x = self.layer_norm(x)
        output = self.output_layer(x)  # [B,T,150]

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(num_layers={self.num_layers}, num_heads={self.num_heads})"
