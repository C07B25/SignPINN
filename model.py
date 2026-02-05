# coding: utf-8
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, Optional

from encoder import Encoder
from ACD import ACD
from batch import Batch
from embeddings import Embeddings
from vocabulary import Vocabulary
from initialization import initialize_model
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD, UNK_TOKEN


def _get_stoi_mapping(vocab: Any) -> Dict[str, int]:
    """
    Return a token->index mapping for several vocab implementations:
      - objects with get_stoi()
      - objects with .stoi (dict-like or mapping)
      - plain dicts
      - objects supporting __getitem__ (try to extract PAD/UNK)
    Returns an empty dict on failure.
    """
    if vocab is None:
        return {}

    # 1) custom Vocabulary with get_stoi()
    try:
        if hasattr(vocab, "get_stoi") and callable(getattr(vocab, "get_stoi")):
            m = vocab.get_stoi()
            if isinstance(m, dict):
                return m
    except Exception:
        pass

    # 2) TorchText-like object with .stoi attribute (dict or mapping)
    try:
        if hasattr(vocab, "stoi"):
            stoi = getattr(vocab, "stoi")
            if callable(stoi):
                stoi = stoi()
            if isinstance(stoi, dict):
                return stoi
            try:
                return dict(stoi)
            except Exception:
                pass
    except Exception:
        pass

    # 3) If it's a plain dict
    try:
        if isinstance(vocab, dict):
            return vocab
    except Exception:
        pass

    # 4) Try basic __getitem__ access to extract PAD/UNK indices
    try:
        pad_idx = vocab[PAD_TOKEN]
        mapping = {PAD_TOKEN: pad_idx}
        try:
            mapping[UNK_TOKEN] = vocab[UNK_TOKEN]
        except Exception:
            pass
        return mapping
    except Exception:
        pass

    # fallback empty mapping
    return {}


def _get_vocab_size(vocab: Any) -> int:
    """
    Determine vocabulary size in a robust way.
    Tries len(vocab), len(vocab.stoi), or highest index+1 from stoi dict.
    Falls back to 0.
    """
    try:
        sz = len(vocab)
        return sz
    except Exception:
        pass

    try:
        if hasattr(vocab, "stoi"):
            stoi = getattr(vocab, "stoi")
            if callable(stoi):
                stoi = stoi()
            if isinstance(stoi, dict):
                max_idx = max(stoi.values()) if stoi else -1
                return max_idx + 1
            try:
                return len(dict(stoi))
            except Exception:
                pass
    except Exception:
        pass

    try:
        if isinstance(vocab, dict):
            max_idx = max(vocab.values()) if vocab else -1
            return max_idx + 1
    except Exception:
        pass

    return 0


class Model(nn.Module):
    def __init__(self, cfg: dict,
                 encoder: Encoder,
                 ACD: ACD,
                 src_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 in_trg_size: int,
                 out_trg_size: int):
        """
        Create Sign-IDD
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        self.ACD = ACD
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        stoi = _get_stoi_mapping(self.src_vocab)

        for tok in [BOS_TOKEN, PAD_TOKEN, EOS_TOKEN]:
            if tok not in stoi:
                raise ValueError(
                    f"Special token '{tok}' missing in vocab. "
                    f"Available tokens (sample): {list(stoi.keys())[:20]} ..."
                )

        self.bos_index = stoi[BOS_TOKEN]
        self.pad_index = stoi[PAD_TOKEN]
        self.eos_index = stoi[EOS_TOKEN]

        self.target_pad = TARGET_PAD
        self.use_cuda = cfg["training"].get("use_cuda", False) if "training" in cfg else False
        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size

    def forward(self, is_train: bool, src: Tensor, trg_input: Tensor,
                src_mask: Tensor, src_lengths: Tensor, trg_mask: Tensor):
        """ Encode source, then diffusion decode """
        encoder_output = self.encode(src=src,
                                     src_length=src_lengths,
                                     src_mask=src_mask)

        diffusion_output = self.diffusion(is_train=is_train,
                                          encoder_output=encoder_output,
                                          trg_input=trg_input,
                                          src_mask=src_mask,
                                          trg_mask=trg_mask)
        return diffusion_output

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor):
        """ Encodes the source sequence """
        return self.encoder(embed_src=self.src_embed(src),
                            src_length=src_length,
                            mask=src_mask)

    def diffusion(self, is_train: bool, encoder_output: Tensor,
                  src_mask: Tensor, trg_input: Tensor, trg_mask: Tensor):
        """ Diffusion decoding """
        return self.ACD(is_train=is_train,
                        encoder_output=encoder_output,
                        input_3d=trg_input,
                        src_mask=src_mask,
                        trg_mask=trg_mask)

    def get_loss_for_batch(self, is_train, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute batch loss.
        IMPORTANT (PINN): pass frame validity mask to loss to avoid penalizing padded frames.
        """
        skel_out = self.forward(
            src=batch.src,
            trg_input=batch.trg_input[:, :, :150],
            src_mask=batch.src_mask,
            src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask,
            is_train=is_train
        )

        # Prefer (B,T) mask if available; fallback to squeeze(1) from (B,1,T)
        mask_bt = getattr(batch, "trg_mask_bt", None)
        if mask_bt is None and getattr(batch, "trg_mask", None) is not None:
            mask_bt = batch.trg_mask.squeeze(1)

        # Loss.forward supports (preds, targets, mask_bt)
        batch_loss = loss_function(skel_out, batch.trg_input[:, :, :150], mask_bt)
        return batch_loss


def build_model(cfg: dict, src_vocab: 'Vocabulary', trg_vocab: 'Vocabulary', checkpoint: Optional[dict] = None):
    """
    Build and initialize the Sign-IDD model.
    Optionally load a checkpoint and resize embeddings if necessary.
    """
    full_cfg = cfg
    cfg_model = cfg["model"]

    src_stoi = _get_stoi_mapping(src_vocab)
    src_padding_idx = src_stoi.get(PAD_TOKEN, 0)

    if not isinstance(trg_vocab, (list, tuple)):
        trg_stoi = _get_stoi_mapping(trg_vocab)
        trg_padding_idx = trg_stoi.get(PAD_TOKEN, 0)
    else:
        trg_padding_idx = 0

    in_trg_size = cfg_model["trg_size"]
    out_trg_size = cfg_model["trg_size"]

    vocab_size = _get_vocab_size(src_vocab)
    if vocab_size == 0:
        vocab_size = max(src_stoi.values()) + 1 if src_stoi else 0

    src_embed = Embeddings(
        **cfg_model["encoder"]["embeddings"],
        vocab_size=vocab_size,
        padding_idx=src_padding_idx
    )

    enc_dropout = cfg_model["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg_model["encoder"]["embeddings"].get("dropout", enc_dropout)

    if "embedding_dim" in cfg_model["encoder"]["embeddings"] and "hidden_size" in cfg_model["encoder"]:
        assert cfg_model["encoder"]["embeddings"]["embedding_dim"] == cfg_model["encoder"]["hidden_size"], \
            "For transformer, embedding_dim must equal hidden_size"

    encoder = Encoder(
        **cfg_model["encoder"],
        emb_size=src_embed.embedding_dim,
        emb_dropout=enc_emb_dropout
    )

    diffusion = ACD(args=cfg_model, trg_vocab=trg_vocab)

    model = Model(
        cfg=full_cfg,
        encoder=encoder,
        ACD=diffusion,
        src_embed=src_embed,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        in_trg_size=in_trg_size,
        out_trg_size=out_trg_size
    )

    initialize_model(model, cfg_model, src_padding_idx, trg_padding_idx)

    if checkpoint is not None:
        state_dict = checkpoint.get("model_state", None) if isinstance(checkpoint, dict) else None
        state_dict = state_dict if state_dict is not None else (checkpoint if isinstance(checkpoint, dict) else None)

        if isinstance(state_dict, dict):
            for emb_key in ("src_embed.lut.weight", "src_embed.weight"):
                if emb_key in state_dict:
                    curr_emb = None
                    lut = getattr(model.src_embed, "lut", None)
                    if lut is not None and hasattr(lut, "weight"):
                        curr_emb = lut.weight
                    elif hasattr(model.src_embed, "weight"):
                        curr_emb = getattr(model.src_embed, "weight")

                    if curr_emb is not None:
                        if state_dict[emb_key].shape != curr_emb.shape:
                            print(f"Skipping {emb_key} due to shape mismatch")
                            state_dict.pop(emb_key, None)

            try:
                model.load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                print(f"Warning: problem loading state_dict: {e}")
        else:
            print("Warning: checkpoint provided but no valid state_dict found in provided object")

    if torch.cuda.is_available() and full_cfg.get("training", {}).get("use_cuda", True):
        model.to(torch.device("cuda"))

    return model


def load_model(checkpoint: Optional[dict], model: nn.Module, src_padding_idx: int):
    """
    Load checkpoint into model safely (used if you want a separate utility).
    """
    if checkpoint is None:
        return model

    model_state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if isinstance(model_state, dict):
        for emb_key in ("src_embed.lut.weight", "src_embed.weight"):
            if emb_key in model_state:
                curr_emb = None
                lut = getattr(model.src_embed, "lut", None)
                if lut is not None and hasattr(lut, "weight"):
                    curr_emb = lut.weight
                elif hasattr(model.src_embed, "weight"):
                    curr_emb = getattr(model.src_embed, "weight")

                if curr_emb is not None:
                    if model_state[emb_key].shape != curr_emb.shape:
                        print(f"[INFO] Checkpoint {emb_key} shape {model_state[emb_key].shape} "
                              f"does not match current model {curr_emb.shape}. Skipping.")
                        model_state.pop(emb_key, None)

        try:
            model.load_state_dict(model_state, strict=False)
        except RuntimeError as e:
            print(f"[INFO] Failed to load full checkpoint: {e}")

    return model
