# coding: utf-8
import torch
from typing import Any, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import TARGET_PAD


class Batch:
    """
    Batch wrapper that normalizes different batch formats.
    Also provides:
      - trg_mask: [B,1,T] bool (already existed)
      - trg_mask_bt: [B,T] bool  (NEW convenience for PINN)
    """

    def __init__(self, torch_batch: Any, pad_index: int, model: Any):
        self.src = None
        self.src_lengths = None
        self.src_mask = None
        self.nseqs = 0

        self.trg = None
        self.trg_input = None
        self.trg_mask = None          # [B,1,T]
        self.trg_mask_bt = None       # [B,T]  (NEW)
        self.trg_lengths = None
        self.ntokens = 0
        self.file_paths: List[str] = []

        self.use_cuda = getattr(model, "use_cuda", False)
        self.target_pad = TARGET_PAD

        # legacy torchtext-like batch
        if hasattr(torch_batch, "src") and hasattr(torch_batch, "file_paths"):
            try:
                if isinstance(torch_batch.src, tuple) or isinstance(torch_batch.src, list):
                    self.src, self.src_lengths = torch_batch.src
                else:
                    self.src = torch_batch.src
                    self.src_lengths = getattr(
                        torch_batch, "src_lengths",
                        torch.tensor([s.size(0) for s in self.src], dtype=torch.long)
                    )
            except Exception:
                self.src = torch_batch.src
                self.src_lengths = getattr(torch_batch, "src_lengths", torch.sum(self.src != pad_index, dim=1))

            self.file_paths = list(getattr(torch_batch, "file_paths", []))
            if hasattr(torch_batch, "trg"):
                self.trg = torch_batch.trg

        # tuple from DataLoader/collate_fn
        elif isinstance(torch_batch, (tuple, list)) and len(torch_batch) >= 3:
            try:
                self.src = torch_batch[0]
                self.src_lengths = torch_batch[1]
                self.trg = torch_batch[2]
                if len(torch_batch) > 3:
                    self.file_paths = list(torch_batch[3])
                else:
                    self.file_paths = []
            except Exception:
                raise ValueError("Unrecognized tuple batch format. Expected (src, src_lengths, trg, files).")
        else:
            raise ValueError("Unrecognized batch format passed to Batch.")

        # normalize lengths
        if isinstance(self.src_lengths, int):
            self.src_lengths = torch.tensor([self.src_lengths] * self.src.size(0), dtype=torch.long)
        if self.src is not None and not isinstance(self.src_lengths, torch.Tensor):
            try:
                self.src_lengths = torch.sum(self.src != pad_index, dim=1).to(torch.long)
            except Exception:
                self.src_lengths = torch.zeros(self.src.size(0), dtype=torch.long)

        # src mask
        if self.src is not None:
            self.src_mask = (self.src != pad_index).unsqueeze(1)  # [B,1,S]
            self.nseqs = self.src.size(0)

        # targets + mask
        if self.trg is not None:
            try:
                self.trg_lengths = self.trg.shape[1]
            except Exception:
                self.trg_lengths = None

            self.trg_input = self.trg.clone()

            try:
                if self.trg.dim() == 3:
                    # fast sentinel: frame valid if first element != TARGET_PAD
                    self.trg_mask = (self.trg[:, :, 0] != self.target_pad).unsqueeze(1)  # [B,1,T]
                else:
                    self.trg_mask = torch.ones((self.trg.size(0), 1, self.trg.size(1)), dtype=torch.bool)
            except Exception:
                self.trg_mask = torch.ones((self.trg.size(0), 1, self.trg.size(1)), dtype=torch.bool)

            # NEW: [B,T] convenience
            self.trg_mask_bt = self.trg_mask.squeeze(1) if self.trg_mask is not None else None

            try:
                self.ntokens = int(torch.sum(self.trg_mask).item())
            except Exception:
                self.ntokens = 0

        if self.file_paths is None:
            self.file_paths = []

        if self.use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        if self.src is not None:
            self.src = self.src.to(device)
            self.src_mask = self.src_mask.to(device)
            self.src_lengths = self.src_lengths.to(device)

        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device)
        if self.trg is not None:
            self.trg = self.trg.to(device)
        if self.trg_mask is not None:
            self.trg_mask = self.trg_mask.to(device)
        if self.trg_mask_bt is not None:
            self.trg_mask_bt = self.trg_mask_bt.to(device)

    def to(self, device_: torch.device):
        if self.src is not None:
            self.src = self.src.to(device_)
            self.src_mask = self.src_mask.to(device_)
            self.src_lengths = self.src_lengths.to(device_)
        if self.trg_input is not None:
            self.trg_input = self.trg_input.to(device_)
        if self.trg is not None:
            self.trg = self.trg.to(device_)
        if self.trg_mask is not None:
            self.trg_mask = self.trg_mask.to(device_)
        if self.trg_mask_bt is not None:
            self.trg_mask_bt = self.trg_mask_bt.to(device_)
        return self
