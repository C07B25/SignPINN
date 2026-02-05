# coding: utf-8
"""
Modernized data.py (kept compatible with your current Batch)
PINN requires correct TARGET_PAD padding; this file already does it.
No structural changes required beyond keeping TARGET_PAD consistent.
"""

import io
import os
from typing import List, Tuple, Iterator

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from vocabulary import build_vocab
from constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD


def _token_to_id(vocab, token):
    try:
        return vocab[token]
    except Exception:
        try:
            return vocab.stoi[token]
        except Exception:
            try:
                return vocab.get(token)
            except Exception:
                return 0


class SignProdDataset(Dataset):
    def __init__(self, path: str, exts: Tuple[str, str, str], trg_size: int, skip_frames: int = 1):
        self.src: List[str] = []
        self.trg: List[torch.Tensor] = []
        self.files: List[str] = []

        src_path, trg_path, file_path = tuple(os.path.expanduser(path + x) for x in exts)

        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
                io.open(file_path, mode='r', encoding='utf-8') as files_file:

            for src_line, trg_line, files_line in zip(src_file, trg_file, files_file):
                src_line, trg_line, files_line = src_line.strip(), trg_line.strip(), files_line.strip()
                if not src_line or not trg_line:
                    continue

                vals = trg_line.split()
                if len(vals) == 0:
                    continue

                try:
                    trg_vals = [float(v) + 1e-8 for v in vals]
                except ValueError:
                    continue

                frames = [trg_vals[i:i + trg_size] for i in range(0, len(trg_vals), trg_size * skip_frames)]
                if len(frames) == 0:
                    continue

                self.src.append(src_line)
                self.trg.append(torch.tensor(frames, dtype=torch.float32))
                self.files.append(files_line)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.trg[idx], self.files[idx]


def collate_fn(batch, vocab, trg_size, lowercase: bool = False, return_lengths: bool = True):
    src, trg, files = zip(*batch)

    src_tok = [s.lower().split() for s in src] if lowercase else [s.split() for s in src]

    src_ids = [torch.tensor([_token_to_id(vocab, tok) for tok in sent], dtype=torch.long) for sent in src_tok]

    try:
        pad_idx = vocab[PAD_TOKEN]
    except Exception:
        try:
            pad_idx = vocab.stoi[PAD_TOKEN]
        except Exception:
            pad_idx = 0

    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_idx) if len(src_ids) > 0 \
        else torch.zeros((0, 0), dtype=torch.long)

    src_lengths = torch.tensor([len(s) for s in src_ids], dtype=torch.long)

    # IMPORTANT: pad targets with TARGET_PAD so Batch can mask correctly
    trg_padded = pad_sequence(trg, batch_first=True, padding_value=TARGET_PAD) if len(trg) > 0 \
        else torch.zeros((0, 0, trg_size), dtype=torch.float32)

    if return_lengths:
        return src_padded, src_lengths, trg_padded, list(files)
    return src_padded, trg_padded, list(files)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False,
                   vocab=None,
                   trg_size: int = 0,
                   lowercase: bool = False) -> Iterator:

    if batch_type == "token":
        batch_type = "sentence"

    actual_shuffle = True if train else shuffle

    if vocab is None and hasattr(dataset, "vocab"):
        vocab = getattr(dataset, "vocab")

    if vocab is None:
        raise ValueError("make_data_iter requires a `vocab` argument or dataset.vocab")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=actual_shuffle,
        collate_fn=lambda b: collate_fn(b, vocab, trg_size, lowercase=lowercase, return_lengths=True),
        pin_memory=False,
        drop_last=False,
    )
    return loader


def load_data(cfg: dict):
    data_cfg = cfg["data"]
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")

    train_path, dev_path, test_path = data_cfg["train"], data_cfg["dev"], data_cfg["test"]

    lowercase = data_cfg.get("lowercase", False)
    max_sent_length = data_cfg.get("max_sent_length", None)

    trg_size = cfg["model"]["trg_size"] + 1
    skip_frames = data_cfg.get("skip_frames", 1)

    train_data = SignProdDataset(train_path, ("." + src_lang, "." + trg_lang, "." + files_lang), trg_size, skip_frames)
    dev_data = SignProdDataset(dev_path, ("." + src_lang, "." + trg_lang, "." + files_lang), trg_size, skip_frames)
    test_data = SignProdDataset(test_path, ("." + src_lang, "." + trg_lang, "." + files_lang), trg_size, skip_frames)

    if max_sent_length is not None:
        def _filter_dataset(ds: SignProdDataset):
            filtered_src, filtered_trg, filtered_files = [], [], []
            for s, t, f in zip(ds.src, ds.trg, ds.files):
                tok_len = len(s.split())
                if tok_len <= max_sent_length and t.size(0) <= max_sent_length:
                    filtered_src.append(s)
                    filtered_trg.append(t)
                    filtered_files.append(f)
            ds.src, ds.trg, ds.files = filtered_src, filtered_trg, filtered_files

        _filter_dataset(train_data)
        _filter_dataset(dev_data)
        _filter_dataset(test_data)

    src_max_size = data_cfg.get("src_voc_limit", None)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)

    try:
        src_vocab = build_vocab(field="src", min_freq=src_min_freq, max_size=src_max_size or None,
                                dataset=train_data, vocab_file=src_vocab_file)
    except Exception:
        counter = {}
        for s in train_data.src:
            for tok in s.split():
                counter[tok] = counter.get(tok, 0) + 1
        items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        if src_max_size:
            items = items[:src_max_size]

        stoi = {tok: i + 2 for i, (tok, _) in enumerate(items)}
        stoi[PAD_TOKEN] = 0
        stoi[UNK_TOKEN] = 1

        class SimpleVocab:
            def __init__(self, stoi):
                self.stoi = stoi
            def __getitem__(self, token):
                return self.stoi.get(token, self.stoi.get(UNK_TOKEN, 1))

        src_vocab = SimpleVocab(stoi)

    trg_vocab = [None] * trg_size
    return train_data, dev_data, test_data, src_vocab, trg_vocab
