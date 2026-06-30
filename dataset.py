"""
dataset.py
==========
Turns the parallel corpus into batches of integer ids.

Each target sequence is wrapped as:  <bos> ... tokens ... <eos>
During training the decoder INPUT is tgt[:-1] and the LABEL is tgt[1:]
(teacher forcing / shifted-right). Source sequences get a trailing <eos>.
Everything is padded to the longest item in the batch with <pad>=0.
"""
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from config import cfg


class SPTokenizer:
    """Thin wrapper around a trained SentencePiece model."""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [cfg.bos_id] + ids
        if add_eos:
            ids = ids + [cfg.eos_id]
        return ids

    def decode(self, ids):
        ids = [i for i in ids
               if i not in (cfg.pad_id, cfg.bos_id, cfg.eos_id)]
        return self.sp.decode(ids)

    def __len__(self):
        return self.sp.get_piece_size()


class TranslationDataset(Dataset):
    def __init__(self, src_tok, tgt_tok, split="train"):
        ds = load_dataset("cfilt/iitb-english-hindi", split=split)
        # IMPORTANT: the corpus is ordered by source and the early rows are
        # mostly software/UI localization strings. Shuffle first so our subset
        # is a representative sample of general Hindi, not just UI text.
        ds = ds.shuffle(seed=cfg.seed)
        n = min(cfg.max_pairs, len(ds))
        self.pairs = []
        for i in range(n):
            t = ds[i]["translation"]
            en, hi = t["en"].strip(), t["hi"].strip()
            if en and hi:
                self.pairs.append((en, hi))
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, hi = self.pairs[idx]
        src = self.src_tok.encode(en, add_eos=True)[: cfg.max_len]
        tgt = self.tgt_tok.encode(hi, add_bos=True, add_eos=True)[: cfg.max_len]
        return torch.tensor(src), torch.tensor(tgt)


def collate(batch):
    srcs, tgts = zip(*batch)
    src = _pad(srcs)
    tgt = _pad(tgts)
    return src, tgt


def _pad(seqs):
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), cfg.pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def build_dataloader(shuffle=True, split="train"):
    src_tok = SPTokenizer(cfg.src_spm)
    tgt_tok = SPTokenizer(cfg.tgt_spm)
    ds = TranslationDataset(src_tok, tgt_tok, split=split)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                    collate_fn=collate)
    return dl, src_tok, tgt_tok
