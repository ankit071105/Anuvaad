"""
translate.py
============
Loads a trained checkpoint and translates English -> Hindi.

Two decoding strategies are provided:
  - greedy_decode : fast, picks the argmax token each step
  - beam_decode   : keeps `beam_size` hypotheses, usually better output

Use from the command line:
    python translate.py "How are you?"
Or import `Translator` (the Flask app does this).
"""
import sys
import torch

from config import cfg
from dataset import SPTokenizer
from model import Transformer
from train import get_device


class Translator:
    def __init__(self, ckpt_path=None, device=None):
        ckpt_path = ckpt_path or cfg.ckpt_path
        self.device = device or get_device()
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.src_tok = SPTokenizer(cfg.src_spm)
        self.tgt_tok = SPTokenizer(cfg.tgt_spm)

        self.model = Transformer(
            src_vocab=ckpt["src_vocab"], tgt_vocab=ckpt["tgt_vocab"],
            d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff,
            num_layers=cfg.num_layers, dropout=0.0,
            max_len=cfg.max_len + 2, pad_id=cfg.pad_id,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def _encode_src(self, text):
        ids = self.src_tok.encode(text, add_eos=True)[: cfg.max_len]
        return torch.tensor([ids], device=self.device)

    @torch.no_grad()
    def greedy_decode(self, text, max_len=None):
        max_len = max_len or cfg.max_len
        src = self._encode_src(text)
        enc = self.model.encode(src)
        ys = torch.tensor([[cfg.bos_id]], device=self.device)
        for _ in range(max_len):
            logits = self.model.decode_step(ys, enc, src)
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_id], dim=1)
            if next_id.item() == cfg.eos_id:
                break
        return self.tgt_tok.decode(ys[0].tolist())

    @torch.no_grad()
    def beam_decode(self, text, beam_size=4, max_len=None, alpha=0.7):
        max_len = max_len or cfg.max_len
        src = self._encode_src(text)
        enc = self.model.encode(src)

        # each beam: (token_ids tensor, cumulative log-prob, finished?)
        beams = [(torch.tensor([[cfg.bos_id]], device=self.device), 0.0, False)]
        for _ in range(max_len):
            candidates = []
            for ys, score, done in beams:
                if done:
                    candidates.append((ys, score, True))
                    continue
                logits = self.model.decode_step(ys, enc, src)
                logp = torch.log_softmax(logits[:, -1], dim=-1)
                topk_lp, topk_id = logp.topk(beam_size, dim=-1)
                for k in range(beam_size):
                    nid = topk_id[0, k].view(1, 1)
                    nl = topk_lp[0, k].item()
                    ny = torch.cat([ys, nid], dim=1)
                    fin = nid.item() == cfg.eos_id
                    candidates.append((ny, score + nl, fin))
            # length-normalized ranking, keep best `beam_size`
            candidates.sort(
                key=lambda x: x[1] / (x[0].size(1) ** alpha), reverse=True)
            beams = candidates[:beam_size]
            if all(b[2] for b in beams):
                break

        best = max(beams, key=lambda x: x[1] / (x[0].size(1) ** alpha))
        return self.tgt_tok.decode(best[0][0].tolist())


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "How are you?"
    tr = Translator()
    print("EN :", text)
    print("HI (greedy):", tr.greedy_decode(text))
    print("HI (beam)  :", tr.beam_decode(text))
