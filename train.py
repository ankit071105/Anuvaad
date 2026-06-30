"""
train.py
========
Step 2 of the pipeline. Trains the from-scratch Transformer.

Features:
  - device auto-select: CUDA > Apple MPS > CPU
  - Noam learning-rate schedule (warmup then decay) from the paper
  - label smoothing (KL-div loss) to avoid overconfidence
  - gradient clipping
  - saves the best checkpoint (by train loss) to ./checkpoints/transformer.pt

Run:  python train.py
"""
import math
import time
import torch
import torch.nn as nn

from config import cfg
from dataset import build_dataloader
from model import Transformer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NoamLR:
    """Learning rate ~ d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)."""
    def __init__(self, optimizer, d_model, warmup):
        self.opt = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * self.warmup ** -1.5,
        )
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def run():
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    dl, src_tok, tgt_tok = build_dataloader(shuffle=True)
    print(f"Batches/epoch: {len(dl)} | src vocab {len(src_tok)} | tgt vocab {len(tgt_tok)}")

    model = Transformer(
        src_vocab=len(src_tok), tgt_vocab=len(tgt_tok),
        d_model=cfg.d_model, num_heads=cfg.num_heads, d_ff=cfg.d_ff,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
        max_len=cfg.max_len + 2, pad_id=cfg.pad_id,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=cfg.pad_id, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, cfg.d_model, cfg.warmup_steps)

    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        for step, (src, tgt) in enumerate(dl, 1):
            src, tgt = src.to(device), tgt.to(device)
            dec_in = tgt[:, :-1]      # teacher forcing input
            label = tgt[:, 1:]        # shifted target

            logits = model(src, dec_in)               # (B, T, V)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                label.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            lr = scheduler.step()
            optimizer.step()

            total += loss.item()
            if step % 50 == 0:
                avg = total / step
                print(f"  epoch {epoch} step {step}/{len(dl)} "
                      f"loss {avg:.3f} ppl {math.exp(min(avg,20)):.1f} lr {lr:.2e}")

        avg = total / len(dl)
        print(f"Epoch {epoch} done | loss {avg:.3f} | "
              f"ppl {math.exp(min(avg,20)):.1f} | {time.time()-t0:.0f}s")

        if avg < best:
            best = avg
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "src_vocab": len(src_tok),
                "tgt_vocab": len(tgt_tok),
            }, cfg.ckpt_path)
            print(f"  saved checkpoint -> {cfg.ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    run()
